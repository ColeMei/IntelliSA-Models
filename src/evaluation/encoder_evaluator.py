import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IacTestDataset(Dataset):
    """Dataset for IaC security smell detection test data."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Use raw content for encoder models
        text = sample['content']
        # Convert label to binary (TP=1, FP=0)
        label = 1 if sample['label'] == 'TP' else 0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'sample_data': sample  # Keep original data for analysis
        }

class EncoderEvaluator:
    """Evaluator for encoder approach using fine-tuned CodeBERT/CodeT5 for IaC security smell detection."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            num_labels=2  # TP/FP binary classification
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        logger.info(f"Model loaded successfully")
    
    def _load_test_data(self, test_path: str, max_samples: Optional[int] = None) -> IacTestDataset:
        """Load test dataset."""
        dataset = IacTestDataset(test_path, self.tokenizer)
        
        if max_samples:
            # Limit dataset size for testing
            dataset.data = dataset.data[:max_samples]
            logger.info(f"Limited evaluation to {max_samples} samples")
        
        logger.info(f"Loaded {len(dataset)} test samples")
        return dataset
    
    def _resolve_threshold(self) -> Optional[float]:
        """Resolve decision threshold from environment or files.

        Supported env vars (defaults to argmax when unset):
          - EVAL_THRESHOLD_MODE: 'argmax' | 'fixed' | 'file'
          - EVAL_THRESHOLD_FIXED: float (used when mode=fixed)
          - EVAL_THRESHOLD_FILE: path to JSON with single threshold (mode=file)

        Behavior changes for locking Stage 3/4 results:
          - When mode='file', the file MUST exist and contain a 'best_threshold' field.
            If not, raise a RuntimeError instead of silently falling back to argmax.
          - Single threshold is used for all test sets (no per-technology mapping).
        """
        mode = os.getenv("EVAL_THRESHOLD_MODE", "argmax").lower().strip()
        if mode == "fixed":
            fixed_val = os.getenv("EVAL_THRESHOLD_FIXED", "")
            try:
                return float(fixed_val)
            except ValueError as exc:
                raise RuntimeError(
                    f"EVAL_THRESHOLD_MODE=fixed but EVAL_THRESHOLD_FIXED is invalid: '{fixed_val}'"
                ) from exc
        if mode == "file":
            path = os.getenv("EVAL_THRESHOLD_FILE")
            if not path or not str(path).strip():
                raise RuntimeError(
                    "EVAL_THRESHOLD_MODE=file but EVAL_THRESHOLD_FILE was not provided"
                )
            p = Path(path)
            if not p.exists():
                raise RuntimeError(
                    "Threshold file not found at "
                    f"EVAL_THRESHOLD_FILE='{p}'. "
                    "Ensure the champion symlink "
                    "'models/experiments/encoder/codet5p_220m_final_sweep_latest' "
                    "points to the Stage 3 run containing "
                    "'threshold_sweep_results.json'."
                )
            # Load JSON threshold file (single threshold for all test sets)
            try:
                with open(p, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and "best_threshold" in data:
                    return float(data["best_threshold"])
                raise RuntimeError(
                    f"Threshold file '{p}' missing required 'best_threshold' field. "
                    "Regenerate Stage 3 results or re-link the champion symlink."
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to read threshold from file '{p}': {exc}"
                ) from exc
        # argmax or unknown
        return None

    def evaluate(
        self,
        test_path: str,
        batch_size: int = 8,
        max_samples: Optional[int] = None,
        save_predictions: bool = True
    ) -> Dict:
        """Evaluate model on test dataset."""
        logger.info("Starting evaluation...")
        
        # Load test data
        test_dataset = self._load_test_data(test_path, max_samples)
        
        # Create temporary training args for evaluation
        eval_args = TrainingArguments(
            output_dir=str(self.output_dir / "temp_eval"),
            per_device_eval_batch_size=batch_size,
            do_eval=True,
            do_train=False,
            logging_dir=None,
            report_to=None
        )
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
        )
        
        start_time = time.time()
        
        # Run evaluation
        logger.info("Running model evaluation...")
        eval_results = trainer.evaluate()
        
        # Get detailed predictions
        logger.info("Getting detailed predictions...")
        predictions = trainer.predict(test_dataset)
        
        evaluation_time = time.time() - start_time
        
        # Process predictions
        # Handle different model output formats (same as training fix)
        if isinstance(predictions.predictions, tuple):
            # T5-based models return tuple (logits, ...)
            logits = predictions.predictions[0]
        else:
            # Simple models like CodeBERT return single tensor
            logits = predictions.predictions

        # Default argmax; optionally apply calibrated threshold
        y_pred = np.argmax(logits, axis=1)
        y_true = predictions.label_ids

        # Get prediction probabilities
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()

        # Apply threshold if configured
        threshold_used: Optional[float] = self._resolve_threshold()
        if threshold_used is not None:
            try:
                y_pred = (probs[:, 1] >= float(threshold_used)).astype(int)
            except Exception:
                pass
        
        # Calculate detailed metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # Handle case where any metric might be None
        if precision is None:
            precision = 0.0
        if recall is None:
            recall = 0.0
        if f1 is None:
            f1 = 0.0
        if support is None:
            support = 0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create detailed results
        detailed_results = []
        for i, sample in enumerate(test_dataset.data):
            detailed_result = {
                'sample_id': i,
                'smell': sample.get('smell', 'unknown'),
                'file': sample.get('file', 'unknown'),
                'true_label': sample['label'],
                'predicted_label': 'TP' if y_pred[i] == 1 else 'FP',
                'probabilities': {
                    'FP': float(probs[i][0]),
                    'TP': float(probs[i][1])
                },
                'content': sample.get('content', '')[:200] + '...',  # Truncate for readability
            }
            detailed_results.append(detailed_result)
        
        # Calculate per-smell metrics
        smell_metrics = self._calculate_per_smell_metrics(detailed_results)
        
        # Prepare results
        results = {
            'model_path': str(self.model_path),
            'test_path': test_path,
            'num_samples': len(test_dataset),
            'evaluation_time': evaluation_time,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(support),
                'eval_loss': float(eval_results.get('eval_loss', 0.0))
            },
            'threshold_used': threshold_used,
            'confusion_matrix': {
                'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            },
            'smell_metrics': smell_metrics,
            'predictions_summary': {
                'total_tp_predicted': int(np.sum(y_pred == 1)),
                'total_fp_predicted': int(np.sum(y_pred == 0)),
                'total_tp_actual': int(np.sum(y_true == 1)),
                'total_fp_actual': int(np.sum(y_true == 0)),
            }
        }
        
        # Save detailed predictions if requested
        if save_predictions:
            pred_file = self.output_dir / "detailed_predictions.json"
            with open(pred_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"Detailed predictions saved to {pred_file}")
        
        # Save results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Clean up temp_eval directory
        import shutil
        temp_eval_dir = self.output_dir / "temp_eval"
        if temp_eval_dir.exists():
            shutil.rmtree(temp_eval_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_eval_dir}")

        logger.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logger.info(f"Results: Acc={accuracy:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

        return results
    
    def _calculate_per_smell_metrics(self, detailed_results: List[Dict]) -> Dict:
        """Calculate metrics per smell type."""
        smell_data = {}
        
        for result in detailed_results:
            smell = result['smell']
            if smell not in smell_data:
                smell_data[smell] = {'y_true': [], 'y_pred': []}
            
            y_true = 1 if result['true_label'] == "TP" else 0
            y_pred = 1 if result['predicted_label'] == "TP" else 0
            
            smell_data[smell]['y_true'].append(y_true)
            smell_data[smell]['y_pred'].append(y_pred)
        
        smell_metrics = {}
        for smell, data in smell_data.items():
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            if len(y_true) > 0:
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )

                # Handle None values in per-smell metrics
                if precision is None:
                    precision = 0.0
                if recall is None:
                    recall = 0.0
                if f1 is None:
                    f1 = 0.0

                smell_metrics[smell] = {
                    'count': len(y_true),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp_actual': sum(y_true),
                    'fp_actual': len(y_true) - sum(y_true),
                }
        
        return smell_metrics

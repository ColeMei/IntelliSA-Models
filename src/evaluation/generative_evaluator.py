import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerativeEvaluator:
    """Evaluator for generative approach using fine-tuned CodeLLaMA with LoRA."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        use_4bit: bool = True,
        max_new_tokens: int = 10,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load config to get base model name
        try:
            with open(self.model_path / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path", "codellama/CodeLlama-34b-hf")
        except Exception:
            # Fallback to default
            base_model_name = "codellama/CodeLlama-34b-hf"
            logger.warning(f"Could not load adapter config, using default base model: {base_model_name}")
        
        # Configure quantization
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapters
        try:
            self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
            logger.info("LoRA adapters loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {e}")
            raise
        
        # Set to evaluation mode
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
    
    def _load_test_data(self, test_path: str) -> List[Dict]:
        """Load test data from JSONL file."""
        data = []
        with open(test_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        
        logger.info(f"Loaded {len(data)} test samples")
        return data
    
    def _predict_single(self, input_text: str) -> Tuple[str, float]:
        """Make prediction on single input."""
        # Format input properly
        formatted_input = f"### Instruction:\n{input_text}\n\n### Response:\n"
        
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                top_p=0.9,
                num_return_sequences=1
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Extract prediction from generated text
        gen_upper = generated_text.upper()
        if "TP" in gen_upper[:10]:
            prediction = "TP"
            confidence = 0.9  # High confidence for clear TP
        elif "FP" in gen_upper[:10]:
            prediction = "FP"
            confidence = 0.9  # High confidence for clear FP
        else:
            # Default to FP if unclear
            prediction = "FP"
            confidence = 0.5  # Low confidence for unclear cases
        
        return prediction, confidence
    
    def evaluate(
        self,
        test_path: str,
        batch_size: int = 1,  # Generative models typically use batch_size=1
        max_samples: Optional[int] = None,
        save_predictions: bool = True
    ) -> Dict:
        """Evaluate model on test dataset."""
        logger.info("Starting evaluation...")
        
        # Load test data
        test_data = self._load_test_data(test_path)
        
        if max_samples:
            test_data = test_data[:max_samples]
            logger.info(f"Limited evaluation to {max_samples} samples")
        
        # Run predictions
        predictions = []
        true_labels = []
        confidences = []
        detailed_results = []
        
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(test_data, desc="Evaluating")):
            try:
                # Use the prompted input
                input_text = sample['with_prompt']
                true_label = sample['label']
                
                # Make prediction
                pred_label, confidence = self._predict_single(input_text)
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                confidences.append(confidence)
                
                # Store detailed result
                detailed_result = {
                    'sample_id': i,
                    'smell': sample.get('smell', 'unknown'),
                    'file': sample.get('file', 'unknown'),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'content': sample.get('content', '')[:200] + '...',  # Truncate for readability
                }
                detailed_results.append(detailed_result)
                
                # Log progress periodically
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_data)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                # Use default values for failed samples
                predictions.append("FP")
                true_labels.append(sample['label'])
                confidences.append(0.0)
                detailed_results.append({
                    'sample_id': i,
                    'smell': sample.get('smell', 'unknown'),
                    'file': sample.get('file', 'unknown'),
                    'true_label': sample['label'],
                    'predicted_label': "FP",
                    'confidence': 0.0,
                    'error': str(e),
                    'content': sample.get('content', '')[:200] + '...',
                })
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics
        # Convert string labels to binary (TP=1, FP=0)
        y_true = [1 if label == "TP" else 0 for label in true_labels]
        y_pred = [1 if label == "TP" else 0 for label in predictions]
        
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
        
        # Calculate per-smell metrics
        smell_metrics = self._calculate_per_smell_metrics(detailed_results)
        
        # Prepare results
        results = {
            'model_path': str(self.model_path),
            'test_path': test_path,
            'num_samples': len(test_data),
            'evaluation_time': evaluation_time,
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(support),
            },
            'confusion_matrix': {
                'tn': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                'fp': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                'fn': int(cm[1, 0]) if cm.shape == (2, 2) else 0,
                'tp': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            },
            'smell_metrics': smell_metrics,
            'average_confidence': float(np.mean(confidences)),
            'predictions_summary': {
                'total_tp_predicted': sum(1 for p in predictions if p == "TP"),
                'total_fp_predicted': sum(1 for p in predictions if p == "FP"),
                'total_tp_actual': sum(1 for t in true_labels if t == "TP"),
                'total_fp_actual': sum(1 for t in true_labels if t == "FP"),
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
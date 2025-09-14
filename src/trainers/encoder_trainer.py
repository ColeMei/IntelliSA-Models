import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from typing import Dict, List, Optional
import numpy as np
import time
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_with_retry(model_name: str, num_labels: int = 2, max_retries: int = 5, base_delay: float = 10.0):
    """
    Load model and tokenizer with retry logic for rate limiting.
    
    Args:
        model_name: HuggingFace model name
        num_labels: Number of classification labels
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (exponential backoff)
    
    Returns:
        tuple: (tokenizer, model)
    """
    trust_remote_code = 'codet5p' in model_name.lower()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading {model_name} (attempt {attempt + 1}/{max_retries})")
            
            # Load tokenizer with retry
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Load model with retry
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                trust_remote_code=trust_remote_code
            )
            
            logger.info(f"Successfully loaded {model_name}")
            return tokenizer, model
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(term in error_str for term in ['429', 'rate limit', 'too many requests'])
            is_dependency = any(term in error_str for term in ['protobuf', 'sentencepiece'])
            
            if is_dependency:
                logger.error(f"Missing dependency for {model_name}: {e}")
                logger.error("Please install: pip install protobuf sentencepiece")
                raise
            
            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                logger.warning(f"Rate limited loading {model_name}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            
            if attempt == max_retries - 1:
                logger.error(f"Failed to load {model_name} after {max_retries} attempts: {e}")
                raise
            
            # For other errors, retry with shorter delay
            delay = 5 + random.uniform(0, 3)
            logger.warning(f"Error loading {model_name}: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    
    raise Exception(f"Failed to load {model_name} after {max_retries} attempts")

class IacDetectionDataset(Dataset):
    """Dataset for IaC security smell detection classification using encoder approach."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data and format for training."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Use content field as input
                input_text = sample['content']
                # Convert TP/FP to 0/1
                target = 1 if sample['label'] == 'TP' else 0
                
                data.append({
                    'input_text': input_text,
                    'target': target,
                    'smell': sample['smell']
                })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sample['input_text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['target'], dtype=torch.long)
        }

def compute_metrics(pred):
    """Compute metrics for evaluation.
    
    Handles different model output formats:
    - CodeBERT/RoBERTa: Returns single logits tensor
    - T5-based models: Returns tuple with logits as first element
    """
    labels = pred.label_ids
    
    # Handle different prediction formats based on model architecture
    if isinstance(pred.predictions, tuple):
        # T5-based models return tuple (logits, ...)
        logits = pred.predictions[0]
    else:
        # Simple models like CodeBERT return single tensor
        logits = pred.predictions
    
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class EncoderTrainer:
    """Trainer for encoder approach using CodeBERT/CodeT5 for IaC security smell detection."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        output_dir: str = "models/encoder"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize tokenizer and model with retry logic
        self.tokenizer, self.model = load_model_with_retry(model_name, num_labels=2)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_datasets(self, train_path: str, val_path: str):
        """Prepare train and validation datasets."""
        self.train_dataset = IacDetectionDataset(train_path, self.tokenizer)
        self.val_dataset = IacDetectionDataset(val_path, self.tokenizer)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def train(
        self,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 50,
        save_steps: int = 50,
        eval_steps: int = 25,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        early_stopping_patience: int = None,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = "f1",
        early_stopping_mode: str = "max",
        lr_scheduler_type: str = "linear",
        fp16: bool = True,
        dataloader_pin_memory: bool = True
    ):
        """Train the model with optimized hyperparameters."""

        # Configure gradient checkpointing if enabled
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr_scheduler_type=lr_scheduler_type,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model=early_stopping_metric,
            greater_is_better=(early_stopping_mode == "max"),
            dataloader_pin_memory=dataloader_pin_memory,
            fp16=fp16,
            # Fix for datasets compatibility
            remove_unused_columns=False,
        )

        # Add early stopping callback if configured
        callbacks = []
        if early_stopping_patience is not None:
            from transformers import EarlyStoppingCallback
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_min_delta
            )
            callbacks.append(early_stopping)
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        
        # Perform threshold sweep if configured
        threshold_sweep_config = getattr(self, 'threshold_sweep_config', None)
        if threshold_sweep_config and threshold_sweep_config.get('enabled', False):
            self._perform_threshold_sweep(trainer, threshold_sweep_config)
    
    def _perform_threshold_sweep(self, trainer, threshold_config):
        """Perform threshold sweep for optimal decision boundary."""
        import numpy as np
        import torch
        
        logger.info("Performing threshold sweep...")
        
        # Get predictions on validation set
        predictions = trainer.predict(self.val_dataset)
        logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()  # Get positive class probabilities
        
        # Get true labels
        true_labels = np.array([item['labels'].item() for item in self.val_dataset])
        
        # Sweep thresholds
        threshold_range = threshold_config.get('range', [0.3, 0.7])
        step = threshold_config.get('step', 0.01)
        metric = threshold_config.get('metric', 'f1')
        
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            pred_labels = (probs >= threshold).astype(int)
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(true_labels, pred_labels)
            elif metric == 'accuracy':
                from sklearn.metrics import accuracy_score
                score = accuracy_score(true_labels, pred_labels)
            else:
                continue
                
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} with {metric}: {best_score:.3f}")
        
        # Save threshold results
        threshold_results = {
            'best_threshold': float(best_threshold),
            'best_score': float(best_score),
            'metric': metric,
            'threshold_range': threshold_range,
            'step': step
        }
        
        import json
        with open(Path(self.output_dir) / "threshold_sweep_results.json", 'w') as f:
            json.dump(threshold_results, f, indent=2)
    
    def predict(self, input_text: str) -> Dict:
        """Make prediction on new input."""
        self.model.eval()
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
        
        return {
            'prediction': 'TP' if prediction.item() == 1 else 'FP',
            'probabilities': probs.squeeze().tolist()
        }

def main():
    """Main training script for encoder approach."""
    
    # Initialize trainer
    trainer = EncoderTrainer(
        model_name="microsoft/codebert-base",
        output_dir="models/encoder"
    )

    # Prepare datasets
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/val.jsonl"
    
    trainer.prepare_datasets(train_path, val_path)
    
    # Train the model
    trainer.train(
        batch_size=2,
        learning_rate=2e-5,
        num_epochs=2,
        warmup_steps=10,
        save_steps=25,
        eval_steps=10
    )
    
    # Test prediction
    test_input = "password = 'secret123'"
    prediction = trainer.predict(test_input)
    logger.info(f"Test prediction: {prediction}")

if __name__ == "__main__":
    main()

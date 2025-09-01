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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChefDetectionDataset(Dataset):
    """Dataset for Chef detection classification using encoder approach."""
    
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
                    'smell': sample['smell'],
                    'confidence': sample['confidence']
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
    """Trainer for encoder approach using CodeBERT/CodeT5."""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        output_dir: str = "models/encoder"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize tokenizer and model
        # CodeT5+ models require trust_remote_code=True
        trust_remote_code = 'codet5p' in model_name.lower()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Use the standard approach instead of custom class
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            trust_remote_code=trust_remote_code
        )
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_datasets(self, train_path: str, val_path: str):
        """Prepare train and validation datasets."""
        self.train_dataset = ChefDetectionDataset(train_path, self.tokenizer)
        self.val_dataset = ChefDetectionDataset(val_path, self.tokenizer)
        
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
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
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
            'confidence': probs.max().item(),
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
    train_path = "data/processed/chef_train.jsonl"
    val_path = "data/processed/chef_val.jsonl"
    
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

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging
from typing import Dict, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChefDetectionDataset(Dataset):
    """Dataset for Chef detection classification using generative approach."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load JSONL data and format for training."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                # Use with_prompt field as input
                input_text = sample['with_prompt']
                # Target: TP or FP
                target = sample['label']
                
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
        
        # Better formatting for instruction tuning
        instruction = sample['input_text']
        response = sample['target']
        
        # Format as conversation
        full_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels - only compute loss on response tokens
        labels = encoding['input_ids'].clone()
        
        # Find where response starts and mask instruction tokens
        instruction_part = f"### Instruction:\n{instruction}\n\n### Response:\n"
        instruction_tokens = self.tokenizer(instruction_part, add_special_tokens=False)['input_ids']
        instruction_length = len(instruction_tokens)
        
        # Mask instruction tokens in labels
        labels[:instruction_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class GenerativeTrainer:
    """Trainer for generative approach using CodeLLaMA with LoRA."""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-32b-hf",
        output_dir: str = "models/generative",
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        
        # Configure quantization for 32B model
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize model with quantization
        logger.info(f"Loading model {model_name} with {'4-bit' if use_4bit else '16-bit'} precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare model for training if using quantization
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA with better target modules for CodeLlama
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def prepare_datasets(self, train_path: str, val_path: str):
        """Prepare train and validation datasets."""
        self.train_dataset = ChefDetectionDataset(train_path, self.tokenizer)
        self.val_dataset = ChefDetectionDataset(val_path, self.tokenizer)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def train(
        self,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        save_steps: int = 100,
        eval_steps: int = 50,
        gradient_accumulation_steps: int = 4
    ):
        """Train the model."""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            # Optimization for memory
            dataloader_num_workers=4,
            group_by_length=True,
            # Better saving strategy
            save_total_limit=2,
            # Report metrics
            report_to=None
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            ),
        )
        
        logger.info("Starting training...")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def predict(self, input_text: str, max_new_tokens: int = 10) -> Dict:
        """Make prediction on new input."""
        self.model.eval()
        
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.1,
                top_p=0.9
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return {
            'prediction': generated_text,
            'confidence': 1.0,
            'full_response': self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

def main():
    """Main training script for generative approach."""
    
    # Initialize trainer
    trainer = GenerativeTrainer(
        model_name="codellama/CodeLlama-32b-hf",
        output_dir="models/generative",
        use_4bit=True
    )
    
    # Prepare datasets
    train_path = "data/processed/chef_train.jsonl"
    val_path = "data/processed/chef_val.jsonl"
    
    trainer.prepare_datasets(train_path, val_path)
    
    # Train the model
    trainer.train(
        batch_size=2,
        learning_rate=5e-5,
        num_epochs=3,
        warmup_steps=100,
        save_steps=100,
        eval_steps=50,
        gradient_accumulation_steps=4
    )

if __name__ == "__main__":
    main()
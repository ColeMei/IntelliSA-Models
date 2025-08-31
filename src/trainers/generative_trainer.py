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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import numpy as np
import re

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
                
                # Extract the instruction part from with_prompt
                with_prompt = sample['with_prompt']
                
                # The with_prompt already contains the full instruction
                # We need to separate instruction from expected response format
                if 'Return ONLY JSON:' in with_prompt:
                    # Split at "Return ONLY JSON:" to get just the instruction
                    parts = with_prompt.split('Return ONLY JSON:')
                    instruction = parts[0].strip()
                    # Add back the response format requirement
                    instruction += '\n\nReturn ONLY JSON: {"decision":"YES|NO","confidence":0.0-1.0}'
                else:
                    instruction = with_prompt
                
                # Create the expected JSON response based on label
                label = sample['label']  # TP or FP
                confidence = sample.get('confidence', 1.0)
                
                # Convert TP/FP to YES/NO
                # TP = True Positive = vulnerability correctly identified = YES
                # FP = False Positive = incorrectly flagged as vulnerability = NO
                decision = "YES" if label == "TP" else "NO"
                target_response = json.dumps({"decision": decision, "confidence": confidence})
                
                data.append({
                    'instruction': instruction,
                    'target_response': target_response,
                    'original_label': label,
                    'confidence': confidence,
                    'smell': sample['smell']
                })
        
        logger.info(f"Loaded {len(data)} samples from {data_path}")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Format using CodeLlama instruction template
        instruction = sample['instruction']
        response = sample['target_response']
        
        # Use CodeLlama's instruction format
        full_text = f"<s>[INST] {instruction} [/INST] {response}</s>"
        
        # Tokenize the full conversation
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Let the data collator handle padding
            return_tensors='pt'
        )
        
        # Tokenize just the instruction part to find where response starts
        instruction_part = f"<s>[INST] {instruction} [/INST] "
        instruction_encoding = self.tokenizer(
            instruction_part,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        # Create labels for training - only compute loss on response tokens
        labels = full_encoding['input_ids'].clone()
        instruction_length = instruction_encoding['input_ids'].shape[1]
        
        # Mask instruction tokens (set to -100 so they're ignored in loss computation)
        labels[:, :instruction_length] = -100
        
        return {
            'input_ids': full_encoding['input_ids'].squeeze(),
            'attention_mask': full_encoding['attention_mask'].squeeze(), 
            'labels': labels.squeeze(),
            # Keep numeric label for evaluation
            'label_id': 1 if sample['original_label'] == 'TP' else 0
        }

class GenerativeTrainer:
    """Trainer for generative approach using CodeLLaMA with LoRA."""
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-34b-hf",
        output_dir: str = "models/generative",
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        
        # Configure quantization for large models
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16  # Better than float16
            )
        else:
            bnb_config = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize model with quantization
        logger.info(f"Loading model {model_name} with {'4-bit' if use_4bit else 'full'} precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False  # Disable cache for training
        )
        
        # Prepare model for LoRA training if using quantization
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        if lora_target_modules is None:
            # Default target modules for CodeLlama
            lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none"
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing for memory efficiency
        try:
            self.model.gradient_checkpointing_enable(use_reentrant=False)
        except TypeError:
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def prepare_datasets(self, train_path: str, val_path: str, max_length: int = 512):
        """Prepare train and validation datasets."""
        self.train_dataset = ChefDetectionDataset(train_path, self.tokenizer, max_length=max_length)
        self.val_dataset = ChefDetectionDataset(val_path, self.tokenizer, max_length=max_length)
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        
        # Print a sample for debugging
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            decoded_sample = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            logger.info(f"Sample training example:\n{decoded_sample[:500]}...")
    
    def train(
        self,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        save_steps: int = 100,
        eval_steps: int = 50,
        gradient_accumulation_steps: int = 4,
        evaluation_strategy: str = "steps",
        save_strategy: str = "steps",
        logging_steps: int = 10,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        fp16: bool = True,
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
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            bf16=True,  # Use bfloat16 instead of fp16 for better stability
            dataloader_pin_memory=False,
            remove_unused_columns=False,  # Keep our custom fields
            group_by_length=True,  # Group similar length sequences for efficiency
            save_total_limit=2,
            report_to=None  # Disable wandb
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
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")

        # Run generation-based evaluation
        self._run_generation_evaluation()
    
    def _run_generation_evaluation(self):
        """Run evaluation using generation on validation set."""
        logger.info("Running generation-based evaluation on validation set...")
        
        predictions = []
        true_labels = []
        
        self.model.eval()
        
        # Evaluate on a subset for speed (you can increase this)
        eval_samples = min(50, len(self.val_dataset))
        
        for i in range(eval_samples):
            try:
                item = self.val_dataset[i]
                
                # Get the instruction part by decoding and splitting
                full_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
                
                if '[/INST]' in full_text:
                    instruction_part = full_text.split('[/INST]')[0] + '[/INST] '
                else:
                    # Fallback - use first half
                    instruction_part = full_text[:len(full_text)//2]
                
                # Tokenize instruction
                inputs = self.tokenizer(
                    instruction_part,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length - 50  # Leave room for response
                ).to(self.model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=30,  # JSON response should be short
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Extract generated part
                generated = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Parse the JSON response
                pred_label = self._extract_decision_from_response(generated)
                true_label = item['label_id']
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                
                # Log first few examples for debugging
                if i < 3:
                    logger.info(f"Sample {i}: Generated='{generated}' -> Pred={pred_label}, True={true_label}")
                    
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                predictions.append(0)  # Default to negative prediction
                true_labels.append(item['label_id'])
        
        # Calculate metrics
        if predictions and true_labels:
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
            
            metrics = {
                "generation_accuracy": accuracy,
                "generation_precision": precision, 
                "generation_recall": recall,
                "generation_f1": f1,
                "eval_samples": eval_samples
            }
            
            logger.info(f"Generation evaluation metrics: {metrics}")
            
            # Save metrics
            with open(Path(self.output_dir) / "generation_eval_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
        else:
            logger.warning("No valid predictions generated during evaluation")
    
    def _extract_decision_from_response(self, response: str) -> int:
        """Extract YES/NO decision from model response and convert to binary label."""
        try:
            # Try to parse as JSON first
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.find('}', start) + 1
                json_str = response[start:end]
                
                # Clean up common JSON formatting issues
                json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # Quote keys
                json_str = re.sub(r':\s*([^",}\s]+)([,}])', r': "\1"\2', json_str)  # Quote unquoted values
                
                try:
                    parsed = json.loads(json_str)
                    decision = parsed.get('decision', '').upper()
                    return 1 if decision == 'YES' else 0
                except json.JSONDecodeError:
                    pass
            
            # Fallback: look for YES/NO keywords
            response_upper = response.upper()
            if 'YES' in response_upper:
                return 1
            elif 'NO' in response_upper:
                return 0
            else:
                return 0  # Default to negative if unclear
                
        except Exception:
            return 0  # Default to negative on any error
    
    def predict(self, input_text: str, max_new_tokens: int = 30) -> Dict:
        """Make prediction on new input."""
        self.model.eval()
        
        # Format input with CodeLlama instruction template
        formatted_input = f"<s>[INST] {input_text} [/INST] "
        
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length - max_new_tokens
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new generated tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Extract decision
        decision = self._extract_decision_from_response(generated_text)
        
        return {
            'prediction': 'TP' if decision == 1 else 'FP',
            'confidence': 1.0,  # Could extract from JSON if needed
            'raw_response': generated_text,
            'full_response': self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

def main():
    """Main training script for generative approach."""
    
    # Initialize trainer
    trainer = GenerativeTrainer(
        model_name="codellama/CodeLlama-34b-hf",
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
import json
import re
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
        max_new_tokens: int = 50,  # Increased for JSON response
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
                bnb_4bit_compute_dtype=torch.bfloat16  # Match training config
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
            torch_dtype=torch.bfloat16,  # Match training config
            use_cache=True  # Enable cache for inference
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
    
    def _format_instruction(self, with_prompt: str) -> str:
        """Extract and format the instruction part from with_prompt."""
        # The with_prompt contains the full instruction
        # We need to extract just the instruction part and format it properly
        if 'Return ONLY JSON:' in with_prompt:
            # Split to get instruction part
            parts = with_prompt.split('Return ONLY JSON:')
            instruction = parts[0].strip()
            # Add back the response format requirement
            instruction += '\n\nReturn ONLY JSON: {"decision":"YES|NO","confidence":0.0-1.0}'
        else:
            instruction = with_prompt
        
        return instruction
    
    def _predict_single(self, sample: Dict, sample_id: int) -> Tuple[str, float]:
        """Make prediction on single input."""
        # Extract and format instruction
        raw_instruction = sample['with_prompt']
        formatted_instruction = self._format_instruction(raw_instruction)
        
        # Use CodeLlama instruction format (matching training)
        formatted_input = f"<s>[INST] {formatted_instruction} [/INST] "
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.max_new_tokens,  # Leave room for response
            padding=False
        ).to(self.model.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            if sample_id < 10:  # Log first few for debugging
                logger.info(f"Sample {sample_id} generated: '{generated_text}'")
            
            # Parse the response
            prediction, confidence = self._parse_response(generated_text, sample_id)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error generating for sample {sample_id}: {e}")
            return "FP", 0.5  # Default fallback
    
    def _parse_response(self, generated_text: str, sample_id: int) -> Tuple[str, float]:
        """Parse model response to extract prediction and confidence."""
        try:
            # Try to find and parse JSON in the response
            json_pattern = r'\{[^}]*\}'
            json_matches = re.findall(json_pattern, generated_text)
            
            for json_str in json_matches:
                try:
                    # Clean up common JSON formatting issues
                    cleaned_json = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)  # Quote keys
                    cleaned_json = re.sub(r':\s*([^",}\s]+)([,}])', r': "\1"\2', cleaned_json)  # Quote values
                    
                    response_data = json.loads(cleaned_json)
                    decision = response_data.get("decision", "").upper()
                    confidence = float(response_data.get("confidence", 0.5))
                    
                    # Map decision to prediction label
                    # During training: TP -> YES, FP -> NO
                    # During evaluation: YES -> TP, NO -> FP
                    if decision == "YES":
                        return "TP", confidence
                    elif decision == "NO":
                        return "FP", confidence
                    else:
                        if sample_id < 5:
                            logger.info(f"Sample {sample_id}: Invalid decision '{decision}', using fallback")
                        return "FP", 0.5
                        
                except json.JSONDecodeError:
                    continue
            
            # Fallback: look for keywords if JSON parsing fails
            text_upper = generated_text.upper()
            if 'YES' in text_upper or '"YES"' in text_upper:
                return "TP", 0.8
            elif 'NO' in text_upper or '"NO"' in text_upper:
                return "FP", 0.8
            else:
                if sample_id < 5:
                    logger.info(f"Sample {sample_id}: No clear decision found, using fallback")
                return "FP", 0.5
                
        except Exception as e:
            if sample_id < 5:
                logger.error(f"Sample {sample_id}: Error parsing response: {e}")
            return "FP", 0.5
    
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
                true_label = sample['label']  # Should be TP or FP
                
                # Make prediction
                pred_label, confidence = self._predict_single(sample, i)
                
                predictions.append(pred_label)
                true_labels.append(true_label)
                confidences.append(confidence)
                
                # Store detailed result
                detailed_result = {
                    'sample_id': i,
                    'smell': sample.get('smell', 'unknown'),
                    'file': sample.get('file', 'unknown'),
                    'line': sample.get('line', 0),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence,
                    'match': pred_label == true_label,
                    'content': sample.get('content', '')[:200] + '...',  # Truncate for readability
                }
                detailed_results.append(detailed_result)
                
                # Log progress periodically
                if (i + 1) % 50 == 0:
                    current_acc = accuracy_score(
                        [1 if t == "TP" else 0 for t in true_labels[:i+1]],
                        [1 if p == "TP" else 0 for p in predictions[:i+1]]
                    )
                    logger.info(f"Processed {i + 1}/{len(test_data)} samples - Current accuracy: {current_acc:.4f}")
                    
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
                    'line': sample.get('line', 0),
                    'true_label': sample['label'],
                    'predicted_label': "FP",
                    'confidence': 0.0,
                    'match': False,
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
        
        # Handle None values
        precision = precision if precision is not None else 0.0
        recall = recall if recall is not None else 0.0
        f1 = f1 if f1 is not None else 0.0
        support = support if support is not None else 0
        
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
            'samples_per_second': len(test_data) / evaluation_time,
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
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'predictions_summary': {
                'total_tp_predicted': sum(1 for p in predictions if p == "TP"),
                'total_fp_predicted': sum(1 for p in predictions if p == "FP"),
                'total_tp_actual': sum(1 for t in true_labels if t == "TP"),
                'total_fp_actual': sum(1 for t in true_labels if t == "FP"),
                'correct_predictions': sum(1 for p, t in zip(predictions, true_labels) if p == t),
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
        logger.info(f"Final Results: Acc={accuracy:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")
        logger.info(f"Speed: {len(test_data) / evaluation_time:.2f} samples/second")
        
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
                
                # Handle None values
                precision = precision if precision is not None else 0.0
                recall = recall if recall is not None else 0.0
                f1 = f1 if f1 is not None else 0.0
                
                smell_metrics[smell] = {
                    'count': len(y_true),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1),
                    'tp_actual': sum(y_true),
                    'fp_actual': len(y_true) - sum(y_true),
                    'tp_predicted': sum(y_pred),
                    'fp_predicted': len(y_pred) - sum(y_pred),
                }
        
        return smell_metrics

def main():
    """Example usage of the evaluator."""
    evaluator = GenerativeEvaluator(
        model_path="models/generative_latest",
        output_dir="results/evaluation",
        use_4bit=True,
        max_new_tokens=50
    )
    
    results = evaluator.evaluate(
        test_path="data/processed/chef_test.jsonl",
        max_samples=None,  # Evaluate all samples
        save_predictions=True
    )
    
    print(f"Evaluation Results:")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")

if __name__ == "__main__":
    main()
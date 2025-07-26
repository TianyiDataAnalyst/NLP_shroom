#!/usr/bin/env python3
"""
Ensemble Methods Implementation for Section 7 - Advanced Model Experiments
Combine predictions from multiple models for improved performance
"""

import torch
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from evaluate import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class EnsemblePredictor:
    """Ensemble predictor combining multiple trained models"""
    
    def __init__(self, model_paths, model_names, test_lang='en', data_path='./data/val'):
        self.model_paths = model_paths
        self.model_names = model_names
        self.test_lang = test_lang
        self.data_path = data_path
        self.metric = load('seqeval')
        
        # Load models and tokenizers
        self.models = []
        self.tokenizers = []
        
        print(f"Loading {len(model_paths)} models for ensemble...")
        for i, (path, name) in enumerate(zip(model_paths, model_names)):
            print(f"Loading model {i+1}/{len(model_paths)}: {name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForTokenClassification.from_pretrained(path)
                model.eval()
                
                self.tokenizers.append(tokenizer)
                self.models.append(model)
                print(f"✅ Successfully loaded {name}")
                
            except Exception as e:
                print(f"❌ Failed to load {name}: {str(e)}")
                self.tokenizers.append(None)
                self.models.append(None)
    
    def load_test_data(self):
        """Load English test data"""
        test_file = f"{self.data_path}/mushroom.{self.test_lang}-val.v2.jsonl"
        
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        print(f"Loading test data from {test_file}")
        dataset = load_dataset('json', data_files={'test': test_file})
        test_data = dataset['test']
        
        print(f"Loaded {len(test_data)} test samples")
        return test_data
    
    def tokenize_test_data(self, test_data, tokenizer):
        """Tokenize test data for inference"""
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['model_output_text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_offsets_mapping=True,
                is_split_into_words=False
            )
            
            # Create labels for evaluation
            labels = []
            for hard_labels, offset_mapping in zip(examples['hard_labels'], tokenized['offset_mapping']):
                label = [0] * len(offset_mapping)
                
                # Mark tokens that overlap with hallucinated spans
                for start, end in hard_labels:
                    for j, (token_start, token_end) in enumerate(offset_mapping):
                        if (token_start < end and token_end > start and 
                            token_start != token_end):  # Ignore special tokens
                            label[j] = 1
                
                labels.append(label)
            
            tokenized['labels'] = labels
            return tokenized
        
        tokenized_data = test_data.map(tokenize_function, batched=True)
        return tokenized_data
    
    def get_model_predictions(self, model, tokenizer, tokenized_data):
        """Get predictions from a single model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_predictions = []
        all_logits = []
        
        with torch.no_grad():
            for example in tokenized_data:
                # Prepare inputs
                input_ids = torch.tensor([example['input_ids']]).to(device)
                attention_mask = torch.tensor([example['attention_mask']]).to(device)
                
                # Run inference
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions and logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.append(predictions.cpu().numpy()[0])
                all_logits.append(logits.cpu().numpy()[0])
        
        return all_predictions, all_logits
    
    def ensemble_voting(self, all_model_predictions, method='majority'):
        """Combine predictions using voting methods"""
        print(f"Combining predictions using {method} voting...")
        
        if method == 'majority':
            # Majority voting
            ensemble_predictions = []
            
            for i in range(len(all_model_predictions[0])):  # For each sample
                sample_predictions = []
                
                # Get predictions from all models for this sample
                for model_preds in all_model_predictions:
                    if model_preds is not None:
                        sample_predictions.append(model_preds[i])
                
                if not sample_predictions:
                    # No valid predictions, use zeros
                    ensemble_predictions.append(np.zeros_like(all_model_predictions[0][i]))
                    continue
                
                # Majority vote for each token
                ensemble_pred = []
                max_len = max(len(pred) for pred in sample_predictions)
                
                for token_idx in range(max_len):
                    votes = []
                    for pred in sample_predictions:
                        if token_idx < len(pred):
                            votes.append(pred[token_idx])
                    
                    if votes:
                        # Majority vote
                        majority_vote = 1 if sum(votes) > len(votes) / 2 else 0
                        ensemble_pred.append(majority_vote)
                    else:
                        ensemble_pred.append(0)
                
                ensemble_predictions.append(np.array(ensemble_pred))
            
            return ensemble_predictions
        
        elif method == 'weighted':
            # Weighted voting (could be based on individual model performance)
            # For now, use equal weights
            return self.ensemble_voting(all_model_predictions, 'majority')
    
    def ensemble_averaging(self, all_model_logits):
        """Combine predictions using logit averaging"""
        print(f"Combining predictions using logit averaging...")
        
        ensemble_predictions = []
        
        for i in range(len(all_model_logits[0])):  # For each sample
            sample_logits = []
            
            # Get logits from all models for this sample
            for model_logits in all_model_logits:
                if model_logits is not None:
                    sample_logits.append(model_logits[i])
            
            if not sample_logits:
                # No valid logits, use zeros
                ensemble_predictions.append(np.zeros(len(all_model_logits[0][i])))
                continue
            
            # Average logits
            avg_logits = np.mean(sample_logits, axis=0)
            
            # Get predictions from averaged logits
            predictions = np.argmax(avg_logits, axis=-1)
            ensemble_predictions.append(predictions)
        
        return ensemble_predictions
    
    def run_ensemble_inference(self, method='majority'):
        """Run ensemble inference on test data"""
        print(f"\n{'='*60}")
        print(f"RUNNING ENSEMBLE INFERENCE - {method.upper()}")
        print(f"{'='*60}")
        
        # Load test data
        test_data = self.load_test_data()
        
        # Get predictions from all models
        all_model_predictions = []
        all_model_logits = []
        labels = None
        
        for i, (model, tokenizer, name) in enumerate(zip(self.models, self.tokenizers, self.model_names)):
            if model is None or tokenizer is None:
                print(f"Skipping {name} (failed to load)")
                all_model_predictions.append(None)
                all_model_logits.append(None)
                continue
            
            print(f"Getting predictions from {name}...")
            
            # Tokenize data for this model
            tokenized_data = self.tokenize_test_data(test_data, tokenizer)
            
            # Get predictions
            predictions, logits = self.get_model_predictions(model, tokenizer, tokenized_data)
            
            all_model_predictions.append(predictions)
            all_model_logits.append(logits)
            
            # Store labels (same for all models)
            if labels is None:
                labels = [example['labels'] for example in tokenized_data]
        
        # Combine predictions
        if method == 'majority' or method == 'weighted':
            ensemble_predictions = self.ensemble_voting(all_model_predictions, method)
        elif method == 'averaging':
            ensemble_predictions = self.ensemble_averaging(all_model_logits)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_predictions, labels
    
    def calculate_metrics(self, predictions, labels):
        """Calculate comprehensive evaluation metrics"""
        print(f"Calculating evaluation metrics...")
        
        # Convert to label names for seqeval
        true_predictions = []
        true_labels = []
        
        # Flatten for token-level metrics
        flat_predictions = []
        flat_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_pred = []
            true_label = []
            
            min_len = min(len(prediction), len(label))
            
            for i in range(min_len):
                pred_id = prediction[i]
                label_id = label[i]
                
                if label_id != -100:  # Ignore special tokens
                    pred_name = "HALLUCINATION" if pred_id == 1 else "O"
                    label_name = "HALLUCINATION" if label_id == 1 else "O"
                    
                    true_pred.append(pred_name)
                    true_label.append(label_name)
                    
                    flat_predictions.append(pred_id)
                    flat_labels.append(label_id)
            
            true_predictions.append(true_pred)
            true_labels.append(true_label)
        
        # Calculate seqeval metrics (span-level)
        seqeval_results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        # Calculate token-level metrics
        token_accuracy = sum(1 for p, l in zip(flat_predictions, flat_labels) if p == l) / len(flat_predictions)
        
        # Calculate detailed classification metrics
        class_report = classification_report(
            flat_labels, flat_predictions, 
            target_names=['O', 'HALLUCINATION'], 
            output_dict=True,
            zero_division=0
        )
        
        # Safe access to classification report
        hallucination_metrics = class_report.get('1', class_report.get('HALLUCINATION', {
            'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0
        }))
        
        results = {
            "span_level": {
                "precision": seqeval_results["overall_precision"],
                "recall": seqeval_results["overall_recall"],
                "f1": seqeval_results["overall_f1"],
                "accuracy": seqeval_results["overall_accuracy"]
            },
            "token_level": {
                "accuracy": token_accuracy,
                "precision": hallucination_metrics['precision'],
                "recall": hallucination_metrics['recall'],
                "f1": hallucination_metrics['f1-score']
            },
            "statistics": {
                "total_tokens": len(flat_labels),
                "total_hallucination_tokens": sum(flat_labels),
                "total_predicted_hallucination_tokens": sum(flat_predictions),
                "total_samples": len(predictions)
            }
        }
        
        return results
    
    def evaluate_ensemble(self, methods=['majority', 'averaging']):
        """Evaluate ensemble using different methods"""
        results = {}
        
        for method in methods:
            print(f"\n{'='*80}")
            print(f"EVALUATING ENSEMBLE METHOD: {method.upper()}")
            print(f"{'='*80}")
            
            try:
                # Run ensemble inference
                predictions, labels = self.run_ensemble_inference(method)
                
                # Calculate metrics
                metrics = self.calculate_metrics(predictions, labels)
                
                # Print results
                print(f"\nSpan-Level Metrics:")
                for metric, value in metrics['span_level'].items():
                    print(f"  {metric.capitalize()}: {value:.4f}")
                
                print(f"\nToken-Level Metrics:")
                for metric, value in metrics['token_level'].items():
                    print(f"  {metric.capitalize()}: {value:.4f}")
                
                results[method] = metrics
                
            except Exception as e:
                print(f"❌ Failed to evaluate {method}: {str(e)}")
                results[method] = {'error': str(e)}
        
        return results

def main():
    """Main function for ensemble evaluation"""
    print(f"\n{'='*80}")
    print(f"ENSEMBLE METHODS EVALUATION - SECTION 7")
    print(f"{'='*80}")
    
    # Define available models
    model_paths = [
        "./results",  # Baseline model
        "./advanced_results/xlm-roberta-base_epochs_12"  # Advanced model
    ]
    
    model_names = [
        "baseline-xlm-roberta-base",
        "advanced-xlm-roberta-base"
    ]
    
    # Check which models exist
    available_paths = []
    available_names = []
    
    for path, name in zip(model_paths, model_names):
        if os.path.exists(path):
            available_paths.append(path)
            available_names.append(name)
            print(f"✅ Found model: {name} at {path}")
        else:
            print(f"❌ Model not found: {name} at {path}")
    
    if len(available_paths) < 2:
        print(f"\n❌ Need at least 2 models for ensemble. Found {len(available_paths)}")
        print(f"Available models: {available_names}")
        return
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(available_paths, available_names)
    
    # Evaluate ensemble methods
    results = ensemble.evaluate_ensemble(['majority', 'averaging'])
    
    # Save results
    ensemble_summary = {
        "model_paths": available_paths,
        "model_names": available_names,
        "ensemble_results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    output_dir = "./ensemble_results"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/ensemble_evaluation.json", 'w') as f:
        json.dump(ensemble_summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {output_dir}/ensemble_evaluation.json")
    
    return results

if __name__ == "__main__":
    main()

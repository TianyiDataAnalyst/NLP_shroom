#!/usr/bin/env python3
"""
Advanced Model Evaluation for Section 7 Experiments
Evaluate trained models on English test data and compare with baseline
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

class AdvancedModelEvaluator:
    """Evaluator for advanced model experiments"""
    
    def __init__(self, test_lang='en', data_path='./data/val', output_dir='./advanced_evaluation'):
        self.test_lang = test_lang
        self.data_path = data_path
        self.output_dir = output_dir
        self.metric = load('seqeval')
        
        os.makedirs(output_dir, exist_ok=True)
        
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
    
    def run_inference(self, model, tokenizer, tokenized_data):
        """Run inference on test data"""
        print(f"Running inference on {len(tokenized_data)} samples...")
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i, example in enumerate(tokenized_data):
                # Prepare inputs
                input_ids = torch.tensor([example['input_ids']]).to(device)
                attention_mask = torch.tensor([example['attention_mask']]).to(device)
                
                # Run inference
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_predictions.append(predictions.cpu().numpy()[0])
                all_labels.append(example['labels'])
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(tokenized_data)} samples")
        
        print(f"Inference completed")
        return all_predictions, all_labels
    
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
            
            for pred_id, label_id in zip(prediction, label):
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
        
        # Calculate confusion matrix
        cm = confusion_matrix(flat_labels, flat_predictions)
        
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
            "detailed_metrics": class_report,
            "confusion_matrix": cm.tolist(),
            "statistics": {
                "total_tokens": len(flat_labels),
                "total_hallucination_tokens": sum(flat_labels),
                "total_predicted_hallucination_tokens": sum(flat_predictions),
                "total_samples": len(predictions)
            }
        }
        
        return results
    
    def evaluate_model(self, model_path, model_name):
        """Evaluate a single model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        print(f"Model Path: {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Tokenize test data
        tokenized_data = self.tokenize_test_data(test_data, tokenizer)
        
        # Run inference
        predictions, labels = self.run_inference(model, tokenizer, tokenized_data)
        
        # Calculate metrics
        results = self.calculate_metrics(predictions, labels)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {model_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nSpan-Level Metrics:")
        for metric, value in results['span_level'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nToken-Level Metrics:")
        for metric, value in results['token_level'].items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        print(f"\nStatistics:")
        for stat, value in results['statistics'].items():
            print(f"  {stat.replace('_', ' ').title()}: {value}")
        
        return results
    
    def compare_with_baseline(self, advanced_results, baseline_results):
        """Compare advanced model results with baseline"""
        print(f"\n{'='*60}")
        print(f"COMPARISON WITH BASELINE")
        print(f"{'='*60}")
        
        comparison = {}
        
        # Compare span-level metrics
        print(f"\nSpan-Level Improvements:")
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            baseline_val = baseline_results['span_level'][metric]
            advanced_val = advanced_results['span_level'][metric]
            improvement = advanced_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else float('inf')
            
            print(f"  {metric.capitalize()}:")
            print(f"    Baseline: {baseline_val:.4f}")
            print(f"    Advanced: {advanced_val:.4f}")
            print(f"    Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            comparison[f"span_{metric}"] = {
                'baseline': baseline_val,
                'advanced': advanced_val,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
        
        # Compare token-level metrics
        print(f"\nToken-Level Improvements:")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            baseline_val = baseline_results['token_level'][metric]
            advanced_val = advanced_results['token_level'][metric]
            improvement = advanced_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else float('inf')
            
            print(f"  {metric.capitalize()}:")
            print(f"    Baseline: {baseline_val:.4f}")
            print(f"    Advanced: {advanced_val:.4f}")
            print(f"    Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
            
            comparison[f"token_{metric}"] = {
                'baseline': baseline_val,
                'advanced': advanced_val,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            }
        
        return comparison

def main():
    """Main evaluation function"""
    print(f"\n{'='*80}")
    print(f"ADVANCED MODEL EVALUATION - SECTION 7")
    print(f"{'='*80}")
    
    evaluator = AdvancedModelEvaluator()
    
    # Evaluate advanced model
    advanced_model_path = "./advanced_results/xlm-roberta-base_epochs_12"
    advanced_results = evaluator.evaluate_model(advanced_model_path, "xlm-roberta-base-advanced")
    
    # Load baseline results for comparison
    baseline_results_file = "./evaluation_results/evaluation_summary.json"
    if os.path.exists(baseline_results_file):
        with open(baseline_results_file, 'r') as f:
            baseline_data = json.load(f)
            baseline_results = baseline_data['evaluation_results']
        
        # Compare with baseline
        comparison = evaluator.compare_with_baseline(advanced_results, baseline_results)
        
        # Save comparison results
        comparison_summary = {
            "advanced_model_path": advanced_model_path,
            "baseline_results": baseline_results,
            "advanced_results": advanced_results,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        comparison_summary = convert_numpy_types(comparison_summary)
        
        with open(f"{evaluator.output_dir}/advanced_vs_baseline_comparison.json", 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\nComparison results saved to {evaluator.output_dir}/advanced_vs_baseline_comparison.json")
    
    else:
        print(f"\nBaseline results not found at {baseline_results_file}")
        print(f"Skipping comparison...")
    
    print(f"\n{'='*80}")
    print(f"ADVANCED MODEL EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    return advanced_results

if __name__ == "__main__":
    main()

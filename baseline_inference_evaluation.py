#!/usr/bin/env python3
"""
Baseline Model Inference and Evaluation for Mushroom Task 2025
Test the trained model on English data and generate comprehensive evaluation
"""

import torch
import json
import os
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from evaluate import load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

def load_trained_model(model_path):
    """Load the trained model and tokenizer"""
    print(f"Loading trained model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    print(f"Model loaded successfully")
    print(f"Model config: {model.config}")
    
    return tokenizer, model

def load_test_data(data_path, test_lang='en'):
    """Load test data for the specified language"""
    test_file = f"{data_path}/mushroom.{test_lang}-val.v2.jsonl"
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    print(f"Loading test data from {test_file}")
    
    dataset = load_dataset('json', data_files={'test': test_file})
    test_data = dataset['test']
    
    print(f"Loaded {len(test_data)} test samples")
    
    return test_data

def tokenize_test_data(test_data, tokenizer):
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
        for i, (hard_labels, offset_mapping) in enumerate(zip(examples['hard_labels'], tokenized['offset_mapping'])):
            label = [0] * len(offset_mapping)
            
            # Mark tokens that are part of hallucinated spans
            for start, end in hard_labels:
                for j, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start >= start and token_end <= end and token_start != token_end:
                        label[j] = 1
            
            labels.append(label)
        
        tokenized['labels'] = labels
        return tokenized
    
    tokenized_data = test_data.map(tokenize_function, batched=True)
    return tokenized_data

def run_inference(model, tokenizer, tokenized_data):
    """Run inference on test data"""
    print(f"Running inference on {len(tokenized_data)} samples...")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
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
            all_logits.append(logits.cpu().numpy()[0])
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(tokenized_data)} samples")
    
    print(f"Inference completed")
    
    return all_predictions, all_labels, all_logits

def calculate_metrics(predictions, labels, tokenizer):
    """Calculate comprehensive evaluation metrics"""
    print(f"Calculating evaluation metrics...")
    
    # Load seqeval metric
    metric = load('seqeval')
    
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
    seqeval_results = metric.compute(predictions=true_predictions, references=true_labels)
    
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

    # Calculate span-level statistics
    total_true_spans = sum(len([label for label in seq if label == "HALLUCINATION"]) for seq in true_labels)
    total_pred_spans = sum(len([pred for pred in seq if pred == "HALLUCINATION"]) for seq in true_predictions)

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
            "total_samples": len(predictions),
            "total_true_spans": total_true_spans,
            "total_predicted_spans": total_pred_spans
        }
    }
    
    return results

def create_visualizations(results, output_dir):
    """Create visualization plots for the results"""
    print(f"Creating visualizations...")
    
    # Create output directory for plots
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['O', 'HALLUCINATION'], 
                yticklabels=['O', 'HALLUCINATION'])
    plt.title('Confusion Matrix - Token Level')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Span-level metrics
    span_metrics = results['span_level']
    metrics_names = list(span_metrics.keys())
    metrics_values = list(span_metrics.values())
    
    ax1.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_title('Span-Level Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(metrics_values):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Token-level metrics
    token_metrics = results['token_level']
    metrics_names = list(token_metrics.keys())
    metrics_values = list(token_metrics.values())
    
    ax2.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title('Token-Level Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(metrics_values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    """Main evaluation function"""
    print(f"\n{'='*60}")
    print(f"BASELINE MODEL INFERENCE AND EVALUATION")
    print(f"{'='*60}")
    
    # Configuration
    model_path = './results'
    data_path = './data/val'
    test_lang = 'en'
    output_dir = './evaluation_results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    tokenizer, model = load_trained_model(model_path)
    
    # Load test data
    test_data = load_test_data(data_path, test_lang)
    
    # Tokenize test data
    tokenized_data = tokenize_test_data(test_data, tokenizer)
    
    # Run inference
    predictions, labels, logits = run_inference(model, tokenizer, tokenized_data)
    
    # Calculate metrics
    results = calculate_metrics(predictions, labels, tokenizer)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
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
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Convert numpy types to Python types for JSON serialization
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

    # Save detailed results
    evaluation_summary = {
        "model_path": model_path,
        "test_language": test_lang,
        "test_samples": len(test_data),
        "evaluation_results": convert_numpy_types(results),
        "timestamp": datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"\nDetailed results saved to {output_dir}/evaluation_summary.json")
    print(f"Visualizations saved to {output_dir}/visualizations/")
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    main()

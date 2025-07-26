#!/usr/bin/env python3
"""
Baseline Model Training and Evaluation for Mushroom Task 2025
Modified for validation with progress monitoring and reduced training time
"""

import torch
import json
import os
import argparse
import time
from datetime import datetime
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from evaluate import load
import numpy as np

# Disable WandB logging
os.environ["WANDB_MODE"] = "disabled"

LABEL_LIST = [0, 1]
LANGS = ['ar', 'de', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh']
MODEL_NAME = 'FacebookAI/xlm-roberta-base'

def create_dataset(data_path, test_lang):
    """Create dataset excluding test language"""
    print(f"Creating dataset from {data_path}, excluding {test_lang}")
    
    # Load all language files except test language
    train_files = []
    val_files = []
    
    for lang in LANGS:
        if lang != test_lang:
            train_file = f"{data_path}/mushroom.{lang}-val.v2.jsonl"
            if os.path.exists(train_file):
                train_files.append(train_file)
                print(f"Added training file: {train_file}")
    
    # Use English validation for validation (if not test language)
    if test_lang != 'en':
        val_file = f"{data_path}/mushroom.en-val.v2.jsonl"
        if os.path.exists(val_file):
            val_files.append(val_file)
            print(f"Added validation file: {val_file}")
    else:
        # Use another language for validation if English is test language
        val_file = f"{data_path}/mushroom.es-val.v2.jsonl"
        if os.path.exists(val_file):
            val_files.append(val_file)
            print(f"Added validation file: {val_file}")
    
    # Create dataset
    dataset = load_dataset('json', 
                          data_files={'train': train_files, 'validation': val_files})
    
    print(f"Dataset created:")
    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")
    
    return dataset

def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize text and align labels"""
    tokenized_inputs = tokenizer(
        examples['model_output_text'],
        truncation=True,
        padding=True,
        max_length=512,
        return_offsets_mapping=True,
        is_split_into_words=False
    )
    
    labels = []
    for i, (hard_labels, offset_mapping) in enumerate(zip(examples['hard_labels'], tokenized_inputs['offset_mapping'])):
        label = [0] * len(offset_mapping)
        
        # Mark tokens that are part of hallucinated spans
        for start, end in hard_labels:
            for j, (token_start, token_end) in enumerate(offset_mapping):
                if token_start >= start and token_end <= end and token_start != token_end:
                    label[j] = 1
        
        labels.append(label)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def train_model(test_lang='en', data_path='./data/val', output_dir='./results', epochs=2):
    """Train the baseline model with reduced epochs for validation"""
    print(f"\n{'='*60}")
    print(f"TRAINING BASELINE MODEL - VALIDATION MODE")
    print(f"{'='*60}")
    print(f"Test Language (excluded): {test_lang}")
    print(f"Data Path: {data_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Training Epochs: {epochs}")
    print(f"Model: {MODEL_NAME}")
    
    start_time = time.time()
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL_LIST),
        id2label={0: "O", 1: "HALLUCINATION"},
        label2id={"O": 0, "HALLUCINATION": 1}
    )
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = create_dataset(data_path, test_lang)
    
    # Tokenize dataset
    print(f"\nTokenizing dataset...")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    print(f"Tokenized dataset:")
    print(f"  Training samples: {len(tokenized_datasets['train'])}")
    print(f"  Validation samples: {len(tokenized_datasets['validation'])}")
    
    # Training arguments - reduced for validation
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',  # Align with eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # Reduced batch size
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,  # Reduced epochs
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=None,  # Disable reporting
    )
    
    # Load metric
    print(f"\nLoading evaluation metric...")
    metric = load('seqeval')
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Convert to label names
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            true_pred = []
            true_label = []
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:  # Ignore special tokens
                    true_pred.append("HALLUCINATION" if pred_id == 1 else "O")
                    true_label.append("HALLUCINATION" if label_id == 1 else "O")
            true_predictions.append(true_pred)
            true_labels.append(true_label)
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        # Calculate additional metrics
        flat_true = [label for sublist in true_labels for label in sublist]
        flat_pred = [pred for sublist in true_predictions for pred in sublist]
        
        # Token-level accuracy
        token_accuracy = sum(1 for t, p in zip(flat_true, flat_pred) if t == p) / len(flat_true)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "token_accuracy": token_accuracy
        }
    
    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING")
    print(f"{'='*60}")
    
    train_result = trainer.train()
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    print(f"\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Save training summary
    training_summary = {
        "model_name": MODEL_NAME,
        "test_language_excluded": test_lang,
        "training_samples": len(tokenized_datasets['train']),
        "validation_samples": len(tokenized_datasets['validation']),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "epochs": epochs,
        "final_training_loss": train_result.training_loss,
        "final_evaluation": eval_results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{output_dir}/training_summary.json", 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"\nTraining summary saved to {output_dir}/training_summary.json")
    
    return trainer, training_summary

def main():
    """Main function for training validation"""
    parser = argparse.ArgumentParser(description='Baseline Model Training Validation')
    parser.add_argument('--test_lang', default='en', help='Language to exclude from training')
    parser.add_argument('--data_path', default='./data/val', help='Path to training data')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    trainer, summary = train_model(
        test_lang=args.test_lang,
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    
    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Training summary: {args.output_dir}/training_summary.json")
    
    return trainer, summary

if __name__ == "__main__":
    main()

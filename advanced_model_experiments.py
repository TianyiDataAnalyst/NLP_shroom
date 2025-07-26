#!/usr/bin/env python3
"""
Section 7: Advanced Model Experiments for Mushroom Task 2025
Enhanced training with full dataset, alternative architectures, and optimization techniques
"""

import torch
import json
import os
import argparse
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    Trainer, TrainingArguments, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset
from evaluate import load
import warnings
warnings.filterwarnings('ignore')

# Disable WandB logging
os.environ["WANDB_MODE"] = "disabled"

# Model configurations for experiments
MODEL_CONFIGS = {
    'xlm-roberta-base': {
        'name': 'FacebookAI/xlm-roberta-base',
        'description': 'Baseline XLM-RoBERTa Base (278M params)',
        'batch_size': 16,
        'learning_rate': 2e-5
    },
    'xlm-roberta-large': {
        'name': 'FacebookAI/xlm-roberta-large', 
        'description': 'XLM-RoBERTa Large (550M params)',
        'batch_size': 8,  # Reduced due to larger model
        'learning_rate': 1e-5
    },
    'mbert-base': {
        'name': 'google-bert/bert-base-multilingual-cased',
        'description': 'Multilingual BERT Base (178M params)',
        'batch_size': 16,
        'learning_rate': 2e-5
    }
}

LABEL_LIST = [0, 1]
LANGS = ['ar', 'de', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh']

class AdvancedTrainingFramework:
    """Advanced training framework with full dataset and optimization techniques"""
    
    def __init__(self, test_lang='en', data_path='./data', output_base_dir='./advanced_results'):
        self.test_lang = test_lang
        self.data_path = data_path
        self.output_base_dir = output_base_dir
        self.metric = load('seqeval')
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        print(f"Advanced Training Framework Initialized")
        print(f"Test Language: {test_lang}")
        print(f"Data Path: {data_path}")
        print(f"Output Directory: {output_base_dir}")
    
    def load_full_dataset(self):
        """Load the complete labeled dataset with enhanced train/validation split"""
        print(f"\n{'='*60}")
        print(f"LOADING FULL LABELED DATASET")
        print(f"{'='*60}")

        # Load all labeled validation data (this is our actual training data)
        train_files = []

        # Use labeled validation data from all languages except test language
        val_dir = Path(f"{self.data_path}/val")
        if val_dir.exists():
            for lang in LANGS:
                if lang != self.test_lang:
                    val_file = val_dir / f"mushroom.{lang}-val.v2.jsonl"
                    if val_file.exists():
                        train_files.append(str(val_file))
                        print(f"Added labeled data: {val_file}")

        # Load all available labeled data
        if not train_files:
            raise ValueError(f"No labeled files found for languages excluding {self.test_lang}")

        print(f"\nLoading {len(train_files)} labeled files...")
        dataset = load_dataset('json', data_files={'train': train_files})
        full_labeled_data = dataset['train']

        print(f"Total labeled samples loaded: {len(full_labeled_data)}")

        # Enhanced train/validation split with stratification
        # Use 75/25 split to ensure sufficient validation data
        train_indices, val_indices = train_test_split(
            range(len(full_labeled_data)),
            test_size=0.25,
            random_state=42,
            stratify=[item['lang'] for item in full_labeled_data]  # Stratify by language
        )

        train_dataset = Dataset.from_dict({
            key: [full_labeled_data[i][key] for i in train_indices]
            for key in full_labeled_data.column_names
        })

        val_dataset = Dataset.from_dict({
            key: [full_labeled_data[i][key] for i in val_indices]
            for key in full_labeled_data.column_names
        })

        print(f"Training split: {len(train_dataset)} samples")
        print(f"Validation split: {len(val_dataset)} samples")

        # Analyze language distribution
        train_langs = [item['lang'] for item in train_dataset]
        val_langs = [item['lang'] for item in val_dataset]

        print(f"\nTraining language distribution:")
        for lang in set(train_langs):
            count = train_langs.count(lang)
            print(f"  {lang}: {count} samples")

        print(f"\nValidation language distribution:")
        for lang in set(val_langs):
            count = val_langs.count(lang)
            print(f"  {lang}: {count} samples")

        # Analyze hallucination distribution
        train_hall_counts = []
        val_hall_counts = []

        for item in train_dataset:
            train_hall_counts.append(len(item['hard_labels']))

        for item in val_dataset:
            val_hall_counts.append(len(item['hard_labels']))

        print(f"\nHallucination span statistics:")
        print(f"  Training - Total spans: {sum(train_hall_counts)}, Avg per sample: {np.mean(train_hall_counts):.2f}")
        print(f"  Validation - Total spans: {sum(val_hall_counts)}, Avg per sample: {np.mean(val_hall_counts):.2f}")

        return {'train': train_dataset, 'validation': val_dataset}
    
    def tokenize_and_align_labels(self, examples, tokenizer):
        """Enhanced tokenization with better label alignment"""
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
            
            # Improved label alignment - mark tokens that overlap with hallucinated spans
            for start, end in hard_labels:
                for j, (token_start, token_end) in enumerate(offset_mapping):
                    # Token overlaps with hallucination span
                    if (token_start < end and token_end > start and 
                        token_start != token_end):  # Ignore special tokens
                        label[j] = 1
            
            labels.append(label)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def compute_metrics(self, p):
        """Enhanced metrics computation"""
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
        
        # Calculate seqeval metrics
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        
        # Calculate additional metrics
        flat_true = [label for sublist in true_labels for label in sublist]
        flat_pred = [pred for sublist in true_predictions for pred in sublist]
        
        # Token-level accuracy
        token_accuracy = sum(1 for t, p in zip(flat_true, flat_pred) if t == p) / len(flat_true)
        
        # Hallucination detection metrics
        hall_true = [1 if label == "HALLUCINATION" else 0 for label in flat_true]
        hall_pred = [1 if pred == "HALLUCINATION" else 0 for pred in flat_pred]
        
        tp = sum(1 for t, p in zip(hall_true, hall_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(hall_true, hall_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(hall_true, hall_pred) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "span_precision": results["overall_precision"],
            "span_recall": results["overall_recall"],
            "span_f1": results["overall_f1"],
            "span_accuracy": results["overall_accuracy"],
            "token_accuracy": token_accuracy,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1
        }
    
    def train_model(self, model_config_name, epochs=12, use_early_stopping=True):
        """Train a model with advanced configuration"""
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL: {model_config_name.upper()}")
        print(f"{'='*60}")
        
        config = MODEL_CONFIGS[model_config_name]
        model_name = config['name']
        
        print(f"Model: {model_name}")
        print(f"Description: {config['description']}")
        print(f"Epochs: {epochs}")
        print(f"Early Stopping: {use_early_stopping}")
        
        start_time = time.time()
        
        # Create output directory for this model
        output_dir = os.path.join(self.output_base_dir, f"{model_config_name}_epochs_{epochs}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tokenizer and model
        print(f"\nLoading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL_LIST),
            id2label={0: "O", 1: "HALLUCINATION"},
            label2id={"O": 0, "HALLUCINATION": 1}
        )
        
        # Load dataset
        print(f"\nLoading dataset...")
        dataset = self.load_full_dataset()
        
        # Tokenize dataset
        print(f"\nTokenizing dataset...")
        tokenized_datasets = {}
        for split in ['train', 'validation']:
            tokenized_datasets[split] = dataset[split].map(
                lambda examples: self.tokenize_and_align_labels(examples, tokenizer),
                batched=True,
                remove_columns=dataset[split].column_names
            )
            # Remove offset_mapping which is not needed for training
            tokenized_datasets[split] = tokenized_datasets[split].remove_columns(['offset_mapping'])
        
        print(f"Tokenized dataset:")
        print(f"  Training samples: {len(tokenized_datasets['train'])}")
        print(f"  Validation samples: {len(tokenized_datasets['validation'])}")
        
        # Enhanced training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=config['learning_rate'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            num_train_epochs=epochs,
            weight_decay=0.01,
            warmup_ratio=0.1,  # Warmup for 10% of training
            logging_steps=50,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_span_f1",
            greater_is_better=True,
            report_to=None,
            gradient_accumulation_steps=2,  # Effective batch size doubling
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # Callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Create trainer
        print(f"\nInitializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Train model
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING - {model_config_name.upper()}")
        print(f"{'='*60}")
        
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED - {model_config_name.upper()}")
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
        print(f"FINAL EVALUATION RESULTS - {model_config_name.upper()}")
        print(f"{'='*60}")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Save training summary
        training_summary = {
            "model_config": model_config_name,
            "model_name": model_name,
            "model_description": config['description'],
            "test_language_excluded": self.test_lang,
            "training_samples": len(tokenized_datasets['train']),
            "validation_samples": len(tokenized_datasets['validation']),
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "epochs": epochs,
            "early_stopping": use_early_stopping,
            "final_training_loss": train_result.training_loss,
            "final_evaluation": eval_results,
            "training_args": training_args.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{output_dir}/training_summary.json", 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"\nTraining summary saved to {output_dir}/training_summary.json")
        
        return trainer, training_summary, output_dir

def main():
    """Main function for advanced model experiments"""
    parser = argparse.ArgumentParser(description='Advanced Model Experiments - Section 7')
    parser.add_argument('--test_lang', default='en', help='Language to exclude from training')
    parser.add_argument('--data_path', default='./data', help='Path to data directory')
    parser.add_argument('--output_dir', default='./advanced_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--models', nargs='+', default=['xlm-roberta-base'], 
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Models to train')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Use early stopping')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"SECTION 7: ADVANCED MODEL EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Test Language: {args.test_lang}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Models to train: {args.models}")
    print(f"Early Stopping: {args.early_stopping}")
    
    # Initialize framework
    framework = AdvancedTrainingFramework(
        test_lang=args.test_lang,
        data_path=args.data_path,
        output_base_dir=args.output_dir
    )
    
    # Train all specified models
    results = {}
    for model_name in args.models:
        try:
            trainer, summary, output_dir = framework.train_model(
                model_config_name=model_name,
                epochs=args.epochs,
                use_early_stopping=args.early_stopping
            )
            results[model_name] = {
                'trainer': trainer,
                'summary': summary,
                'output_dir': output_dir
            }
            print(f"\n✅ Successfully trained {model_name}")
            
        except Exception as e:
            print(f"\n❌ Failed to train {model_name}: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    print(f"\n{'='*80}")
    print(f"ADVANCED MODEL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    
    # Summary of results
    for model_name, result in results.items():
        if 'error' in result:
            print(f"❌ {model_name}: FAILED - {result['error']}")
        else:
            summary = result['summary']
            eval_results = summary['final_evaluation']
            print(f"✅ {model_name}:")
            print(f"   Span F1: {eval_results.get('eval_span_f1', 0):.4f}")
            print(f"   Token F1: {eval_results.get('eval_token_f1', 0):.4f}")
            print(f"   Training time: {summary['training_time_minutes']:.2f} minutes")
    
    return results

if __name__ == "__main__":
    main()

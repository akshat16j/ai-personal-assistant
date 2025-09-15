#!/usr/bin/env python3
"""
Training script for fine-tuning FLAN-T5-Small models for email triage and event extraction.
Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

import json
import os
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np

# Create output directory for LoRA adapters
LORA_OUTPUT_DIR = Path("models/lora_adapters")
LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_triage_data(examples, tokenizer):
    """Preprocess data for triage model training."""
    inputs = [f"Classify email priority: {text}" for text in examples["text"]]
    targets = examples["label"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=32, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_extraction_data(examples, tokenizer):
    """Preprocess data for extraction model training."""
    inputs = [f"Extract event information: {text}" for text in examples["input_text"]]
    targets = examples["output_text"]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels = tokenizer(targets, max_length=256, truncation=True, padding=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def setup_lora_config():
    """Setup LoRA configuration for parameter-efficient fine-tuning."""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"]
    )

def train_triage_model():
    """Train the email triage model using LoRA."""
    print("Starting triage model training...")
    
    # Load model and tokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Load and preprocess dataset
    triage_data = load_dataset("data/triage_dataset_json.json")
    dataset = Dataset.from_dict({
        "text": [item["text"] for item in triage_data],
        "label": [item["label"] for item in triage_data]
    })
    
    # Preprocess data
    tokenized_dataset = dataset.map(
        lambda x: preprocess_triage_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./triage_training_output",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save LoRA adapters
    triage_output_path = LORA_OUTPUT_DIR / "triage"
    triage_output_path.mkdir(exist_ok=True)
    model.save_pretrained(str(triage_output_path))
    tokenizer.save_pretrained(str(triage_output_path))
    
    print(f"Triage model saved to {triage_output_path}")

def train_extraction_model():
    """Train the event extraction model using LoRA."""
    print("Starting extraction model training...")
    
    # Load model and tokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Load and preprocess dataset
    extraction_data = load_dataset("data/extraction_dataset.json")
    dataset = Dataset.from_dict({
        "input_text": [item["input_text"] for item in extraction_data],
        "output_text": [item["output_text"] for item in extraction_data]
    })
    
    # Preprocess data
    tokenized_dataset = dataset.map(
        lambda x: preprocess_extraction_data(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./extraction_training_output",
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save LoRA adapters
    extraction_output_path = LORA_OUTPUT_DIR / "extraction"
    extraction_output_path.mkdir(exist_ok=True)
    model.save_pretrained(str(extraction_output_path))
    tokenizer.save_pretrained(str(extraction_output_path))
    
    print(f"Extraction model saved to {extraction_output_path}")

def main():
    """Main training function."""
    print("=" * 60)
    print("AI Daily Briefing Agent V2 - Model Training")
    print("=" * 60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Train both models
    train_triage_model()
    print("\n" + "-" * 40 + "\n")
    train_extraction_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print(f"LoRA adapters saved in: {LORA_OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
"""
Script for running LoRA fine-tuning on a pre-trained language model.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from model.load_base_model import ModelLoader
from data.prepare_dataset import DatasetPreparator


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def load_training_config(config_path: Path) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_outputs(output_dir: Path, model, tokenizer, training_args, final_metrics=None):
    """
    Save all training outputs in an organized structure.
    
    Args:
        output_dir: Base output directory
        model: Trained model
        tokenizer: Tokenizer
        training_args: Training arguments used
        final_metrics: Optional dictionary of final training metrics
    """
    # Save model and tokenizer
    model_dir = output_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save training arguments
    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(training_args.to_dict(), f, indent=2)
        
    # Save final metrics if available
    if final_metrics:
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
            
    # Save GPU info
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
        with open(output_dir / 'gpu_info.json', 'w') as f:
            json.dump(gpu_info, f, indent=2)


def run_training(model, tokenizer, dataset, output_dir="./results"):
    """
    Run LoRA fine-tuning on the model
    """
    logger = logging.getLogger(__name__)

    # Prepare model for k-bit training
    logger.info("Preparing model for k-bit training")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    logger.info("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=32,  # attention heads
        lora_alpha=64,  # alpha scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Add LoRA adaptor
    model = get_peft_model(model, lora_config)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        report_to="none"
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    logger.info("Starting training")
    trainer.train()

    # Save the final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_args.json",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="configs/lora_config.json",
        help="Path to LoRA configuration file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data/fastapi_mined_dataset.json",
        help="Path to FastAPI training dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Custom output directory"
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use dummy data for testing"
    )
    
    args = parser.parse_args()
    
    run_training(
        training_config_path=args.training_config,
        lora_config_path=args.lora_config,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_dummy_data=args.use_dummy_data
    ) 

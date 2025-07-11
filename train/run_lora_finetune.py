"""
Script for running QLoRA fine-tuning on Code Llama 7B Instruct Model.
"""

import json
import logging
from pathlib import Path
import argparse
from datetime import datetime

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def setup_logging(log_dir: Path):
    """
    Setup logging configuration.

    Args:
        log_dir: Path to logging file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the JSON config file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_outputs(output_dir: Path, model, tokenizer, training_args, final_metrics=None):
    """
    Save all training outputs in an organized structure.
    
    Args:
        output_dir: Base output directory.
        model: Trained model.
        tokenizer: Tokenizer.
        training_args: Training arguments used.
        final_metrics: Optional dictionary of final training metrics.
    """
    model_dir = output_dir / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(output_dir / 'training_args.json', 'w') as f:
        json.dump(training_args.to_dict(), f, indent=2)

    if final_metrics:
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)

    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
        with open(output_dir / 'gpu_info.json', 'w') as f:
            json.dump(gpu_info, f, indent=2)


def run_training(
    model,
    tokenizer,
    dataset,
    training_config_path="configs/training_args.json",
    hub_config_path="configs/hub_config.json",
    output_dir="./results"
):
    """
    Run LoRA fine-tuning on the model using configuration files.
    
    Note: Model should already be quantized and have LoRA adapters applied
    via the ModelLoader class before calling this function.
    
    Args:
        model: The model to fine-tune (already with LoRA adapters).
        tokenizer: The tokenizer.
        dataset: Training dataset.
        training_config_path: Path to training arguments JSON config.
        hub_config_path: Path to HuggingFace Hub configuration JSON config.
        output_dir: Directory for saving checkpoints.

    Returns:
        Trained model.
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading training config from: {training_config_path}")
    training_config_dict = load_config(training_config_path)
    
    logger.info(f"Loading hub config from: {hub_config_path}")
    hub_config_dict = load_config(hub_config_path)

    hub_model_id = None
    if hub_config_dict.get("push_to_hub", False):
        hub_model_id = f"{hub_config_dict['hub_model_id_prefix']}-{datetime.now().strftime('%Y%m%d')}"
        logger.info(f"Will push to Hub: {hub_model_id}")

    logger.info("Setting up training arguments from config")

    training_config_dict.update({
        "output_dir": output_dir,
        "push_to_hub": hub_config_dict.get("push_to_hub", False),
        "hub_model_id": hub_model_id,
        "hub_strategy": hub_config_dict.get("hub_strategy", "every_save") if hub_config_dict.get("push_to_hub", False) else "no"
    })
    
    training_args = TrainingArguments(**training_config_dict)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    logger.info("Starting training")
    trainer.train()

    if hub_config_dict.get("push_to_hub", False) and hub_model_id:
        logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
        trainer.push_to_hub()
        logger.info("Model pushed successfully!")
    else:
        logger.info("Skipping push to Hub")

    return trainer.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LoRA fine-tuning")
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/training_args.json",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--hub_config",
        type=str, 
        default="configs/hub_config.json",
        help="Path to HuggingFace Hub configuration file"
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
        default="./results",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Note: This main function would need model, tokenizer, and dataset
    # In practice, this is called from the notebook where these are available
    print("This script is designed to be imported and used from the notebook.")
    print("Use: from train.run_lora_finetune import run_training")

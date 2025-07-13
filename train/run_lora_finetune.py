"""
Script for running QLoRA fine-tuning on Code Llama 7B Instruct Model.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import argparse

from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.data.data_collator import default_data_collator


def setup_logging(log_dir: Path):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files.
    """
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Dictionary with configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_outputs(output_dir: Path, model, tokenizer, training_args, final_metrics=None):
    """
    Save training outputs including model, tokenizer, and metrics.
    
    Args:
        output_dir: Directory to save outputs.
        model: Trained model.
        tokenizer: Tokenizer.
        training_args: Training arguments used.
        final_metrics: Final training metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saving training outputs to {output_dir}")
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    config_file = output_dir / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(training_args.to_dict(), f, indent=2)
    
    if final_metrics:
        metrics_file = output_dir / "final_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, indent=2)


def run_training(
    model,
    tokenizer,
    dataset,
    training_config_path="configs/training_args.json",
    hub_config_path="configs/hub_config.json",
    output_dir="./results"
):
    """
    Run LoRA fine-tuning with comprehensive logging and configuration.

    Args:
        model: Model to train.
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

    # Split dataset for evaluation if eval_strategy is not "no"
    train_dataset = dataset
    eval_dataset = None
    
    if training_args.eval_strategy != "no":
        logger.info("Splitting dataset for evaluation")
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Evaluation examples: {len(eval_dataset)}")
    else:
        logger.info(f"Training examples: {len(train_dataset)}")

    # Add callbacks
    callbacks = []
    if eval_dataset is not None and training_args.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks
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

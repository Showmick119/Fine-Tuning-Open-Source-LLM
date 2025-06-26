"""
Script for running LoRA fine-tuning on a pre-trained language model.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

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


def run_training(
    training_config_path: str,
    lora_config_path: str,
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_dummy_data: bool = False
):
    """
    Run the complete fine-tuning pipeline.
    
    Args:
        training_config_path: Path to training configuration file
        lora_config_path: Path to LoRA configuration file
        data_path: Optional path to training data
        output_dir: Optional custom output directory
        use_dummy_data: Whether to use dummy data for testing
    """
    # Setup paths
    training_config_path = Path(training_config_path)
    output_dir = Path(output_dir) if output_dir else Path('outputs')
    log_dir = output_dir / 'logs'
    
    # Setup logging
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    logger.info("Starting fine-tuning process")
    
    # Load configurations
    training_config = load_training_config(training_config_path)
    if output_dir:
        training_config['output_dir'] = str(output_dir / 'checkpoints')
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer")
    model_loader = ModelLoader(lora_config_path)
    model, tokenizer = model_loader.load_base_model()
    model = model_loader.add_lora_adapter(model)
    
    # Prepare dataset
    logger.info("Preparing dataset")
    data_preparator = DatasetPreparator(
        tokenizer=tokenizer,
        max_length=training_config.get('max_seq_length', 512),
        data_path=data_path
    )
    dataset = data_preparator.prepare_dataset(use_dummy=use_dummy_data)
    
    # Split dataset if evaluation is enabled
    if training_config.get('do_eval', False):
        dataset = dataset.train_test_split(
            test_size=0.1,
            shuffle=True,
            seed=42
        )
    
    # Initialize training arguments
    training_args = TrainingArguments(**training_config)
    
    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'] if training_config.get('do_eval', False) else dataset,
        eval_dataset=dataset['test'] if training_config.get('do_eval', False) else None,
        data_collator=data_collator,
    )
    
    try:
        # Run training
        logger.info("Starting training")
        trainer.train()
        
        # Save the final model
        logger.info("Saving final model")
        trainer.save_model()
        
        # Save the tokenizer
        tokenizer.save_pretrained(training_config['output_dir'])
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


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
        help="Path to training data file"
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
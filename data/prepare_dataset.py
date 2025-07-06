"""
Module for preparing and preprocessing FastAPI training dataset.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import logging

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

class DatasetPreparator:
    """Handles FastAPI dataset preparation and preprocessing for CodeLlama fine-tuning."""
    
    def __init__(self, dataset_path: str = "data/fastapi_mined_dataset.json"):
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)

    def load_and_prepare_dataset(self) -> Dataset:
        """
        Load and prepare the FastAPI dataset for fine-tuning
        """
        # Load raw data
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Prepare data in the format expected by the model
        processed_data = []
        for item in raw_data:
            # Format the instruction and input
            instruction = f"Category: {item['category']}\nDifficulty: {item['difficulty']}\n\n{item['instruction']}"
            
            # Combine input and output with proper formatting
            input_text = item.get('input', '')
            if input_text:
                instruction += f"\n\nInput:\n{input_text}"
            
            # Create the formatted example
            processed_data.append({
                "instruction": instruction,
                "output": item['output']
            })

        # Create HuggingFace dataset
        dataset = Dataset.from_list(processed_data)
        self.logger.info(f"Created dataset with {len(dataset)} examples")
        
        return dataset

    def format_prompt(self, instruction: str, input_text: str = "", category: str = "", difficulty: str = "") -> str:
        """
        Format the instruction and input into a prompt suitable for CodeLlama.
        
        Args:
            instruction: The main instruction/task
            input_text: Optional input context
            category: FastAPI category (auth, endpoints, etc.)
            difficulty: Task difficulty level
        """
        # Build context section
        context = f"Category: {category}\nDifficulty: {difficulty}\n\n" if category and difficulty else ""
        
        # Format with CodeLlama prompt structure
        if input_text and input_text.strip():
            return f"<s>[INST] {context}{instruction}\n\nInput:\n{input_text} [/INST]"
        return f"<s>[INST] {context}{instruction} [/INST]"
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize and format examples for training.
        
        Args:
            examples: Batch of examples with instruction, input, output, and metadata
            
        Returns:
            Processed examples with input_ids and labels
        """
        model_inputs = {"input_ids": [], "labels": [], "attention_mask": []}
        
        for idx in range(len(examples["instruction"])):
            # Format prompt with metadata
            prompt = self.format_prompt(
                instruction=examples["instruction"][idx],
                input_text=examples["input"][idx],
                category=examples["category"][idx],
                difficulty=examples["difficulty"][idx]
            )
            
            # Ensure output ends with EOS token
            output = examples["output"][idx].strip()
            if not output.endswith(self.tokenizer.eos_token):
                output = output + self.tokenizer.eos_token
                
            # Combine prompt and output
            full_text = prompt + output
            
            # Tokenize with truncation safeguards
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            
            # Create labels, setting prompt tokens to -100 (ignored in loss)
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            
            labels = tokenized["input_ids"].copy()
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
            
            model_inputs["input_ids"].append(tokenized["input_ids"])
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append(tokenized["attention_mask"])
        
        return model_inputs
    
    def prepare_dataset(self, use_dummy: bool = False) -> Dataset:
        """
        Prepare the FastAPI dataset for training.
        
        Args:
            use_dummy: Ignored (kept for backwards compatibility)
            
        Returns:
            Processed dataset ready for training
        """
        # Load dataset
        dataset = self.load_and_prepare_dataset()
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing FastAPI examples",
        )
        
        return processed_dataset


if __name__ == "__main__":
    # Example usage
    preparator = DatasetPreparator(
        dataset_path="data/data/fastapi_mined_dataset.json"
    )
    
    # Test dataset preparation
    dataset = preparator.prepare_dataset()
    print(f"✓ Prepared dataset with {len(dataset)} examples")
    print(f"✓ Input IDs shape: {len(dataset[0]['input_ids'])}")
    print(f"✓ Labels shape: {len(dataset[0]['labels'])}") 
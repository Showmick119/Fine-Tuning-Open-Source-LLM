"""
Module for preparing and preprocessing datasets for LLM fine-tuning.
"""

from typing import Dict, List, Optional, Union
from pathlib import Path
import json

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

class DatasetPreparator:
    """Handles dataset preparation and preprocessing for LLM fine-tuning."""
    
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        max_length: int = 512,
        data_path: Optional[str] = None
    ):
        """
        Initialize the dataset preparator.
        
        Args:
            tokenizer: HuggingFace tokenizer or path/name
            max_length: Maximum sequence length for tokenization
            data_path: Optional path to custom dataset
        """
        self.tokenizer = (
            tokenizer if isinstance(tokenizer, PreTrainedTokenizer)
            else AutoTokenizer.from_pretrained(tokenizer)
        )
        self.max_length = max_length
        self.data_path = Path(data_path) if data_path else None
        
    def load_dummy_data(self) -> Dataset:
        """Create a small dummy dataset for testing purposes."""
        dummy_data = {
            "instruction": [
                "Write a poem about AI",
                "Explain quantum computing",
                "Write a story about space exploration"
            ],
            "input": [""] * 3,  # Empty inputs for instruction-only examples
            "output": [
                "In circuits of light and neural streams,\nAI dances in digital dreams...",
                "Quantum computing leverages quantum mechanical phenomena...",
                "The starship Horizon glided silently through the cosmic void..."
            ]
        }
        return Dataset.from_dict(dummy_data)
    
    def load_custom_data(self) -> Dataset:
        """Load custom dataset from JSON or JSONL file."""
        if not self.data_path or not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different data formats
            if isinstance(data, list):
                # Data is a list of examples (CodeAlpaca format)
                return Dataset.from_list(data)
            else:
                # Data is a dictionary with column names as keys
                return Dataset.from_dict(data)
        else:  # Assume JSONL
            return Dataset.from_json(str(self.data_path))
            
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the instruction and input into a prompt suitable for CodeLlama."""
        if input_text and input_text.strip():
            return f"[INST] {instruction}\n\nInput:\n{input_text} [/INST]"
        return f"[INST] {instruction} [/INST]"
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize and format examples for training.
        
        Args:
            examples: Batch of examples with instruction, input, and output fields
            
        Returns:
            Processed examples with input_ids and labels
        """
        model_inputs = {"input_ids": [], "labels": []}
        
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            # Format prompt and combine with output
            prompt = self.format_prompt(instruction, inp)
            full_text = prompt + output
            
            # Tokenize
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
        
        return model_inputs
    
    def prepare_dataset(self, use_dummy: bool = False) -> Dataset:
        """
        Prepare the complete dataset for training.
        
        Args:
            use_dummy: Whether to use dummy data instead of loading from file
            
        Returns:
            Processed dataset ready for training
        """
        # Load raw data
        dataset = self.load_dummy_data() if use_dummy else self.load_custom_data()
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset",
        )
        
        return processed_dataset


if __name__ == "__main__":
    # Example usage
    preparator = DatasetPreparator(
        tokenizer="mistralai/Mistral-7B-v0.1",
        max_length=512
    )
    
    # Prepare dummy dataset
    dataset = preparator.prepare_dataset(use_dummy=True)
    print(f"Prepared dataset with {len(dataset)} examples") 
"""
Module for preparing and preprocessing FastAPI training dataset.
"""

from typing import Dict, List
import json
import logging
import re

from datasets import Dataset

class DatasetPreparator:
    """
    Handles FastAPI dataset preparation and preprocessing for CodeLlama fine-tuning.
    """
    
    def __init__(self, dataset_path: str = "data/data/fastapi_mined_dataset.json"):
        """
        Initialize the dataset preparator.
        
        Args:
            dataset_path: Path to the dataset.
        """
        self.dataset_path = dataset_path
        self.logger = logging.getLogger(__name__)

    def enhance_code_snippet(self, code: str) -> str:
        """
        Enhance incomplete code snippets by adding proper imports and structure.
        
        Args:
            code: Raw code snippet that may be incomplete
            
        Returns:
            Complete FastAPI code with proper imports and structure
        """
        code = code.strip()
        
        # Check if code already has imports
        has_fastapi_import = bool(re.search(r"from\s+fastapi\s+import|import\s+fastapi", code, re.IGNORECASE))
        has_app_instance = bool(re.search(r"app\s*=\s*FastAPI\s*\(", code, re.IGNORECASE))
        
        # Start building the enhanced code
        enhanced_code = []
        
        # Add necessary imports if missing
        if not has_fastapi_import:
            enhanced_code.append("from fastapi import FastAPI, HTTPException, Depends, status")
            
        # Add common imports based on code content
        if "Session" in code and "sqlalchemy" not in code.lower():
            enhanced_code.append("from sqlalchemy.orm import Session")
        if "JSONResponse" in code:
            enhanced_code.append("from fastapi.responses import JSONResponse")
        if "RedirectResponse" in code:
            enhanced_code.append("from fastapi.responses import RedirectResponse")
        if "List[" in code or "Dict[" in code:
            enhanced_code.append("from typing import List, Dict, Any")
        if "BaseModel" in code:
            enhanced_code.append("from pydantic import BaseModel")
        
        # Add empty line after imports
        if enhanced_code:
            enhanced_code.append("")
        
        # Add app instance if missing
        if not has_app_instance and "@app." in code:
            enhanced_code.append("app = FastAPI()")
            enhanced_code.append("")
        
        # Add the original code
        enhanced_code.append(code)
        
        return "\n".join(enhanced_code)

    def load_and_prepare_dataset(self) -> Dataset:
        """
        Load and prepare the FastAPI dataset for fine-tuning.

        Returns:
            FastAPI data wrapped inside a Dataset object from the transformers library.
        """
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed_data = []
        for item in raw_data:
            instruction = f"Category: {item['category']}\nDifficulty: {item['difficulty']}\n\n{item['instruction']}"

            input_text = item.get('input', '')
            if input_text:
                instruction += f"\n\nInput:\n{input_text}"

            # Enhance the output code to be complete and runnable
            enhanced_output = self.enhance_code_snippet(item['output'])

            processed_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": enhanced_output,
                "category": item['category'],
                "difficulty": item['difficulty']
            })

        dataset = Dataset.from_list(processed_data)
        self.logger.info(f"Created dataset with {len(dataset)} examples")
        
        return dataset

    def format_prompt(self, instruction: str, input_text: str = "", category: str = "", difficulty: str = "") -> str:
        """
        Format the instruction and input into a prompt suitable for CodeLlama.
        
        Args:
            instruction: The main instruction/task.
            input_text: Optional input context.
            category: FastAPI category (auth, endpoints, etc.).
            difficulty: Task difficulty level.
        
        Returns:
            Processed string with the appropiate CodeLlama tags.
        """
        context = f"Category: {category}\nDifficulty: {difficulty}\n\n" if category and difficulty else ""

        if input_text and input_text.strip():
            return f"<s>[INST] {context}{instruction}\n\nInput:\n{input_text} [/INST]"
        return f"<s>[INST] {context}{instruction} [/INST]"
    
    def preprocess_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize and format examples for training.
        
        Args:
            examples: Batch of examples with instruction, input, output, and metadata.
            
        Returns:
            Processed examples with input_ids and labels.
        """
        model_inputs = {"input_ids": [], "labels": [], "attention_mask": []}
        
        for idx in range(len(examples["instruction"])):
            prompt = self.format_prompt(
                instruction=examples["instruction"][idx],
                input_text=examples["input"][idx],
                category=examples["category"][idx],
                difficulty=examples["difficulty"][idx]
            )

            output = examples["output"][idx].strip()
            if not output.endswith(self.tokenizer.eos_token):
                output = output + self.tokenizer.eos_token

            full_text = prompt + output

            # Tokenize with FIXED padding to ensure exact same length every time
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()

            # Find where the prompt ends and the output begins
            prompt_tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt"
            )
            prompt_length = prompt_tokenized["input_ids"].shape[1]

            # Create labels - mask the prompt tokens with -100
            labels = input_ids.clone()
            labels[:prompt_length] = -100

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["labels"].append(labels)

        return model_inputs

    def prepare_dataset(self) -> Dataset:
        """
        Prepare the dataset for training by applying preprocessing function.
        
        Returns:
            Dataset: Processed dataset ready for training.
        """
        if not hasattr(self, 'tokenizer'):
            raise AttributeError("Tokenizer not set. Please set self.tokenizer before calling prepare_dataset()")
        
        if not hasattr(self, 'max_length'):
            self.max_length = 512
            self.logger.warning("max_length not set. Using default value of 512")

        self.logger.info("Loading and preparing dataset...")
        dataset = self.load_and_prepare_dataset()

        self.logger.info("Tokenizing dataset...")
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing FastAPI examples"
        )

        self.logger.info(f"Dataset prepared with {len(dataset)} examples")
        return dataset


if __name__ == "__main__":
    preparator = DatasetPreparator(
        dataset_path="data/data/fastapi_mined_dataset.json"
    )

    dataset = preparator.prepare_dataset()
    print(f"✓ Prepared dataset with {len(dataset)} examples")
    print(f"✓ Input IDs shape: {len(dataset[0]['input_ids'])}")
    print(f"✓ Labels shape: {len(dataset[0]['labels'])}") 
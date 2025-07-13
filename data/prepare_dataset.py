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
        
        # Check what's already present
        has_fastapi_import = bool(re.search(r"from\s+fastapi\s+import|import\s+fastapi", code, re.IGNORECASE))
        has_app_instance = bool(re.search(r"app\s*=\s*FastAPI\s*\(", code, re.IGNORECASE))
        has_router_instance = bool(re.search(r"router\s*=\s*APIRouter\s*\(", code, re.IGNORECASE))
        
        # Check if code uses app or router decorators
        uses_app_decorator = bool(re.search(r"@\s*app\.", code, re.IGNORECASE))
        uses_router_decorator = bool(re.search(r"@\s*router\.", code, re.IGNORECASE))
        
        # Start building the enhanced code
        enhanced_code = []
        
        # Add necessary imports if missing
        if not has_fastapi_import:
            imports = ["from fastapi import FastAPI, HTTPException, Depends, status"]
            
            # Add router import if needed
            if uses_router_decorator and not has_router_instance:
                imports.append("from fastapi import APIRouter")
            
            # Add context-aware imports based on code content
            if "Session" in code and "sqlalchemy" not in code.lower():
                imports.append("from sqlalchemy.orm import Session")
            if "JSONResponse" in code:
                imports.append("from fastapi.responses import JSONResponse")
            if "RedirectResponse" in code:
                imports.append("from fastapi.responses import RedirectResponse")
            if "List[" in code or "Dict[" in code:
                imports.append("from typing import List, Dict, Any")
            if "BaseModel" in code:
                imports.append("from pydantic import BaseModel")
            if "Path(" in code and "from fastapi import" not in code:
                # Fix incorrect Path usage - Path should be from fastapi not as a type
                imports.append("from fastapi import Path")
            
            enhanced_code.extend(imports)
            enhanced_code.append("")
        
        # Add app/router instance if missing
        if uses_app_decorator and not has_app_instance:
            enhanced_code.append("app = FastAPI()")
            enhanced_code.append("")
        elif uses_router_decorator and not has_router_instance:
            enhanced_code.append("router = APIRouter()")
            enhanced_code.append("")
        
        # Fix common issues in the code
        fixed_code = code
        
        # Fix incorrect Path usage (Path should be a parameter annotation, not a type)
        if "user_id: Path" in fixed_code and "Path(" not in fixed_code:
            fixed_code = re.sub(r"user_id:\s*Path\b", "user_id: int", fixed_code)
        
        # Fix missing app instance for orphaned decorators
        if uses_app_decorator and not has_app_instance and "app = FastAPI()" not in enhanced_code:
            # Insert app creation before the first decorator
            lines = fixed_code.split('\n')
            for i, line in enumerate(lines):
                if '@app.' in line:
                    lines.insert(i, 'app = FastAPI()')
                    lines.insert(i+1, '')
                    break
            fixed_code = '\n'.join(lines)
        
        # Add the processed code
        enhanced_code.append(fixed_code)
        
        result = "\n".join(enhanced_code)
        
        # Clean up any double empty lines
        result = re.sub(r'\n\n\n+', '\n\n', result)
        
        return result

    def create_augmented_examples(self, example: dict) -> list:
        """
        Create multiple variations of a single example to increase dataset size.
        
        Args:
            example: Original example dictionary
            
        Returns:
            List of augmented examples
        """
        augmented = [example]  # Include original
        
        # Skip augmentation for very complex examples (to avoid noise)
        if example.get('complexity_score', 0) > 20:
            return augmented
        
        original_output = example['output']
        
        # Variation 1: Add more detailed error handling
        if 'HTTPException' in original_output and 'try:' not in original_output:
            enhanced_output = original_output.replace(
                'raise HTTPException(',
                'try:\n        # Database operation\n        pass\n    except Exception as e:\n        raise HTTPException('
            )
            augmented.append({
                **example,
                'instruction': example['instruction'] + '\nInclude comprehensive error handling',
                'output': enhanced_output,
                'difficulty': 'intermediate' if example['difficulty'] == 'beginner' else example['difficulty']
            })
        
        # Variation 2: Add response models for GET endpoints
        if '@app.get(' in original_output or '@router.get(' in original_output:
            if 'response_model=' not in original_output and 'List[' not in original_output:
                augmented.append({
                    **example,
                    'instruction': example['instruction'] + '\nInclude proper response model typing',
                    'input': example.get('input', '') + '\nAdd Pydantic response models',
                    'output': original_output.replace('def ', 'def ').replace(
                        '):', ') -> Dict[str, Any]:'),
                    'category': 'models'
                })
        
        # Variation 3: Add async version for non-database operations
        if 'async def' not in original_output and 'Session' not in original_output:
            async_output = original_output.replace('def ', 'async def ')
            augmented.append({
                **example,
                'instruction': example['instruction'] + '\nImplement as async function',
                'output': async_output,
                'tags': example.get('tags', []) + ['async']
            })
        
        # Variation 4: Add dependency injection pattern
        if 'Depends(' not in original_output and len(original_output.split('\n')) < 8:
            # Add a simple dependency
            dependency_output = original_output
            if 'def ' in dependency_output:
                func_line = [line for line in dependency_output.split('\n') if 'def ' in line][0]
                enhanced_func = func_line.replace('def ', 'def ').replace(
                    '):', ', current_user: str = Depends(get_current_user)):')
                dependency_output = dependency_output.replace(func_line, enhanced_func)
                
            augmented.append({
                **example,
                'instruction': example['instruction'] + '\nAdd user authentication dependency',
                'input': example.get('input', '') + '\nRequire authenticated user',
                'output': dependency_output,
                'category': 'auth'
            })
        
        return augmented

    def load_and_prepare_dataset(self) -> Dataset:
        """
        Load and prepare the FastAPI dataset for fine-tuning with augmentation.

        Returns:
            FastAPI data wrapped inside a Dataset object from the transformers library.
        """
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        processed_data = []
        total_augmented = 0
        
        for item in raw_data:
            # Create augmented examples
            augmented_examples = self.create_augmented_examples(item)
            total_augmented += len(augmented_examples)
            
            for aug_example in augmented_examples:
                instruction = f"Category: {aug_example['category']}\nDifficulty: {aug_example['difficulty']}\n\n{aug_example['instruction']}"

                input_text = aug_example.get('input', '')
                if input_text:
                    instruction += f"\n\nInput:\n{input_text}"

                # Enhance the output code to be complete and runnable
                enhanced_output = self.enhance_code_snippet(aug_example['output'])

                processed_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": enhanced_output,
                    "category": aug_example['category'],
                    "difficulty": aug_example['difficulty']
                })

        dataset = Dataset.from_list(processed_data)
        self.logger.info(f"Created dataset with {len(dataset)} examples (augmented from {len(raw_data)} to {total_augmented})")
        
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
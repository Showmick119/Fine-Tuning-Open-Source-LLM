"""
Script for generating text using a fine-tuned model with LoRA adapter.
"""

import argparse
from pathlib import Path
from typing import Optional, List, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel


class TextGenerator:
    """Handles text generation using a fine-tuned model."""
    
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str,
        device: Optional[str] = None,
        load_8bit: bool = False,
        max_length: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        """
        Initialize the text generator.
        
        Args:
            base_model_name: Name or path of the base model
            adapter_path: Path to the fine-tuned LoRA adapter
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            load_8bit: Whether to load the model in 8-bit mode
            max_length: Maximum length for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(
            base_model_name,
            adapter_path,
            load_8bit
        )
        
    def _load_model_and_tokenizer(
        self,
        base_model_name: str,
        adapter_path: str,
        load_8bit: bool
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer."""
        print(f"Loading base model: {base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        if load_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )
        
        # Load LoRA adapter
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        if not load_8bit and self.device == "cuda":
            model = model.half()
        
        model.eval()
        if self.device == "cpu":
            model = model.to(self.device)
            
        return model, tokenizer
        
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format the instruction and input into a prompt."""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
        
    @torch.inference_mode()
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on instruction and optional input.
        
        Args:
            instruction: The instruction for the model
            input_text: Optional input text
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        # Format prompt
        prompt = self.format_prompt(instruction, input_text)
        
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        # Set generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or (self.max_length - inputs["input_ids"].shape[1]),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs
        }
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )
        
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
        
    def generate_batch(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of instructions.
        
        Args:
            instructions: List of instructions
            input_texts: Optional list of input texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)
            
        if len(instructions) != len(input_texts):
            raise ValueError(
                f"Number of instructions ({len(instructions)}) must match "
                f"number of inputs ({len(input_texts)})"
            )
            
        return [
            self.generate(instruction, input_text, **kwargs)
            for instruction, input_text in zip(instructions, input_texts)
        ]


def main():
    """Command-line interface for text generation."""
    parser = argparse.ArgumentParser(description="Generate text using fine-tuned model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Instruction for the model"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional input text"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TextGenerator(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=args.device,
        load_8bit=args.load_8bit,
        temperature=args.temperature
    )
    
    # Generate and print response
    response = generator.generate(
        instruction=args.instruction,
        input_text=args.input,
        max_new_tokens=args.max_new_tokens
    )
    
    print("\nGenerated Response:")
    print("-" * 40)
    print(response)
    print("-" * 40)


if __name__ == "__main__":
    main() 
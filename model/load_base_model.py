"""
Module for loading and configuring the base model with LoRA adapters.
"""

import json
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)


class ModelLoader:
    """Handles loading and configuring the base model with LoRA."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the model loader.
        
        Args:
            config_path: Path to the LoRA configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def load_base_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load the base model and tokenizer with specified configuration.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading base model: {self.config['base_model_name']}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model_name'],
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare model loading kwargs
        model_kwargs = {
            "device_map": self.config.get("device_map", "auto"),
            "trust_remote_code": True,
        }
        
        # Add quantization config if specified
        if self.config.get("load_in_8bit", False):
            model_kwargs.update({
                "load_in_8bit": True,
                "torch_dtype": torch.float16,
            })
        elif self.config.get("load_in_4bit", False):
            model_kwargs.update({
                "load_in_4bit": True,
                "torch_dtype": torch.float16,
            })
        elif torch_dtype := self.config.get("torch_dtype"):
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model_name'],
            **model_kwargs
        )
        
        # Prepare for training if using quantization
        if self.config.get("load_in_8bit") or self.config.get("load_in_4bit"):
            model = prepare_model_for_kbit_training(model)
            
        return model, tokenizer
        
    def add_lora_adapter(
        self,
        model: PreTrainedModel,
        adapter_path: Optional[str] = None
    ) -> PreTrainedModel:
        """
        Add LoRA adapter to the model.
        
        Args:
            model: Base model to add adapter to
            adapter_path: Optional path to pre-trained adapter weights
            
        Returns:
            Model with LoRA adapter
        """
        lora_config = LoraConfig(
            **self.config['lora_config'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        if adapter_path:
            print(f"Loading pre-trained adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            print("Initializing new LoRA adapter")
            model = get_peft_model(model, lora_config)
            
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model


if __name__ == "__main__":
    # Example usage
    loader = ModelLoader("configs/lora_config.json")
    
    # Load base model and tokenizer
    base_model, tokenizer = loader.load_base_model()
    
    # Add LoRA adapter
    model = loader.add_lora_adapter(base_model)
    
    print("Model loading complete") 
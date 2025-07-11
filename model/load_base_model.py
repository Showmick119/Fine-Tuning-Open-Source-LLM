"""
Module for loading and configuring the base model with LoRA adapters.
"""

import json
from pathlib import Path
from typing import Optional, Union, Tuple
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

DEFAULT_CONFIG = {
    "base_model_name": "codellama/CodeLlama-7b-Instruct-hf",
    "lora_config": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "load_in_4bit": True,
    "bnb_config": {
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16"
    },
    "device_map": "auto",
    "torch_dtype": "bfloat16"
}

class ModelLoader:
    """
    Handles loading and configuring the base model with LoRA.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the model loader.
        
        Args:
            config_path: Optional path to the LoRA configuration file. If not provided, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is not None:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                self.logger.warning(f"Config file not found at {self.config_path}, using default config")
                self.config = DEFAULT_CONFIG
            else:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
        else:
            self.config = DEFAULT_CONFIG
            
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration parameters.
        """
        required_fields = {"base_model_name", "lora_config"}
        missing = required_fields - set(self.config.keys())
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
            
        lora_required = {"r", "lora_alpha", "target_modules"}
        missing_lora = lora_required - set(self.config['lora_config'].keys())
        if missing_lora:
            raise ValueError(f"Missing required LoRA config fields: {missing_lora}")

    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the base model and tokenizer with optimized settings for fine-tuning.

        Returns:
            Returns a tuple of the Code Llama model and tokenizer with LoRA and quantization configurations.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config["load_in_4bit"],
            bnb_4bit_quant_type=self.config["bnb_config"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, self.config["bnb_config"]["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=self.config["bnb_config"]["bnb_4bit_use_double_quant"]
        )

        self.logger.info(f"Loading model: {self.config['base_model_name']}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model_name'],
            quantization_config=bnb_config,
            device_map=self.config["device_map"],
            trust_remote_code=True,
            torch_dtype=getattr(torch, self.config["torch_dtype"])
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(**self.config["lora_config"])
        model = get_peft_model(model, lora_config)

        self.logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model_name'],
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer


if __name__ == "__main__":
    loader = ModelLoader("configs/lora_config.json")

    base_model, tokenizer = loader.load_model_and_tokenizer()
    
    print("Model loading complete")

    if torch.cuda.is_available():
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB") 
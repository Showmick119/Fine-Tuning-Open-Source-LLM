# LLM Fine-tuning with LoRA

This repository provides a clean, modular implementation for fine-tuning large language models (LLMs) using Low-Rank Adaptation (LoRA) via the PEFT library. The project is designed to work with HuggingFace-compatible models and is optimized for Google Colab environments.

## Features

- 🚀 Fine-tune LLMs using LoRA (Low-Rank Adaptation)
- 📦 Support for various HuggingFace models (default: Mistral-7B)
- 💾 8-bit quantization support for efficient training
- 📊 Comprehensive logging and checkpointing
- 🔄 Easy-to-use training and inference pipelines
- 📓 Jupyter notebooks for interactive usage
- 🎯 Clean, typed, and modular Python code

## Project Structure

```
llm-finetuning-lora/
├── configs/
│   ├── lora_config.json        # LoRA adapter configuration
│   └── training_args.json      # Training hyperparameters
├── data/
│   ├── raw/                    # Place your raw data files here
│   └── prepare_dataset.py      # Dataset preparation utilities
├── model/
│   └── load_base_model.py      # Model loading and LoRA configuration
├── train/
│   └── run_lora_finetune.py    # Training pipeline
├── inference/
│   └── generate_text.py        # Text generation utilities
├── notebooks/
│   ├── 1_finetune_lora.ipynb  # Fine-tuning demonstration
│   └── 2_test_finetuned.ipynb # Inference demonstration
├── outputs/
│   ├── checkpoints/           # Saved model checkpoints
│   └── logs/                  # Training logs
├── requirements.txt           # Project dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-finetuning-lora.git
   cd llm-finetuning-lora
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Your Data

Place your training data in the `data/raw/` directory. The data should be in JSON or JSONL format with the following structure:

```json
{
    "instruction": "Your instruction here",
    "input": "Optional input text",
    "output": "Expected output text"
}
```

### 2. Configure the Training

1. Adjust LoRA settings in `configs/lora_config.json`:
   ```json
   {
       "base_model_name": "mistralai/Mistral-7B-v0.1",
       "lora_config": {
           "r": 8,
           "lora_alpha": 16,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "lora_dropout": 0.05
       }
   }
   ```

2. Modify training parameters in `configs/training_args.json`:
   ```json
   {
       "num_train_epochs": 3,
       "per_device_train_batch_size": 4,
       "learning_rate": 2e-4
   }
   ```

### 3. Run Fine-tuning

#### Option 1: Using Python Script

```bash
python train/run_lora_finetune.py \
    --training_config configs/training_args.json \
    --lora_config configs/lora_config.json \
    --data_path data/raw/your_data.json
```

#### Option 2: Using Jupyter Notebook

Open and run `notebooks/1_finetune_lora.ipynb` in Google Colab or your local Jupyter environment.

### 4. Generate Text with Fine-tuned Model

#### Option 1: Using Python Script

```bash
python inference/generate_text.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --adapter_path outputs/checkpoints \
    --instruction "Your instruction here"
```

#### Option 2: Using Jupyter Notebook

Open and run `notebooks/2_test_finetuned.ipynb` to interactively generate text with your fine-tuned model.

## Advanced Usage

### Custom Dataset Preparation

Extend the `DatasetPreparator` class in `data/prepare_dataset.py` to handle your specific data format:

```python
from data.prepare_dataset import DatasetPreparator

preparator = DatasetPreparator(
    tokenizer="mistralai/Mistral-7B-v0.1",
    max_length=512,
    data_path="path/to/your/data.json"
)
dataset = preparator.prepare_dataset()
```

### Customizing the Training Loop

Modify `train/run_lora_finetune.py` to add custom training features:

```python
from train.run_lora_finetune import run_training

run_training(
    training_config_path="configs/training_args.json",
    lora_config_path="configs/lora_config.json",
    data_path="path/to/your/data.json",
    output_dir="custom/output/path"
)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.36+
- PEFT 0.7+
- Datasets 2.15+
- Accelerate 0.25+
- bitsandbytes 0.41+

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# 🚀 CodeLlama Fine-tuning with QLoRA

This repository provides a clean, modular implementation for fine-tuning **CodeLlama-7b-Instruct** using **QLoRA (4-bit quantization)** via the PEFT library on the **CodeAlpaca-20k** dataset. The project is optimized for Google Colab environments and includes comprehensive evaluation on HumanEval.

## Features

- 🤖 Fine-tune **CodeLlama-7b-Instruct** using QLoRA (4-bit quantization)
- 📊 Train on **CodeAlpaca-20k** dataset (20,000 high-quality code examples)
- ⚡ **QLoRA** support for efficient training on single GPU
- 📈 **HumanEval** evaluation for benchmarking code generation
- 🔄 Easy-to-use training and inference pipelines
- 📓 Colab-ready Jupyter notebooks with rich documentation
- 🎯 Clean, typed, and modular Python code

## Project Structure

```
llm-finetuning-lora/
├── configs/
│   ├── lora_config.json        # QLoRA configuration for CodeLlama
│   └── training_args.json      # Training hyperparameters
├── data/
│   ├── code_alpaca_20k.json    # CodeAlpaca-20k training dataset
│   └── prepare_dataset.py      # Dataset preparation utilities
├── model/
│   └── load_base_model.py      # Model loading with QLoRA support
├── train/
│   └── run_lora_finetune.py    # Training pipeline
├── inference/
│   └── generate_text.py        # Code generation utilities
├── evaluate/
│   └── evaluate_model.py       # HumanEval evaluation module
├── notebooks/
│   ├── 1_train_model.ipynb     # 🚀 Fine-tuning demonstration
│   ├── 2_test_model.ipynb      # 🧪 Interactive testing
│   └── 3_evaluate_model.ipynb  # 📊 HumanEval evaluation
├── outputs/
│   ├── checkpoints/            # Saved LoRA adapters
│   ├── logs/                   # Training logs
│   └── evaluation/             # Evaluation results
├── requirements.txt            # Project dependencies
└── README.md                  # This file
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

### 1. Dataset: CodeAlpaca-20k

The repository comes with the **CodeAlpaca-20k** dataset (`data/code_alpaca_20k.json`) containing 20,000 high-quality instruction-following examples for code generation. The dataset format:

```json
{
    "instruction": "Write a Python function to...",
    "input": "Optional input context",
    "output": "def example_function():\n    # Implementation"
}
```

### 2. QLoRA Configuration

The model is pre-configured for **CodeLlama-7b-Instruct** with QLoRA settings in `configs/lora_config.json`:

```json
{
    "base_model_name": "codellama/CodeLlama-7b-Instruct-hf",
    "load_in_4bit": true,
    "bnb_config": {
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16"
    },
    "lora_config": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    }
}
```

### 3. Quick Start with Google Colab

#### 🚀 Fine-tuning (Notebook)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/llm-finetuning-lora/blob/main/notebooks/1_train_model.ipynb)

1. Open `notebooks/1_train_model.ipynb` in Google Colab
2. Run all cells to fine-tune CodeLlama on CodeAlpaca-20k
3. Training takes ~1-2 hours on Colab's T4 GPU

#### 🧪 Testing (Notebook)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/llm-finetuning-lora/blob/main/notebooks/2_test_model.ipynb)

1. Open `notebooks/2_test_model.ipynb` 
2. Test your fine-tuned model on various coding tasks
3. Interactive prompt interface for custom testing

#### 📊 Evaluation (Notebook)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/llm-finetuning-lora/blob/main/notebooks/3_evaluate_model.ipynb)

1. Open `notebooks/3_evaluate_model.ipynb`
2. Benchmark your model on HumanEval (164 coding problems)
3. Get Pass@1 scores and detailed analysis

### 4. Command Line Usage

#### Fine-tuning
```bash
python train/run_lora_finetune.py \
    --training_config configs/training_args.json \
    --lora_config configs/lora_config.json
```

#### Code Generation
```bash
python inference/generate_text.py \
    --base_model codellama/CodeLlama-7b-Instruct-hf \
    --adapter_path outputs/checkpoints \
    --instruction "Write a Python function to implement binary search"
```

#### HumanEval Evaluation
```bash
python evaluate/evaluate_model.py \
    --base_model codellama/CodeLlama-7b-Instruct-hf \
    --adapter_path outputs/checkpoints
```

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
- lighteval 0.4+ (for HumanEval evaluation)

### Hardware Requirements
- **Training**: 12GB+ GPU (Colab T4 works well)
- **Inference**: 6GB+ GPU or CPU
- **Evaluation**: 8GB+ GPU for HumanEval benchmark

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

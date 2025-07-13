# 🚀 FastAPI Code Generation with Fine-Tuned CodeLlama-7B

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Showmick119/Fine-Tuning-Open-Source-LLM/blob/main/notebooks/finetune_code_llama.ipynb)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/Showmick119/codellama-7b-fastapi-finetuned-20250713)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready fine-tuning pipeline for **CodeLlama-7b-Instruct** specialized in **FastAPI code generation**. This project demonstrates advanced techniques including **QLoRA quantization**, **intelligent dataset curation**, and **adaptive evaluation systems** to create a model that generates high-quality, production-ready FastAPI applications.

## 🎯 **Results & Performance**

| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| **FastAPI Code Quality** | 75.0/100 | **85.0/100** | **+10.0 points** |
| **Code Completeness** | 59.9/100 | **65.8/100** | **+5.8 points** |
| **Training Time** | - | **~11 minutes** | T4 GPU |
| **Dataset Size** | - | **570 examples** | Curated & Augmented |

### **Key Improvements**
- ✅ **Proper FastAPI imports and structure** - Contextual import generation
- ✅ **Database integration patterns** - SQLAlchemy, MongoDB support  
- ✅ **Error handling with HTTP status codes** - Professional error responses
- ✅ **Authentication and validation logic** - JWT, OAuth2, Pydantic models
- ✅ **Production-ready code patterns** - Router organization, dependency injection

## 🏗️ **Project Architecture**

```
Fine-Tuning-Open-Source-LLM/
├── 📊 configs/
│   ├── lora_config.json          # QLoRA adapter configuration
│   ├── training_args.json        # Optimized training hyperparameters
│   └── hub_config.json          # HuggingFace Hub deployment settings
├── 📁 data/
│   ├── fastapi_miner.py          # GitHub FastAPI pattern mining
│   ├── prepare_dataset.py        # Intelligent dataset preprocessing
│   └── data/
│       └── fastapi_mined_dataset.json  # 331 real-world FastAPI patterns
├── 🤖 model/
│   └── load_base_model.py        # Model loading with quantization
├── 🔬 train/
│   └── run_lora_finetune.py      # Complete training pipeline
├── 📈 evaluate/
│   ├── fastapi_evaluator.py      # FastAPI-specific code evaluation
│   └── llm_judge.py             # GPT-based code quality assessment
├── 📓 notebooks/
│   ├── finetune_code_llama.ipynb     # 🚀 Complete fine-tuning workflow
│   └── colab_test_base_model.ipynb  # 🧪 Base model evaluation
└── 📋 requirements.txt           # Production dependencies
```

## 🚀 **Quick Start**

### **Option 1: Use Pre-trained Model (Recommended)**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model
model_name = "Showmick119/codellama-7b-fastapi-finetuned-20250713"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True  # For efficient inference
)

# Generate FastAPI code
prompt = "[INST] Create a FastAPI POST endpoint for user registration with email validation [/INST]"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response[len(prompt):].strip())
```

### **Option 2: Fine-tune Your Own Model**

1. **Open in Google Colab** (Recommended):
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Showmick119/Fine-Tuning-Open-Source-LLM/blob/main/notebooks/finetune_code_llama.ipynb)

2. **Local Setup**:
   ```bash
   git clone https://github.com/Showmick119/Fine-Tuning-Open-Source-LLM.git
   cd Fine-Tuning-Open-Source-LLM
   pip install -r requirements.txt
   ```

3. **Configure HuggingFace Access**:
   ```python
   import os
   os.environ['HF_TOKEN'] = 'your_hf_token_here'
   os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'  # For evaluation
   ```

4. **Run Training**: Execute the Jupyter notebook cells sequentially.

## 🔬 **Technical Innovation**

### **1. Intelligent Dataset Curation**
- **GitHub Mining**: Automated extraction of real-world FastAPI patterns
- **Smart Augmentation**: Context-aware code variations (1.72x dataset expansion)
- **Import Optimization**: Only adds necessary imports based on actual usage

### **2. Advanced Training Configuration**
- **QLoRA Optimization**: 4-bit quantization with LoRA adapters
- **Efficient Training**: ~11 minutes on T4 GPU with 570 examples
- **Progress Monitoring**: Evaluation every 10 steps with detailed metrics

### **3. Comprehensive Evaluation System**
- **FastAPI Evaluator**: Syntax, imports, endpoints, error handling validation
- **GPT Judge**: Code quality, best practices, completeness assessment
- **Adaptive Scoring**: Complexity-aware evaluation criteria

## 📊 **Dataset & Training Details**

### **Dataset Composition**
| Category | Examples | Description |
|----------|----------|-------------|
| **Authentication** | 215 | JWT, OAuth2, session management |
| **Database** | 94 | SQLAlchemy, MongoDB integration |
| **Endpoints** | 74 | REST API patterns, CRUD operations |
| **Models** | 29 | Pydantic schemas, validation |
| **Validation** | 4 | Input validation, error handling |

### **Training Configuration**
- **Base Model**: `codellama/CodeLlama-7b-Instruct-hf`
- **Quantization**: 4-bit with bitsandbytes
- **LoRA Settings**: r=64, alpha=16, dropout=0.1
- **Training**: 3 epochs, 1e-4 learning rate, cosine scheduler
- **Hardware**: Single T4 GPU (15GB VRAM)

## 🎯 **Use Cases**

### **Enterprise Development**
- **API Scaffolding**: Generate complete FastAPI applications
- **Code Review**: Ensure best practices and error handling
- **Documentation**: Auto-generate OpenAPI specifications

### **Education & Learning**
- **FastAPI Tutorials**: Generate teaching examples
- **Code Completion**: IDE integration for FastAPI development
- **Best Practices**: Learn proper FastAPI patterns

### **Rapid Prototyping**
- **MVP Development**: Quick API prototypes
- **Microservices**: Generate service templates
- **Integration Testing**: Create test endpoints

## 🤝 **Model Comparison**

| Feature | Base CodeLlama-7B | Fine-tuned Model |
|---------|-------------------|------------------|
| FastAPI Imports | ❌ Often missing | ✅ Always correct |
| Error Handling | ❌ Basic/incomplete | ✅ Comprehensive |
| Status Codes | ❌ Rarely used | ✅ Proper HTTP codes |
| Database Patterns | ❌ Generic | ✅ FastAPI-specific |
| Production Ready | ❌ Requires editing | ✅ Near production-ready |

## 📈 **Evaluation Results**

### **Test Case Performance**
1. **Authentication Endpoint**: +70.0 FastAPI, +78.0 GPT improvement
2. **CRUD Operations**: +20.0 FastAPI, +21.0 GPT improvement  
3. **Dependency Injection**: -10.0 FastAPI, +43.5 GPT improvement
4. **User Management**: 100.0/100 FastAPI score maintained

### **Key Improvements Demonstrated**
- **No more repetitive imports** (eliminated base model hallucinations)
- **Context-aware imports** (only includes what's actually used)
- **Professional error handling** (proper HTTP status codes)
- **Database integration** (SQLAlchemy, MongoDB patterns)

## 🛠️ **Advanced Features**

### **Intelligent Import Generation**
```python
# Before: Always hardcoded imports
from fastapi import FastAPI, HTTPException, Depends, status, APIRouter, Path

# After: Context-aware imports
from fastapi import FastAPI, HTTPException  # Only what's used
```

### **Production-Ready Error Handling**
```python
@app.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate):
    try:
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already exists"
            )
        return user_service.create(user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )
```

## 📝 **Requirements**

### **Core Dependencies**
```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.15.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
openai>=1.0.0          # For GPT evaluation
fastapi>=0.104.0       # For testing generated code
```

### **Hardware Requirements**
- **Minimum**: T4 GPU (15GB VRAM) for training
- **Recommended**: A100 or V100 for faster training
- **Inference**: CPU compatible with 8-bit quantization

## 🏆 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest

# Run tests
pytest tests/

# Format code
black .
```

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Meta AI** for the base CodeLlama-7b-Instruct model
- **HuggingFace** for the transformers library and model hosting
- **Microsoft** for QLoRA implementation and training optimizations
- **FastAPI community** for the excellent framework and patterns

## 📞 **Contact & Support**

- **GitHub Issues**: [Report bugs or request features](https://github.com/Showmick119/Fine-Tuning-Open-Source-LLM/issues)
- **Model Downloads**: [HuggingFace Hub](https://huggingface.co/Showmick119/codellama-7b-fastapi-finetuned-20250713)
- **Discussions**: [GitHub Discussions](https://github.com/Showmick119/Fine-Tuning-Open-Source-LLM/discussions)

---

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ for the FastAPI and AI community*

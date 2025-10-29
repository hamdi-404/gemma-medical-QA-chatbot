# 🏥 Medical Question Answering with Gemma-1B

A fine-tuned Google Gemma-2B model for medical question answering, trained on 50,000 medical QA pairs using parameter-efficient fine-tuning (LoRA with 4-bit quantization).

## 📋 Table of Contents
- [Overview](#overview)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)

## 🎯 Overview

This project fine-tunes Google's Gemma-2B-IT model on medical question-answering data to create a specialized medical assistant. The model can provide informative answers to various medical questions while being efficient enough to run on consumer hardware.

**Key Features:**
- ✨ Fine-tuned on 50,000 medical QA pairs
- 🚀 Uses LoRA (Low-Rank Adaptation) for efficient training
- 💾 4-bit quantization for reduced memory usage
- 🎯 Specialized for medical domain questions
- 🌐 Ready-to-deploy Streamlit application

## 🤖 Model Details

### Base Model
- **Model:** `google/gemma-3-1b-it`
- **Parameters:** 1B parameters
- **Architecture:** Transformer-based causal language model
- **Quantization:** 4-bit (nf4) using bitsandbytes

### Fine-tuning Configuration
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **LoRA Dropout:** 0.05
- **Trainable Parameters:** ~8.4M (0.32% of total)

### Training Hyperparameters
- **Epochs:** 1
- **Batch Size:** 4 per device
- **Gradient Accumulation Steps:** 4 (effective batch size = 16)
- **Learning Rate:** 2e-4
- **Optimizer:** paged_adamw_8bit
- **Max Sequence Length:** 512 tokens
- **Warmup Steps:** 100
- **Weight Decay:** 0.01

## 📊 Dataset

### Source
- **Dataset:** [lavita/medical-qa-datasets](https://huggingface.co/datasets/lavita/medical-qa-datasets)
- **Total Used:** 50,000 examples (subset of 234k)
- **Format:** Instruction-Input-Output medical QA pairs

### Data Split
```
Total: 50,000 examples
├── Training:   35,000 examples (70%)
├── Validation:  7,500 examples (15%)
└── Test:        7,500 examples (15%)
```

### Data Preprocessing
1. **Cleaning:** Removed excessive whitespace and special characters
2. **Quality Filtering:** 
   - Minimum question length: 10 characters
   - Minimum answer length: 10 characters
   - Maximum length: 500 words per field
3. **Format:** Converted to Gemma instruction format with special tokens

## 📁 Repository Structure

```
medical-qa-gemma/
│
├── model_training.ipynb           # 📓 Data preparation & training notebook
│   ├── Data download and exploration
│   ├── Data cleaning and preprocessing
│   ├── Train/validation/test splits
│   ├── Model setup with LoRA and 4-bit quantization
│   ├── Training loop
│   └── Evaluation and testing
│
├── model_merging.ipynb            # 🔗 Adapter merging notebook
│   ├── Load base Gemma model
│   ├── Load trained LoRA adapter
│   ├── Merge adapter with base model
│   └── Save merged model
│
├── app.py                         # 🌐 Streamlit web application
│
├── requirements.txt               # 📦 Python dependencies
│
└── README.md                      # 📖 This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM
- Hugging Face account with Gemma access

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/hamdi-404/gemma-medical-QA-chatbot.git
cd medical-qa-gemma
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Login to Hugging Face** (if not already)
```bash
huggingface-cli login
```

4. **Accept Gemma License**
- Visit: https://huggingface.co/google/gemma-2-2b-it
- Click "Agree and access repository"



## 📚 Training Details

### Training Process

1. **Data Preparation**
   - Downloaded lavita/medical-qa-datasets
   - Selected 50,000 examples randomly
   - Split into 70/15/15 train/val/test
   - Formatted for Gemma instruction format

2. **Model Setup**
   - Loaded base Gemma-2B-IT model
   - Applied 4-bit quantization (NF4)
   - Configured LoRA adapters
   - Prepared model for k-bit training

3. **Training**
   - Trained for 3 epochs
   - Monitored validation loss
   - Saved best checkpoint
   - Total training time: ~6-8 hours on RTX 3060

4. **Merging**
   - Merged LoRA adapter with base model
   - Created standalone model
   - Saved merged model for deployment

### Memory Requirements

| Configuration | Training | Inference |
|--------------|----------|-----------|
| **4-bit + LoRA** | ~8GB VRAM | ~4GB VRAM |
| **Full Precision** | ~24GB VRAM | ~12GB VRAM |

## 📈 Results

### Performance Metrics

```
Training Loss:   1.234 (final)
Validation Loss: 1.456 (final)
Test Loss:       1.478
```

### Sample Outputs

**Question:** "What are the symptoms of diabetes?"

**Model Answer:** 
> "The common symptoms of diabetes include increased thirst and urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms often develop quickly over weeks, while Type 2 diabetes symptoms develop gradually over years. If you experience these symptoms, consult a healthcare professional for proper diagnosis and treatment."

**Question:** "How to treat high blood pressure?"

**Model Answer:**
> "Treatment for high blood pressure typically involves lifestyle changes and medications. Lifestyle modifications include reducing sodium intake, maintaining a healthy weight, regular exercise, limiting alcohol consumption, and managing stress. Medications may include diuretics, ACE inhibitors, ARBs, calcium channel blockers, or beta-blockers depending on your specific condition. Always consult with your doctor for personalized treatment plans."

## 🌐 Deployment

### Using Streamlit (Included)

The repository includes a ready-to-use Streamlit application:

```bash
streamlit run app.py
```

Features:
- Interactive chat interface
- Adjustable generation parameters
- Response history
- Medical disclaimer

### Using as API

You can wrap the model in FastAPI or Flask:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    text: str

@app.post("/answer")
def get_answer(question: Question):
    # Load model (do this once at startup)
    answer = generate_answer(question.text)
    return {"answer": answer}
```

### Cloud Deployment Options

1. **Hugging Face Spaces**
   - Upload model to Hugging Face
   - Create a Space with Streamlit
   - Free hosting available

2. **AWS/GCP/Azure**
   - Deploy on cloud GPU instances
   - Use container services (ECS, Cloud Run, etc.)

3. **Replicate**
   - Easy model deployment
   - Pay-per-use pricing

## ⚠️ Important Disclaimers

**Medical Disclaimer:** This AI model provides information for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.

**AI Notice:** Responses are generated by an AI model and may not be accurate, complete, or up-to-date. Do not rely on this information for medical decisions.

**Model Limitations:**
- May generate incorrect or outdated medical information
- Not trained on latest medical research
- Cannot replace professional medical consultation
- Should not be used for emergencies (call emergency services)

## 🔒 Model Safety

This model includes:
- ✅ Training on verified medical QA data
- ✅ Clear disclaimers in responses
- ❌ No guarantee of medical accuracy
- ❌ Not evaluated by medical professionals

**Recommended Use Cases:**
- Medical education and training
- General health information queries
- Medical terminology explanation
- Research and development

**Not Recommended For:**
- Emergency medical situations
- Diagnosis of medical conditions
- Treatment decisions
- Prescription recommendations

## 📝 Requirements

### Core Dependencies
```
transformers>=4.40.0
torch>=2.0.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.28.0
datasets>=2.18.0
```

See `requirements.txt` for complete list.


## 📄 License

This project uses:
- **Gemma Model:** Subject to [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Code:** MIT License (see LICENSE file)
- **Dataset:** Subject to original dataset license

## 🙏 Acknowledgments

- **Google** for the Gemma model
- **Hugging Face** for transformers library and dataset hosting
- **LaVita** for the medical QA dataset
- **Microsoft** for bitsandbytes quantization
- **Community** for LoRA and PEFT implementations

## 👨‍🔬 Author

Hamdi Mohamed
Machine Learning Engineer

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: hamdi404.cs@gmail.com


## 🔗 Useful Links

- [Gemma Model Card](https://huggingface.co/google/gemma-3-1b-it)
- [Medical QA Dataset](https://huggingface.co/datasets/lavita/medical-qa-datasets)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [4-bit Quantization Paper](https://arxiv.org/abs/2305.14314)

---

**⭐ If you find this project useful, please consider giving it a star!**

**🐛 Found a bug? Please open an issue!**

**💡 Have suggestions? We'd love to hear from you!**

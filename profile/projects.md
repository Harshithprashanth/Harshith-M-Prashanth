# 🚀 Projects — Harshith Prashanth

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)](https://langchain.com)

---

## 📋 Table of Contents

- [LLM & LLMOps Projects](#llm--llmops-projects)
- [Signal Processing & Biomedical ML](#signal-processing--biomedical-ml)
- [IoT & Embedded Systems](#iot--embedded-systems)
- [Computer Vision & Multi-Modal AI](#computer-vision--multi-modal-ai)
- [Academic Course Projects](#academic-course-projects)
- [Research Reproductions & Experiments](#research-reproductions--experiments)
- [How to Run](#how-to-run)

---

## 🤖 LLM & LLMOps Projects

### Project 1: Domain-Adaptive RAG System for Scientific Literature

**Repository**: [PLACEHOLDER: GitHub link]  
**Domain**: LLMOps · RAG · NLP  
**Status**: ✅ Complete

#### Problem Statement

General-purpose LLM APIs perform poorly on technical scientific literature because they lack the domain-specific context required to answer precise engineering and research questions accurately. A scientist querying a chatbot about specific papers, methodologies, or technical details needs faithful, grounded answers — not hallucinated plausibilities.

#### System Architecture

```
PDF / arXiv Papers
       │
       ▼
Document Loader (LangChain)
       │
       ▼
Semantic Chunker (sentence-transformers)
       │
       ▼
Embedding Model (all-mpnet-base-v2)
       │
       ▼
ChromaDB Vector Store
       │
   ┌───┴────────────────────────────┐
   ▼                                ▼
Standard Retrieval              HyDE Retrieval
   │                                │
   └──────────────┬─────────────────┘
                  ▼
         Cross-Encoder Re-ranker
                  │
                  ▼
         Context Assembly
                  │
                  ▼
         LLM Generation (GPT-4 / fine-tuned)
                  │
                  ▼
         RAGAS Evaluation
```

#### Implementation Details

**Data Pipeline**:
- Ingests PDFs from local filesystem and arXiv papers via arXiv API
- Applies recursive character splitting with 512-token chunks and 50-token overlap
- Generates semantic embeddings using sentence-transformers (all-mpnet-base-v2)
- Stores embeddings with metadata (title, abstract, authors, year, section) in ChromaDB

**Advanced Retrieval**:
- HyDE: GPT-3.5-turbo generates a hypothetical answer; the answer embedding retrieves more semantically relevant documents than the question embedding alone
- Multi-query decomposition: complex queries decomposed into 3–5 sub-queries; results merged via reciprocal rank fusion
- Cross-encoder re-ranking: top-20 bi-encoder results re-ranked using cross-encoder/ms-marco-MiniLM-L-6-v2

**Evaluation (RAGAS)**:
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
# faithfulness: 0.91, answer_relevancy: 0.88, context_precision: 0.85
```

**Tech Stack**: Python · LangChain · ChromaDB · Sentence-Transformers · OpenAI API · HuggingFace · RAGAS · FastAPI · Docker · LangSmith

**Quantitative Results**:
| Metric | Score |
|--------|-------|
| Faithfulness (RAGAS) | 0.91 |
| Answer Relevancy | 0.88 |
| Context Precision | 0.85 |
| Mean Query Latency | 1.8s |

**Future Work**: Domain-specific embedding fine-tuning; graphRAG with citation graph traversal; multi-document synthesis

---

### Project 2: LLM Fine-tuning Pipeline for Technical Domain Adaptation

**Repository**: [PLACEHOLDER: GitHub link]  
**Domain**: LLMOps · LLM Fine-tuning  
**Status**: ✅ Complete

#### Problem Statement

Out-of-the-box LLMs lack depth in specialized engineering domains. Full fine-tuning is prohibitively expensive (requires 80GB+ VRAM for 7B models). QLoRA enables high-quality domain adaptation on consumer and academic GPU budgets.

#### Implementation Details

**Quantization Configuration**:
```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

**LoRA Configuration**:
```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Training Results**:
| Metric | Value |
|--------|-------|
| VRAM Usage (vs. full FT) | 65% reduction |
| Performance vs. Full FT | 97% retention |
| Inference Speedup (GGUF Q4) | 3.2× |
| Training Time (7B, 1 epoch) | ~4 hours on A100 40GB |

**Tech Stack**: PyTorch · HuggingFace PEFT · TRL · BitsAndBytes · W&B · MLflow · Ollama · LangChain

---

### Project 3: Multi-Agent LLM Orchestration Framework

**Repository**: [PLACEHOLDER: GitHub link]  
**Domain**: LLMOps · Agentic AI  
**Status**: ✅ Complete

#### Agent Architecture

```
User Research Request
         │
         ▼
    Orchestrator Agent (LangGraph)
    ┌─────┬─────┬──────┬──────────┐
    ▼     ▼     ▼      ▼          ▼
 Lit.   Data  Summ.  Citation  Report
Review  Anal.  Agent  Manager   Gen.
Agent  Agent          Agent    Agent
  │      │               │       │
arXiv  Python         Zotero   LaTeX
 API    REPL            API    Template
  │      │
ChromaDB Pandas
```

**Key Results**:
- 78% reduction in manual research workflow time (measured across 15 research tasks)
- Full LangSmith observability: all agent steps, tool calls, and LLM interactions traced

**Tech Stack**: LangChain · LangGraph · LangSmith · OpenAI API · Claude API · FastAPI · Redis · Docker

---

### Project 4: LLM-Powered IoT Anomaly Detection & Natural Language Interface

**Repository**: [PLACEHOLDER: GitHub link]  
**Domain**: LLMOps · IoT · Edge AI  
**Status**: ✅ Complete

#### System Architecture

Industrial IoT sensor network → MQTT broker → InfluxDB → RAG pipeline → Mistral-7B (Ollama) → Operator NL interface

**Key Technical Details**:
- Mistral-7B quantized to GGUF Q4_K_M format via llama.cpp
- RAG context injection: last 100 sensor readings + historical anomaly records + equipment documentation
- Function calling for actuator control: structured JSON outputs parsed to MQTT control commands
- Real-time streaming response via FastAPI WebSocket

**Results**:
| Metric | Value |
|--------|-------|
| Mean Response Latency (edge) | 1.7s |
| Query Accuracy (human eval) | 93% |
| False Positive Rate (anomaly) | 4.2% |

**Tech Stack**: Ollama · Mistral-7B · LangChain · MQTT · InfluxDB · Grafana · Raspberry Pi 4B · FastAPI · ChromaDB

---

### Project 5: LLM Evaluation & Benchmarking Suite

**Repository**: [PLACEHOLDER: GitHub link]  
**Domain**: LLMOps · LLM Evaluation  
**Status**: ✅ Complete

Evaluated GPT-4, GPT-3.5-turbo, Claude Haiku/Sonnet/Opus, Gemini Pro/Flash, Llama-2-7B/13B, Llama-3-8B, Mistral-7B, and Phi-3-mini across 8 task categories: factual QA, multi-step reasoning, code generation, mathematical reasoning, instruction following, safety and refusal, creative writing, and domain-specific engineering.

**Evaluation Paradigms**:
1. Reference-based: ROUGE-L, BLEU-4, BERTScore F1
2. LLM-as-Judge: GPT-4 scoring with structured rubrics (calibrated against human experts)
3. Constitutional evaluation: safety and alignment assessment

**Tech Stack**: OpenAI API · Anthropic API · HuggingFace · DeepEval · RAGAS · TruLens · W&B · Streamlit · Docker

---

## 📡 Signal Processing & Biomedical ML

### Project 6: Intelligent Fault Detection in Industrial IoT

**Repository**: [PLACEHOLDER: GitHub link]

Vibration signal analysis system combining classical DSP features (wavelet packet decomposition, FFT peak features) with ML classifiers and a CNN-LSTM deep learning model for multi-class fault classification (normal, inner race fault, outer race fault, ball fault) in rotating machinery.

**Pipeline**: Vibration signal acquisition (ADXL345 accelerometer) → Raspberry Pi 4B → MQTT → InfluxDB → Feature extraction → ML inference → LLM-generated maintenance recommendations (Mistral-7B via LangChain)

**Results**:
| Model | Accuracy |
|-------|---------|
| SVM (wavelet features) | 89.3% |
| Random Forest | 91.7% |
| CNN-LSTM | **94.7%** |

False positive reduction vs. threshold-based baseline: 23%

**Tech Stack**: Python · TensorFlow · Raspberry Pi · MQTT · InfluxDB · Scikit-learn · LangChain · Mistral-7B

---

### Project 7: ECG Signal Denoising Using Deep Learning

**Repository**: [PLACEHOLDER: GitHub link]

Convolutional autoencoder with multi-head attention mechanisms trained on the MIT-BIH Arrhythmia Database (48 recordings, 30-minute duration, 360 Hz sampling). Artificially corrupts signals with additive white Gaussian noise at 5–20 dB SNR, trains autoencoder to reconstruct clean signal.

**Architecture**: 1D CNN encoder (7 layers, 64→128→256 channels) → bottleneck → attention gate → decoder (mirror of encoder)

**Results**:
| Method | SNR Improvement (dB) |
|--------|---------------------|
| Wiener Filter | 11.2 |
| Wavelet Denoising | 14.2 |
| **Proposed Autoencoder** | **18.3** |

**Tech Stack**: PyTorch · MATLAB · NumPy · SciPy · Wfdb (PhysioNet toolkit)

---

### Project 8: Automated Plant Disease Detection with LLM Advisory

**Repository**: [PLACEHOLDER: GitHub link]

Fine-tuned ResNet-50 (ImageNet pre-trained) on PlantVillage dataset (54,309 images, 38 classes covering 14 crop species). Integrated with LangChain advisory chatbot providing treatment recommendations, pesticide information, and agronomic guidance in natural language.

**Training**: Learning rate 1e-4, cosine schedule, 30 epochs, data augmentation (random crop, horizontal flip, colour jitter, Cutout)

**Results**: Top-1 accuracy 96.2% on held-out test set; competitive with published SOTA on PlantVillage

**Tech Stack**: TensorFlow/Keras · ResNet-50 · Flask · LangChain · OpenAI API · PIL

---

### Project 9: Real-Time Speech Emotion Recognition

**Repository**: [PLACEHOLDER: GitHub link]

MFCC (40 coefficients) + Δ + ΔΔ feature extraction (25ms window, 10ms hop) fed into a Bidirectional LSTM (256 units × 2 layers) trained on RAVDESS (24 actors, 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised). Real-time inference via PyAudio at 44.1 kHz.

**Results**: 87.4% 8-class accuracy on RAVDESS test set; 42ms inference latency per 500ms audio segment

**Tech Stack**: TensorFlow · Librosa · Bidirectional LSTM · Streamlit · PyAudio

---

## 🌐 IoT & Embedded Systems

### Project 10: Smart Home Energy Monitor

**Repository**: [PLACEHOLDER: GitHub link]

Non-intrusive load monitoring (NILM) system using current transformers (SCT-013-000) connected to ESP32, transmitting real-time power measurements to MQTT broker. ML classifier (Random Forest) identifies individual appliances from aggregate current signatures. LLM-powered weekly energy report generation.

**Tech Stack**: ESP32 · MQTT · Node-RED · InfluxDB · Grafana · Python · Scikit-learn · LangChain

---

### Project 11: Agricultural IoT Platform with Edge AI

**Repository**: [PLACEHOLDER: GitHub link]

Multi-parameter agricultural monitoring system (soil moisture, temperature, humidity, light intensity, pH) deployed on Arduino Mega + Raspberry Pi gateway. Random Forest anomaly detection runs at the edge; LLM advisory (Phi-3-mini via Ollama) provides NL crop management recommendations based on sensor trends.

**Tech Stack**: Arduino · Raspberry Pi · Python · MQTT · SQLite · Scikit-learn · Ollama · Phi-3-mini

---

## 🖼️ Computer Vision & Multi-Modal AI

### Project 12: Multi-Modal Sentiment Analysis with LLM Enhancement

**Repository**: [PLACEHOLDER: GitHub link]

Late-fusion multi-modal architecture combining:
- **Textual**: BERT-base fine-tuned on sentiment data (768-d embedding)
- **Visual**: ResNet-50 facial expression features (2048-d → 256-d projection)
- **Acoustic**: MFCC + LSTM (128-d sequence encoding)
- **LLM Meta-Reasoning**: GPT-4 receives concatenated modality summaries and provides final sentiment analysis with explanation

Evaluated on CMU-MOSI benchmark (sentiment intensity regression + binary classification).

**Results**: 4.3% relative improvement in binary accuracy over best unimodal (BERT) baseline on CMU-MOSI

**Tech Stack**: PyTorch · BERT · OpenCV · HuggingFace · OpenAI API · LangChain

---

### Project 13: Document Layout Understanding with Vision-Language Models

**Repository**: [PLACEHOLDER: GitHub link]

Fine-tuned PaddleOCR + LLaVA pipeline for structured information extraction from engineering datasheets and technical documents. Extracts tables, diagrams, and text into structured JSON for downstream RAG pipeline consumption.

**Tech Stack**: PaddleOCR · LLaVA · Python · FastAPI · Pydantic

---

## 🎓 Academic Course Projects

### M.Tech Data Science Course Projects

| Project | Course | Description | Tech |
|---------|--------|-------------|------|
| Sentiment Analysis Comparative Study | NLP | LSTM vs. BERT vs. GPT-2 on SST-2 | PyTorch, HuggingFace |
| Big Data Pipeline for Twitter Analytics | Big Data | Spark streaming + ML classification | Apache Spark, Kafka |
| Predictive Analytics for Student Performance | ML | Feature engineering + ensemble methods | Scikit-learn, XGBoost |
| Image Segmentation for Medical Imaging | Computer Vision | U-Net on DRIVE retinal dataset | PyTorch, segmentation-models |

### B.E. ECE Course Projects

| Project | Course | Description | Tech |
|---------|--------|-------------|------|
| OFDM System Simulation | Digital Communications | BER analysis with AWGN/Rayleigh channels | MATLAB |
| 4-bit ALU on FPGA | VLSI / Digital Design | Structural VHDL implementation | VHDL, Xilinx Vivado |
| Adaptive FIR Filter (LMS) | DSP | Real-time noise cancellation demo | MATLAB / C (TMS320) |
| Smart Irrigation Controller | Embedded Systems | Soil-moisture based Arduino controller | Arduino C, soil sensors |
| AM/FM Transceiver Design | Analog Electronics | PCB-level design and characterisation | KiCad, LTSpice |
| UART Protocol Implementation | Microprocessors | Bare-metal STM32 UART driver | C, STM32CubeIDE |

---

## 📚 Research Reproductions & Experiments

| Paper | Original Authors | My Implementation | Notes |
|-------|-----------------|------------------|-------|
| "LoRA: Low-Rank Adaptation of LLMs" | Hu et al., 2021 | PyTorch from scratch | Validated on GPT-2 fine-tuning |
| "RAFT: Adapting Language Model to Domain" | Zhang et al., 2024 | Domain RAG + fine-tune combo | Extended to ECE domain |
| "HyDE: Hypothetical Document Embeddings" | Gao et al., 2022 | LangChain integration | Benchmarked on scientific QA |
| "QLoRA: Efficient Finetuning of Quantized LLMs" | Dettmers et al., 2023 | BitsAndBytes + PEFT | Replicated core results |

---

## ⚙️ How to Run

### LLM Projects Setup

```bash
# Clone repository
git clone https://github.com/Harshithprashanth/[repo-name]
cd [repo-name]

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, ANTHROPIC_API_KEY etc.

# Start vector database
docker run -d --name chromadb -p 8000:8000 chromadb/chroma

# Run application
python src/app.py --config config/config.yaml
```

### ML / Signal Processing Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy scipy librosa pandas scikit-learn matplotlib

# Run training
python train.py --epochs 30 --batch_size 32 --lr 1e-4

# Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### Hardware / IoT Setup

```bash
# Raspberry Pi edge inference setup
pip install ollama fastapi uvicorn paho-mqtt influxdb-client

# Pull quantized model
ollama pull mistral:7b-instruct-v0.2-q4_K_M

# Start IoT gateway
python iot_gateway.py --broker mqtt://localhost:1883 --db influxdb://localhost:8086
```

---

*Last Updated: March 2026*

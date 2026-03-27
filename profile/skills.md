# 🛠 Technical Skills — Harshith Prashanth

---

## 📋 Table of Contents

- [Skills Overview](#skills-overview)
- [LLM & LLMOps Skills (Featured)](#llm--llmops-skills-featured)
- [Machine Learning & AI](#machine-learning--ai)
- [Data Science & Analytics](#data-science--analytics)
- [Signal Processing](#signal-processing)
- [Embedded Systems & Hardware](#embedded-systems--hardware)
- [Software Engineering & DevOps](#software-engineering--devops)
- [Academic & Research Tools](#academic--research-tools)
- [Soft Skills](#soft-skills)

---

## 📊 Skills Overview

| Domain | Key Technologies | Level |
|--------|-----------------|-------|
| **LLM Engineering** | LangChain, HuggingFace, LoRA/QLoRA, RAG | Advanced |
| **LLMOps** | LangSmith, W&B, MLflow, BentoML, FastAPI | Intermediate–Advanced |
| **ML/DL Frameworks** | PyTorch, TensorFlow, scikit-learn | Advanced |
| **Signal Processing** | MATLAB, SciPy, DSP algorithms | Advanced |
| **Data Science** | Pandas, NumPy, statistical modelling | Advanced |
| **Embedded Systems** | Arduino, Raspberry Pi, STM32, FPGA | Advanced |
| **Programming** | Python, C/C++, MATLAB | Advanced |
| **DevOps** | Git, Docker, Linux, CI/CD | Intermediate |

---

## 🤖 LLM & LLMOps Skills (Featured)

### LLM Development

#### Transformer Architecture
**Level**: Advanced

Deep understanding of the complete transformer architecture acquired through paper study, implementation from scratch, and debugging fine-tuned models:
- Multi-head self-attention: query/key/value projections, attention score computation, causal masking
- Positional encodings: sinusoidal, rotary position embeddings (RoPE), ALiBi
- Tokenization: BPE, WordPiece, SentencePiece; vocabulary size trade-offs; special token handling
- Scaling laws (Chinchilla, Kaplan et al.): compute-optimal training, model size vs. data size trade-offs
- Model families: GPT-2/3/4, BERT/RoBERTa/DeBERTa, T5/FLAN-T5, LLaMA-2/3, Mistral-7B/8x7B, Mixtral, Phi-2/3, CodeLlama, DeepSeek-Coder
- Can implement a complete GPT-style model from scratch in PyTorch including training loop, evaluation, and sampling

#### Fine-tuning (Full / LoRA / QLoRA)
**Level**: Advanced

Comprehensive fine-tuning expertise:
- **Full fine-tuning**: Gradient accumulation, learning rate scheduling (cosine with warmup), mixed precision (bf16/fp16), DeepSpeed ZeRO stages 1–3, gradient checkpointing
- **LoRA (Low-Rank Adaptation)**: Rank selection (r = 4–64), alpha scaling, target module identification (q_proj/v_proj/k_proj/o_proj/gate_proj/up_proj/down_proj), adapter merging
- **QLoRA**: 4-bit NormalFloat (NF4) quantization via BitsAndBytes, double quantization, paged Adam optimizer; achieving 65% VRAM reduction in practice
- **SFT (Supervised Fine-tuning)**: Instruction formatting (Alpaca, ChatML, ShareGPT templates), dataset quality filtering, loss masking on prompt tokens
- **DPO (Direct Preference Optimisation)**: preference dataset construction, reference model setup, DPO loss implementation
- **RLHF**: reward model training, PPO fine-tuning via TRL; theoretical understanding of RLAIF and Constitutional AI
- Models fine-tuned: Llama-2-7B/13B, Mistral-7B, Phi-2, CodeLlama-7B

#### Prompt Engineering
**Level**: Advanced

- Zero-shot and few-shot prompting: example selection strategies, format sensitivity analysis
- Chain-of-Thought (CoT): standard CoT, zero-shot CoT ("Let's think step by step"), self-consistency via majority voting
- Tree-of-Thought (ToT): branching reasoning, evaluation-guided search
- ReAct (Reasoning + Acting): interleaved thought-action-observation traces for tool use
- DSPy: declarative LLM programming, teleprompter-based automatic prompt optimisation
- Prompt A/B testing: systematic evaluation of prompt variants using LLM-as-judge and human evaluation
- System prompt engineering for persona definition, output format enforcement, safety constraints

#### Retrieval-Augmented Generation (RAG)
**Level**: Advanced

Progressive RAG expertise from naive to production-grade:
- **Naive RAG**: chunking → embedding → vector similarity search → context injection
- **Advanced RAG techniques**:
  - HyDE (Hypothetical Document Embeddings): generate hypothetical answer, embed for retrieval
  - Multi-query decomposition: break complex queries into sub-queries for parallel retrieval
  - Contextual compression: extract only relevant segments from retrieved chunks
  - Cross-encoder re-ranking: improve retrieval precision with bi-encoder + cross-encoder pipeline
  - Parent-child chunking: retrieve small chunks, expand to parent context for generation
  - Hybrid retrieval: BM25 (sparse) + dense vector retrieval with reciprocal rank fusion
  - Self-RAG: retrieve-then-reflect; critique and revise retrieved context
- **Vector databases**: ChromaDB (local, fast prototyping), FAISS (efficient large-scale), Pinecone (managed, production), Weaviate (graph-structured)
- **Embedding models**: text-embedding-ada-002, text-embedding-3-large, sentence-transformers (all-MiniLM, all-mpnet, instructor-xl), domain-specific fine-tuned embeddings
- **Evaluation**: RAGAS (faithfulness, answer relevancy, context precision, context recall), TruLens, DeepEval

#### LLM APIs
**Level**: Advanced

- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4o, GPT-4 Vision; function calling, structured outputs (JSON mode), streaming, token usage optimization, rate limiting, error handling
- **Anthropic**: Claude Haiku, Sonnet, Opus; extended thinking, Constitutional AI principles
- **Google Gemini**: Gemini Pro, Gemini Flash, Gemini Ultra; multi-modal capabilities
- **Cohere**: Command R/R+, embedding models, rerank API
- **LiteLLM**: unified proxy for multi-provider API management with fallback routing

---

### LLMOps Stack

#### Orchestration Frameworks
| Framework | Proficiency | Key Capabilities |
|-----------|------------|-----------------|
| LangChain | Advanced | Chains, agents, memory, retrieval, tools |
| LangGraph | Advanced | Stateful multi-agent workflows, cycles, persistence |
| LlamaIndex | Intermediate–Advanced | Data connectors, query engines, multi-modal |
| LangSmith | Intermediate–Advanced | Tracing, evaluation, prompt hub, dataset management |
| Semantic Kernel | Basic–Intermediate | Plugin system, planner, memory |

#### Serving & Deployment
| Tool | Proficiency | Use Case |
|------|------------|---------|
| FastAPI | Intermediate–Advanced | REST API endpoints for LLM services |
| BentoML | Intermediate | Model packaging, serving, scaling |
| Ollama | Intermediate–Advanced | Local LLM serving with GGUF models |
| vLLM | Intermediate | High-throughput LLM serving with continuous batching |
| Text Generation Inference (TGI) | Intermediate | HuggingFace production serving |
| llama.cpp | Intermediate | CPU/GPU inference for GGUF quantized models |

#### Monitoring & Experiment Tracking
| Tool | Proficiency | Use Case |
|------|------------|---------|
| Weights & Biases | Intermediate–Advanced | Fine-tuning experiment tracking, model registry |
| MLflow | Intermediate–Advanced | Experiment tracking, model versioning, serving |
| LangSmith | Intermediate–Advanced | LLM call tracing, evaluation datasets, monitoring |
| DVC | Intermediate | Dataset versioning, pipeline reproducibility |

#### LLM Evaluation
| Framework | Proficiency | Metrics |
|-----------|------------|---------|
| RAGAS | Intermediate–Advanced | Faithfulness, answer relevancy, context precision/recall |
| DeepEval | Intermediate | Hallucination, answer relevancy, contextual precision |
| TruLens | Intermediate | RAG triad: groundedness, context relevance, answer relevance |
| Custom LLM-as-Judge | Intermediate–Advanced | Domain-specific rubric-based evaluation |

#### Safety & Guardrails
| Tool | Proficiency |
|------|------------|
| Guardrails AI | Intermediate |
| NeMo Guardrails | Basic–Intermediate |
| Constitutional AI principles | Theoretical–Practical |

---

## 🔬 Machine Learning & AI

### Deep Learning Frameworks

| Framework | Proficiency | Key Applications |
|-----------|------------|-----------------|
| **PyTorch** | Intermediate–Advanced | Custom architectures, fine-tuning, research implementations |
| **TensorFlow / Keras** | Advanced | Production model training, transfer learning, TFLite |
| **HuggingFace Transformers** | Advanced | Pre-trained models, fine-tuning, pipeline API |
| **HuggingFace PEFT** | Advanced | LoRA, QLoRA, prefix tuning, prompt tuning |
| **TRL** | Advanced | SFT trainer, DPO trainer, reward model training |

### Classical ML

| Tool | Proficiency |
|------|------------|
| Scikit-learn | Advanced |
| XGBoost | Intermediate–Advanced |
| LightGBM | Intermediate–Advanced |
| Optuna (hyperparameter optimisation) | Intermediate |
| SHAP (explainability) | Intermediate |

---

## 📊 Data Science & Analytics

| Tool / Library | Proficiency | Key Uses |
|---------------|------------|---------|
| **Pandas** | Advanced | Data manipulation, cleaning, aggregation, time-series |
| **NumPy** | Advanced | Numerical computing, array operations, linear algebra |
| **Matplotlib / Seaborn** | Advanced | Statistical visualisation, research-quality plots |
| **Plotly / Dash** | Intermediate | Interactive visualisation, dashboards |
| **Jupyter / JupyterLab** | Advanced | Exploratory analysis, reproducible research notebooks |
| **Apache Spark** | Basic–Intermediate | Distributed data processing for large-scale datasets |
| **SQL** | Intermediate | Relational database querying and analysis |
| **SciPy** | Advanced | Scientific computing, signal processing, statistics |

---

## 📡 Signal Processing

| Skill | Proficiency |
|-------|------------|
| Discrete Fourier Transform (DFT) / FFT | Advanced |
| FIR / IIR Digital Filter Design | Advanced |
| Adaptive Filtering (LMS, RLS, Kalman) | Intermediate–Advanced |
| Wavelet Transform & Multi-Resolution Analysis | Intermediate–Advanced |
| Time-Frequency Analysis (STFT, spectrogram) | Advanced |
| MFCC & Audio Feature Extraction | Advanced |
| MATLAB Signal Processing Toolbox | Advanced |
| SciPy Signal Processing | Advanced |
| Waveform Analysis & ECG/EEG Processing | Intermediate–Advanced |

---

## 🔌 Embedded Systems & Hardware

| Platform / Tool | Proficiency | Key Projects |
|----------------|------------|-------------|
| **Arduino (AVR/ARM)** | Advanced | IoT sensors, actuator control, real-time data logging |
| **Raspberry Pi 3B/4B/5** | Advanced | Edge AI, MQTT gateway, local LLM inference |
| **STM32 (Cortex-M)** | Intermediate | Real-time control, UART/SPI/I2C peripherals, FreeRTOS |
| **FPGA (Xilinx Artix-7 / Basys 3)** | Intermediate | DSP hardware acceleration, signal acquisition |
| **PCB Design (KiCad)** | Basic–Intermediate | Schematic capture, layout, gerber file generation |
| **MQTT Protocol** | Advanced | IoT data streaming, broker configuration |
| **InfluxDB / Grafana** | Intermediate | Time-series storage, real-time monitoring dashboards |

---

## 🛠 Software Engineering & DevOps

| Tool / Practice | Proficiency |
|----------------|------------|
| **Git / GitHub** | Advanced |
| **Docker** | Intermediate |
| **Linux / Bash scripting** | Intermediate |
| **Python (PEP-8, type hints, docstrings)** | Advanced |
| **C / C++ (embedded context)** | Intermediate–Advanced |
| **REST API design (FastAPI, Flask)** | Intermediate–Advanced |
| **GitHub Actions (CI/CD)** | Intermediate |
| **Virtual environments (venv, conda)** | Advanced |
| **VS Code / PyCharm** | Advanced |

---

## 📝 Academic & Research Tools

| Tool | Proficiency | Use |
|------|------------|-----|
| **LaTeX** | Intermediate | Academic papers, thesis writing, technical reports |
| **Overleaf** | Intermediate | Collaborative LaTeX editing |
| **Zotero** | Intermediate | Reference management, citation organisation |
| **Obsidian / Notion** | Intermediate | Research notes, knowledge management |
| **arXiv** | Advanced | Literature search, pre-print submission |
| **Semantic Scholar API** | Intermediate | Automated literature review pipelines |

---

## 🤝 Soft Skills

| Skill | Description |
|-------|-------------|
| **Technical Writing** | Authoring clear, rigorous academic papers, technical reports, and documentation |
| **Research Methodology** | Experimental design, hypothesis formulation, ablation studies, statistical analysis |
| **Presentation** | Communicating complex technical content to both specialist and general audiences |
| **Collaboration** | Cross-functional team collaboration on hardware-software integrated projects |
| **Self-Directed Learning** | Rapid acquisition of new frameworks, architectures, and research domains |
| **Problem Decomposition** | Breaking complex research challenges into tractable sub-problems |

---

*Last Updated: March 2026*

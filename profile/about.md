# 🧑‍💻 About Me — Harshith Prashanth

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/[PLACEHOLDER: LinkedIn-username])
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Harshithprashanth)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:[PLACEHOLDER: email@domain.com])

---

## 📋 Table of Contents

- [Professional Biography](#professional-biography)
- [Academic Journey](#academic-journey)
- [The ECE–Data Science–LLM Intersection](#the-ece-data-science-llm-intersection)
- [LLM & LLMOps Journey](#llm--llmops-journey)
- [Research Philosophy](#research-philosophy)
- [Goals & Vision](#goals--vision)

---

## 👤 Professional Biography

I am Harshith Prashanth, an Electronics and Communication Engineering graduate now pursuing my Master of Technology in Data Science, with an unwavering focus on research and development at the convergence of large language models, intelligent signal processing, and embedded computing systems. My career trajectory has been defined by a relentless curiosity that refuses to respect disciplinary boundaries — moving from transistor-level circuit design and DSP algorithm implementation through statistical machine learning into the frontier of foundation model engineering and LLMOps.

My undergraduate years immersed me in the mathematical and physical foundations of signal processing, communication theory, embedded systems, and hardware design. This grounding gave me something rare in the LLM space: an intuitive understanding of resource constraints, real-time processing requirements, signal fidelity, and the engineering realities of deploying intelligent systems in physical environments. These are not abstract concerns for me — they are daily engineering challenges I have navigated through hardware prototyping, FPGA programming, and embedded system deployment.

The transition into data science and machine learning was organic rather than abrupt. I began applying ML techniques to signal processing problems during my final undergraduate years — using random forests for vibration fault classification, convolutional networks for ECG denoising, and LSTMs for time-series prediction in IoT sensor streams. Each project deepened my appreciation for the interplay between domain knowledge and data-driven modelling.

The discovery of transformer architectures and the emergence of large language models redirected the entire arc of my research ambitions. I became deeply absorbed in understanding not just how to use LLMs, but how they work, how they fail, and how they can be adapted to specialized technical domains. I systematically built expertise in fine-tuning paradigms (LoRA, QLoRA, full fine-tuning), retrieval-augmented generation architecture, production LLMOps pipelines, multi-agent orchestration, and LLM evaluation frameworks.

Today my research centres on three grand challenges: (1) adapting foundation models to engineering and scientific domains without catastrophic forgetting or hallucination amplification; (2) deploying LLM reasoning capabilities at the edge where connectivity and compute are constrained; and (3) creating hybrid architectures that allow LLMs to reason meaningfully about physical signals — vibration, electrical, biological, and RF. I approach each challenge with academic rigour, empirical discipline, and an engineer's pragmatic focus on measurable outcomes.

I am passionate about open science, reproducible research, and knowledge sharing. I believe that the most consequential advances in AI will come from researchers who combine deep theoretical understanding with hands-on engineering experience — and I strive to embody that combination in every project I undertake. I am actively seeking PhD opportunities, research collaborations, and R&D roles where I can contribute to and learn from world-class teams working on foundational AI challenges.

---

## 🎓 Academic Journey

### Phase 1 — Electronics & Communication Engineering (B.E.)

My formal education began with a Bachelor of Engineering in Electronics and Communication Engineering at [PLACEHOLDER: University Name]. The programme provided rigorous training in the mathematical foundations that underpin all modern AI: linear algebra, probability theory, signal analysis, information theory, and numerical methods — all taught through the lens of physical systems and engineering applications.

Key formative experiences during this period included:

- **Digital Signal Processing**: Deep engagement with DFT, FFT, FIR/IIR filter design, adaptive filtering (LMS, RLS), and time-frequency analysis. I developed an appreciation for the trade-offs between computational complexity and signal quality that now informs my thinking about LLM efficiency.
- **Embedded Systems**: Extensive hands-on work with Arduino, Raspberry Pi, STM32, and FPGA platforms. Programming resource-constrained devices taught me to think carefully about memory, compute, and power budgets — directly applicable to edge AI deployment.
- **Communication Systems**: Study of modulation, channel coding, MIMO systems, and wireless protocols provided intuition about information encoding and transmission that maps naturally to the tokenization and attention mechanisms in transformer models.
- **VLSI Design**: Understanding how algorithms map to silicon forced a hardware-aware perspective that continues to shape my approach to model optimization and deployment.

By my third year, I had begun exploring machine learning — initially as a tool to solve signal processing problems, then as an object of study in its own right. My final-year project applied deep learning to intelligent fault detection in industrial IoT systems, winning recognition at the departmental level and igniting my commitment to research-oriented work.

### Phase 2 — Machine Learning Discovery & NLP Exploration

Following graduation, I spent a period of intensive self-directed learning and project work, systematically building expertise in ML and deep learning through the Stanford ML Specialization, the deeplearning.ai Deep Learning Specialization, and hands-on project work spanning computer vision, time-series analysis, and NLP. I developed fluency with TensorFlow, Keras, PyTorch, and scikit-learn, and began contributing to open-source ML projects.

My encounter with the original BERT paper and the subsequent GPT family of models was a turning point. The idea that a single pre-trained language model could be adapted to dozens of tasks — including tasks with no obvious linguistic surface form — struck me as one of the most profound engineering innovations of the past decade. I began studying transformer architectures in depth, reading foundational papers (Attention Is All You Need, BERT, GPT-3, InstructGPT, LoRA, RAG) and implementing key components from scratch in PyTorch.

### Phase 3 — LLMs, LLMOps & M.Tech Data Science

The commencement of my M.Tech in Data Science at [PLACEHOLDER: University Name] in 2024 provided the formal academic framework to pursue LLM research with institutional support. My coursework in advanced ML, statistical learning theory, NLP, and big data analytics has reinforced and extended my self-directed learning, while providing access to research supervision, computational resources, and an active research community.

During this phase I have built and deployed five production-grade LLM projects (detailed in [llm_work.md](llm_work.md)), developed a systematic understanding of the LLMOps engineering stack, and begun formulating original research contributions in domain-adaptive LLMs, LLM-signal processing fusion, and edge LLM deployment.

---

## 🔗 The ECE–Data Science–LLM Intersection

The combination of ECE background, data science rigour, and LLM engineering expertise creates a distinctive research perspective that I believe is both rare and highly valuable:

**Hardware Thinking × LLM Engineering**
My embedded systems background gives me first-principles intuition about the resource constraints that govern LLM deployment in practice. Where many LLM researchers treat deployment as an afterthought, I approach model design with constant awareness of memory bandwidth, compute budgets, and latency requirements. This perspective has directly motivated my work on QLoRA (65% VRAM reduction), GGUF quantization for edge inference, and the design of lightweight RAG pipelines suitable for edge hardware.

**Signal Intelligence × Language Model Reasoning**
Few researchers have explored the intersection of physical signal processing and LLM reasoning. My ECE background enables me to formulate research questions that others cannot easily pose: Can transformers learn to reason about spectral characteristics of vibration signals? What tokenization strategies best preserve temporal structure in time-series data for LLM consumption? Can multi-modal architectures fuse DSP features with language embeddings for joint signal-text reasoning? These are the questions that motivate my most exciting ongoing research.

**Real-Time Constraints × Production LLMOps**
IoT and embedded systems operate under strict latency budgets. Designing LLM applications that must respond within milliseconds on edge hardware — rather than seconds via cloud API — requires a fundamentally different engineering approach. My background in real-time embedded programming directly informs my LLMOps work on latency optimization, efficient inference, and edge deployment patterns.

---

## 🤖 LLM & LLMOps Journey

### Stage 1 — Transformer Architecture & Theory (2022–2023)

Beginning with careful study of the foundational papers, I built a deep theoretical understanding of the transformer architecture: multi-head self-attention, positional encodings, layer normalisation, feed-forward sub-layers, and the encoder-decoder structure. I implemented a GPT-style autoregressive language model from scratch in PyTorch, training it on a small technical corpus to validate my understanding. This foundation has proved invaluable for debugging fine-tuned models and reasoning about failure modes.

### Stage 2 — Fine-Tuning & Domain Adaptation (2023)

Systematic exploration of fine-tuning approaches:
- **Full fine-tuning** on small domain-specific datasets to establish baselines
- **LoRA** (Low-Rank Adaptation): understanding rank selection, alpha scaling, target module choice, and the mathematical intuition behind low-rank weight updates
- **QLoRA**: 4-bit NormalFloat quantization combined with LoRA adapters via BitsAndBytes; achieving dramatic VRAM reduction without commensurate quality loss
- **Instruction tuning**: dataset formatting (Alpaca, ShareGPT, ChatML templates), quality filtering, and the impact of instruction diversity on zero-shot generalisation
- **DPO** (Direct Preference Optimisation): alignment without a separate reward model

Models fine-tuned: Llama-2 (7B, 13B), Mistral-7B, Phi-2, CodeLlama-7B.

### Stage 3 — RAG Systems & Retrieval Engineering (2023–2024)

Progressive development of RAG expertise from naive vector search to production-grade retrieval systems:
- **Naive RAG**: embedding + similarity search as a baseline
- **Advanced RAG**: HyDE (Hypothetical Document Embeddings), multi-query decomposition, contextual compression, cross-encoder re-ranking, parent-child chunking, hybrid BM25+dense retrieval
- **Vector databases**: ChromaDB, FAISS, Pinecone, Weaviate — trade-offs in scalability, update latency, and filtering capabilities
- **Chunking strategies**: fixed-size, semantic, recursive character, and document-structure-aware chunking; impact on retrieval precision
- **RAGAS evaluation**: measuring faithfulness, answer relevancy, context precision, and context recall

### Stage 4 — LLMOps & Production Engineering (2024–Present)

Building production-grade LLM infrastructure:
- **Experiment tracking**: W&B and MLflow for fine-tuning experiments, prompt version management
- **Serving infrastructure**: FastAPI, BentoML, vLLM, Ollama
- **Observability**: LangSmith tracing, custom evaluation pipelines, continuous monitoring
- **Multi-agent systems**: LangGraph for stateful agent workflows, ReAct patterns, tool use
- **Guardrails**: Guardrails AI, NeMo Guardrails for output validation and safety

---

## 🧭 Research Philosophy

> *"Good research is rigorous, reproducible, and honest about its limitations."*

**Rigour**: I believe in understanding methods deeply rather than treating them as black boxes. Every model I train, I understand the loss landscape. Every RAG system I build, I understand the retrieval mechanics. This depth is what separates researchers from practitioners, and it is what allows me to innovate rather than merely apply existing tools.

**Reproducibility**: All research code and datasets I produce are released publicly with full documentation, environment specifications, and seed-controlled experiments. Reproducibility is not a bureaucratic requirement — it is an ethical obligation to the research community.

**Empirical honesty**: I report full results including failure modes, not just cherry-picked successes. Negative results are scientifically valuable and I treat them as such.

**Real-world impact**: I am drawn to research problems with clear pathways to practical impact — intelligent industrial systems, accessible healthcare technology, efficient AI on constrained devices. Academic novelty in isolation is insufficient; I want my work to matter outside the conference room.

**Ethical AI development**: LLMs carry significant risks of hallucination, bias, and misuse. I treat responsible AI not as a compliance checklist but as a core design constraint, incorporating guardrails, evaluation, and uncertainty quantification from the outset of every project.

**Open science**: I believe in sharing pre-prints, datasets, benchmarks, and tools openly. The pace of AI progress depends on community collaboration, and I am committed to being a net contributor to that ecosystem.

---

## 🎯 Goals & Vision

### Short-Term (2024–2026)
- Complete M.Tech in Data Science with distinction, maintaining high CGPA
- Publish 2–3 peer-reviewed papers at top venues (ACL, EMNLP, NeurIPS, or ICLR) on LLM domain adaptation, LLM-signal processing fusion, or edge LLM deployment
- Open-source a production-grade LLMOps toolkit for engineering domain RAG systems
- Build a strong HuggingFace model hub presence with fine-tuned models for technical domains
- Secure a prestigious research internship at a leading AI laboratory or industrial R&D division

### Medium-Term (2026–2030)
- Pursue a PhD in Computer Science, Electrical Engineering, or an interdisciplinary AI programme at a world-class research university
- Establish a coherent and recognized research identity at the intersection of LLMs, signal intelligence, and edge AI
- Build a track record of top-tier publications, open-source contributions, and conference presentations
- Mentor undergraduate and fellow graduate researchers, contributing to research community growth

### Long-Term (2030+)
- Lead an R&D research team or laboratory at an academic institution or advanced industrial research centre
- Pioneer the field of physics-aware language models — foundation models that understand and reason about the physical world through its signals
- Contribute to the responsible development of AI systems that are both powerful and trustworthy
- Build lasting open-source infrastructure that enables the broader research community to work at the ECE-LLM intersection

---

*Last Updated: March 2026*

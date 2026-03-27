# Research Statement

## Harshith Prashanth

*M.Tech Data Science Scholar | Electronics & Communication Engineer*  
[PLACEHOLDER: University Name] | [PLACEHOLDER: email@domain.com] | GitHub: github.com/Harshithprashanth

---

## Overarching Research Theme

My research programme is unified by a single motivating question: *How can large language models be made to understand, reason about, and act upon the physical world as conveyed through electronic signals and sensor data?* This question sits at the junction of three disciplines that I have cultivated in depth — electronics and communication engineering, data science and machine learning, and large language model engineering and deployment. The convergence of these fields creates a research space that is both scientifically rich and practically consequential, with applications spanning industrial automation, healthcare informatics, wireless communications, and autonomous embedded systems.

The past decade has witnessed extraordinary advances in the capabilities of artificial intelligence systems. Deep learning models have surpassed human performance on perceptual benchmarks. Large language models exhibit emergent reasoning capabilities across a remarkable range of linguistic and symbolic tasks. Yet a persistent gap remains: today's most capable AI systems are fundamentally blind to the physical world in its native form. They process text and images, but they cannot natively reason about the vibration signature of a failing bearing, the ST-segment elevation in an ECG trace, the spectral occupancy of a congested wireless channel, or the multi-dimensional pattern of sensor readings from an industrial process drifting toward a fault condition. Bridging this gap — bringing language model intelligence into contact with physical signal intelligence — is the central ambition of my research.

---

## Background and Motivation

My academic formation began with a Bachelor of Engineering in Electronics and Communication Engineering, where I developed deep competence in digital signal processing, communication theory, embedded systems, and hardware design. This technical foundation is not merely biographical context — it is an active component of my research approach. The ECE perspective provides intuitions about signal fidelity, noise characteristics, bandwidth constraints, and real-time processing requirements that are absent from most NLP and ML training programmes. When I design a RAG pipeline for an IoT sensor query system, I am simultaneously reasoning about the statistical properties of sensor noise, the temporal autocorrelation of process data, and the appropriate granularity for embedding sensor context — considerations that would not arise for a researcher without this background.

The transition to data science and machine learning began with applying learned models to signal processing problems: deep neural networks for ECG denoising, random forests for vibration fault classification, LSTMs for IoT time-series anomaly detection. These projects demonstrated the power of learned representations but also exposed their limitations — models trained on one domain of signals generalised poorly to others, and their predictions provided no interpretable explanation accessible to human operators. Language models, I hypothesised, could address both limitations: their broad pre-training provides semantic generalization, and their textual output is inherently interpretable.

The emergence of powerful LLMs and the associated LLMOps engineering stack has created an unprecedented opportunity. For the first time, we have models capable of sophisticated multi-step reasoning that can also be efficiently adapted to specialised domains, integrated with external knowledge retrieval systems, and deployed on resource-constrained edge hardware. The engineering challenges of doing this reliably — building production-quality LLM pipelines that are accurate, observable, efficient, and safe — constitute a second major research thrust that I pursue alongside the more fundamental scientific questions.

---

## Research Agenda

### Thrust 1: Domain-Adaptive LLMs for Engineering Sciences

**Motivation and Gap**: General-purpose large language models, including state-of-the-art models such as GPT-4 and Claude Opus, exhibit characteristic failure modes when applied to specialised engineering domains. They hallucinate technical details — incorrect circuit values, non-existent standards references, or plausible-sounding but physically impossible signal processing claims. This failure is attributable to the nature of pre-training corpora: web-scale text is dominated by general and informal content, with limited coverage of engineering textbooks, technical standards documents, and scientific journal papers at the depth required for professional competence.

**Research Questions**: (1) What parameter-efficient fine-tuning (PEFT) configurations — LoRA rank, alpha, target module selection, learning rate schedule — yield optimal domain adaptation for engineering texts while preserving general capability? (2) When is RAG preferable to fine-tuning for engineering domain adaptation, and what factors determine this choice? (3) What evaluation frameworks best capture engineering-domain LLM competence, and what failure modes do existing benchmarks miss?

**Methodology**: I take a rigorous empirical approach, conducting controlled ablation studies across a curated suite of engineering domain tasks: signal processing question-answering, embedded C code generation, circuit analysis, and communications system design. I am constructing EngLLM-Bench — a 2,400-question expert-validated benchmark across four engineering subdisciplines — to provide a standardised evaluation basis for this research. Fine-tuning experiments use QLoRA on Llama-3 and Mistral-7B with systematic variation of hyperparameters; RAG experiments compare retrieval strategies (dense, sparse, hybrid, HyDE) with controlled ablations on embedding model choice and chunking strategy.

**Expected Contributions**: A publicly released engineering domain LLM benchmark; empirical fine-tuning guidelines for technical domain adaptation; open-source tools for engineering RAG pipeline construction; publications at ACL or EMNLP.

**Impact**: Improved LLM tools for engineering education, research assistance, and industrial automation.

---

### Thrust 2: LLM-Enhanced Signal Processing and Multi-Modal Reasoning

**Motivation and Gap**: Signal processing and language modelling have evolved almost entirely independently, yet they share deep structural similarities: both are fundamentally concerned with sequence modelling, pattern recognition, and information extraction from temporal or structured data. I hypothesise that there exists a productive integration in which LLMs provide high-level reasoning and context about signals — classifying fault types from vibration signatures, interpreting ECG rhythms in clinical context, or explaining anomalous sensor readings in natural language — while classical DSP methods handle low-level feature extraction and noise reduction.

**Research Questions**: (1) What tokenization strategies best preserve temporal and spectral information when converting physical time-series to LLM-consumable token sequences? (2) Can LLMs fine-tuned on signal-description pairs learn to reason meaningfully about signal characteristics without explicit feature engineering? (3) What joint signal-text architectures, combining DSP preprocessing with transformer attention, provide the best multi-modal reasoning performance?

**Methodology**: I investigate three representation strategies for physical signals: (a) structured text descriptions of signal features (amplitude, frequency, SNR, identified patterns); (b) discrete tokenization of signal samples using VQ-VAE learned codebooks; and (c) spectral feature extraction (MFCC, wavelet coefficients) followed by projection into the LLM embedding space. Each strategy is evaluated on signal QA benchmarks spanning industrial vibration, ECG arrhythmia identification, and IoT anomaly explanation tasks.

**Expected Contributions**: A time-series tokenization framework for LLM consumption; multi-modal signal-text benchmark; novel joint architectures for signal-language reasoning; publications at ICASSP or NeurIPS.

---

### Thrust 3: Edge LLM Deployment and Hardware-Aware Optimisation

**Motivation and Gap**: The most impactful deployments of LLM intelligence are often in resource-constrained environments — factory floors, agricultural fields, clinical settings in resource-limited healthcare systems, and mobile devices. These environments are incompatible with cloud-dependent LLM APIs: connectivity is unreliable, latency requirements are strict, and data privacy constraints may prohibit sending sensor data to external servers. Edge LLM deployment — running quantized, distilled models on ARM and RISC-V hardware — is therefore a critical enabling technology for the applications I envision. Yet the field lacks rigorous, standardised benchmarks comparing deployment frameworks and quantization methods on real edge hardware.

**Research Questions**: (1) What is the Pareto frontier of model quality versus latency versus memory footprint for quantized LLMs on ARM Cortex-A platforms? (2) What quantization methods (GPTQ, GGUF, AWQ, SqueezeLLM) best preserve domain-specific knowledge acquired during fine-tuning? (3) What is the minimum viable model capability for providing useful natural language reasoning over IoT sensor data on a Raspberry Pi class device?

**Methodology**: Systematic benchmarking of GGUF-quantized Llama-3 (1B, 3B, 8B) and Phi-3-mini variants across Raspberry Pi 4B (4GB RAM), Raspberry Pi 5 (8GB RAM), NVIDIA Jetson Nano, and Orange Pi 5 Pro. Measurement methodology: tokens/second throughput, first-token latency (time-to-first-token), peak RAM usage, power draw (via INA219 shunt current sensor), and quality preservation (perplexity on held-out domain text; accuracy on domain QA tasks).

**Expected Contributions**: EdgeLLM-Bench benchmark dataset and evaluation harness; best-practice deployment guide; open-source automated quantization and benchmarking toolkit; publications at MLSys or a systems/embedded ML venue.

---

### Thrust 4: LLMOps for Scientific Research Automation

**Motivation and Gap**: Scientific research involves a substantial proportion of time spent on automatable tasks: literature search and synthesis, data processing and visualisation, experimental logging, report generation, and citation management. LLMs and agentic AI systems are well-positioned to assist with these tasks, but production-quality research automation requires careful engineering — robust evaluation, failure detection, appropriate human oversight, and reproducibility guarantees that the research community expects.

**Research Questions**: (1) What multi-agent architectures most reliably automate the literature review workflow while avoiding hallucinated citations and factually incorrect syntheses? (2) How should LLM applications in research settings be evaluated continuously to detect quality degradation over time? (3) What LLMOps infrastructure patterns enable reproducible, auditable LLM-assisted research?

**Methodology**: Evaluation-driven development: I build research automation pipelines alongside rigorous evaluation suites, measuring faithfulness, citation accuracy, synthesis quality, and time-to-completion against expert human baselines. I apply LangSmith for full pipeline observability and W&B for experiment management.

**Expected Contributions**: Open-source LLMOps toolkit for research automation; best-practice guidelines for LLM in scientific workflows; publications at NeurIPS Datasets & Benchmarks or ICLR workshops.

---

## Methodology and Research Philosophy

My approach to research is empirical, rigorous, and committed to reproducibility. I formulate explicit hypotheses, design controlled experiments with appropriate ablations and baselines, report results honestly including failure cases, and release all code, datasets, and model checkpoints publicly. I believe this commitment to open science is both an ethical obligation and a practical necessity for progress in the field.

I apply standard quantitative evaluation methodology: held-out test sets, statistical significance testing for comparative claims, multiple independent runs for stochastic methods, and domain-expert validation for evaluation rubrics when automated metrics are insufficient. I am attentive to the risk of benchmark overfitting and evaluation leakage, and I design evaluation protocols to mitigate these risks.

Interdisciplinary breadth is central to my approach. The most important contributions in my target research space will require fluency across NLP, signal processing, systems engineering, and LLMOps — a combination that is rare and that I have cultivated intentionally.

---

## Expected Contributions and Timeline

**Short-term (2024–2026)**: EngLLM-Bench benchmark release; QLoRA fine-tuning analysis paper; time-series tokenization framework; EdgeLLM-Bench initial release. Target venues: ACL, EMNLP, NeurIPS, ICASSP.

**Medium-term (2026–2029, PhD)**: Novel joint signal-language architectures; comprehensive edge LLM deployment framework; research automation LLMOps toolkit. Target venues: ICLR, NeurIPS, MLSys, IEEE Trans. Signal Processing.

**Long-term (2029+)**: Foundational contributions to Physics-Aware Language Models; open research infrastructure enabling the signal-processing × LLM research community.

---

## Conclusion

My research programme addresses a gap that is both scientifically significant and practically urgent: the integration of large language model reasoning with physical signal intelligence, delivered through robust production engineering and deployable on resource-constrained hardware. The three thrusts of my agenda — domain-adaptive LLMs, LLM-enhanced signal processing, and edge deployment — are mutually reinforcing: domain adaptation provides the specialised knowledge needed for signal reasoning; edge deployment makes that reasoning accessible in the physical environments where signals are generated; and LLMOps infrastructure ensures that the resulting systems are reliable, observable, and continuously improving.

I am excited to pursue this research programme with the support of a strong supervisory team, access to computational resources, and a collaborative research community. I welcome conversations with researchers working in related areas — particularly at the intersection of NLP, signal processing, embedded systems, and production AI engineering.

---

*Last Updated: March 2026*

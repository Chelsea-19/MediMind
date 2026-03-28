# MediMind 🩺

**A risk-aware, evidence-grounded, multimodal medical assistant for Chinese patient education and preliminary triage support.**

## Research Objectives
This system has been heavily refactored from a simple Jupyter Notebook prototype into a robust, publication-aligned research repository. The focus is no longer just on creating a visible UI, but rather on supporting systematic experiments, safety evaluations, and grounding answers purely in verified data.

Key architectural upgrades include:
1. **Safety and Risk Assessment Module (`src/safety`)**: Heuristic checks for red-flag emergency symptoms prior to model generation. The system explicitly triages users rather than providing diagnostic claims.
2. **Modular RAG Pipeline (`src/retrieval`)**: An abstract vector store handling ingestion, chunking limits, and retrieving traceable metadata (Source limits, chunk ID).
3. **Structured Outputs (`src/generation`)**: The model must conform to strict JSON structures indicating underlying reasoning paths, risk levels, and recommended actionable steps.
4. **Evaluation Engine (`src/evaluation`)**: Ready-to-use hooks for calculating Recall@K on retrieval, strict compliance tracking for triage correctness, and experiment log reporting in JSON formats.

## Architecture Map
- `app/`: Streamlit frontend, maintained as a thin client interface interacting with underlying Python SDK logics.
- `src/retrieval/`: Ingestion logic, sliding chunk strategy, Vector Storage (Chroma) implementation.
- `src/generation/`: Multimodal Generative LLM wrappers, dynamically formatted system prompts, prompt constraints.
- `src/safety/`: Request gating (e.g. denying prescription requests), automated high-risk triage logic.
- `src/evaluation/`: Metric functions (Recall@K, Triage Acc) mapping outputs to baseline logs.
- `configs/`: YAML based centralized configuration avoiding hard-coded logic.
- `scripts/`: Direct python execution files for headless operations (e.g., executing eval sweeps).
- `data/`: (Not pushed) Persistent DB directories and evaluation benchmarks.
- `results/`: Output directories for metric evaluations and ablation tests.

## Setup Instructions

1. **Install dependencies:**
   It is recommended to use a virtual environment or conda:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration Settings:**
   Review and adjust settings in `configs/default_config.yaml` to specify your quantized model target and inference settings. By default, it targets `Qwen/Qwen2-VL-7B-Instruct`.

3. **Run the Core Evaluation Skeleton:**
   Allows testing logic locally without launching a server:
   ```bash
   python scripts/run_eval.py
   ```

4. **Launch Application:**
   ```bash
   PYTHONPATH=. streamlit run app/app.py
   ```

## Statement of Purpose
This repository does not constitute a certified medical device and represents a **research prototype**. The system intentionally refuses to prescribe medications or provide definitive diagnoses, limiting its scope to patient general education with rigorous linkage to source medical texts. This positions the research favorably for systems evaluation or AI safety reviews without violating bioethics constraints.

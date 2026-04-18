# ColPali Multi-Modal Document Intelligence (RAG-Based QA System)

**DSAI 413 — Assignment 1**
**Author:** Abdelrahman Al-Attar

---

## Overview

A **Multi-Modal Retrieval-Augmented Generation** system using **ColPali** — a vision-language model that embeds document page images directly, bypassing traditional OCR/text extraction. The system retrieves visually relevant document pages using late-interaction (MaxSim) scoring and generates citation-backed answers.

**Live Demo: [https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run](https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run)**

Deployed on [Modal](https://modal.com) with T4 GPU + Gradio web interface.

## Architecture

```
PDF Documents
      |
  [Multi-Modal Ingestion]  →  Page images + text + tables + figure metadata
      |
  [ColQwen2.5 Embedding]  →  Multi-vector embeddings (~1030 x 128 per page)
      |                        (cached in Modal Volume)
  [MaxSim Retrieval]       →  Top-K relevant pages with source attribution
      |
  [Gradio UI]              →  Interactive QA with citations [Doc: name, Page: N]
```

### Deployment Architecture (Modal)

```
┌─────────────────────────────────────────────────────┐
│  Modal Cloud                                        │
│                                                     │
│  ┌──────────────────────┐  remote call   ┌────────┐ │
│  │  Gradio UI (CPU)     │ ──────────────→│ ColPali│ │
│  │  @modal.asgi_app()   │                │ Engine │ │
│  │  max_containers=1    │ ←──────────────│ (T4)   │ │
│  │  scaledown: 20 min   │  results+imgs  │        │ │
│  └──────────────────────┘                └────────┘ │
│                                              │      │
│                                         ┌────┴────┐ │
│                                         │ Volume  │ │
│                                         │ (cache) │ │
│                                         └─────────┘ │
└─────────────────────────────────────────────────────┘
```

- **GPU container** (`ColPaliEngine`): Loads model, embeds pages, runs MaxSim retrieval. Stays alive 5 min after last request, then scales to zero.
- **CPU container** (`ui`): Serves Gradio interface, calls GPU class remotely. Stays alive 20 min.
- **Volume**: Caches downloaded PDFs + pre-computed embeddings. First run computes everything; subsequent runs load from cache instantly.

## Dataset

**IPCC Sixth Assessment Report (AR6)** — Working Group I (The Physical Science Basis). These documents are maximally multi-modal with complex charts, geographic maps, statistical tables, scientific figures, and dense text.

## Features

| Feature | Description |
|---------|-------------|
| Multi-modal ingestion | Text + table + figure extraction per page |
| Visual embedding | ColQwen2.5 multi-vector embeddings (T4 GPU) |
| MaxSim retrieval | Late-interaction scoring for fine-grained matching |
| QA chatbot | Gradio web interface deployed on Modal |
| Source attribution | `[Doc: name, Page: N]` citations in every answer |
| Evaluation suite | Benchmark queries across text, table, figure, multi-modal |
| Similarity heatmaps | Visualize which page regions match query tokens |
| Token pooling | 3-5x compression with <3% quality loss |
| Baseline comparison | ColPali vs TF-IDF text-based retrieval |

## Deployment

### Prerequisites

```bash
pip install modal
modal setup          # authenticate with Modal (one-time)
```

### Deploy to Modal (production)

```bash
modal deploy app.py
```

Current deployment: [https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run](https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run)

### Dev mode (hot reload)

```bash
modal serve app.py
```

### Run locally (no Modal)

```bash
pip install -r requirements.txt
python app_local.py                          # no API
python app_local.py --provider gemini --api-key ...  # with Gemini
python app_local.py --share                  # public Gradio link
```

### Run the notebook

```bash
pip install -r requirements.txt
jupyter notebook ColPali_RAG_Pipeline.ipynb
```

## Project Structure

```
colpali-rag-assignment/
├── app.py                         # Modal deployment (GPU + Gradio)
├── app_local.py                   # Local Gradio app (no Modal)
├── ColPali_RAG_Pipeline.ipynb     # Full notebook with pipeline + evaluation
├── TECHNICAL_REPORT.md            # 2-page technical report
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── src/                           # Modular source code
│   ├── __init__.py
│   ├── ingestion.py               # Multi-modal PDF ingestion
│   ├── embedding.py               # ColPali model + embedding generation
│   ├── retrieval.py               # MaxSim retrieval engine
│   ├── generation.py              # LLM answer generation with citations
│   └── evaluation.py              # Multi-modal evaluation suite
└── data/                          # Auto-populated at runtime
    ├── pdfs/                      # Downloaded PDFs
    └── embeddings.pt              # Cached embeddings
```

## Cost

- Modal free tier: **$30/month** in compute credits
- T4 GPU: ~$0.59/hour — only billed while processing queries
- Scales to zero when idle — no cost when nobody is using it
- $30 free credits = ~50 hours of T4 GPU time (more than enough)

## Deliverables

1. **Codebase** — Modular `src/` package + notebook (this repo)
2. **Demo Application** — [Live on Modal](https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run)
3. **Technical Report** — `TECHNICAL_REPORT.md`
4. **Video Demonstration** — [[link](https://drive.google.com/file/d/1O7g_OihAFULYf6_1qomDVNRybPvwvcrO/view?usp=sharing)]

## References

1. Faysse et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
2. [colpali-engine](https://github.com/illuin-tech/colpali) — Official implementation
3. [ViDoRe Benchmark](https://huggingface.co/spaces/vidore/vidore-leaderboard)
4. [IPCC AR6 WG1 Report](https://www.ipcc.ch/report/ar6/wg1/)

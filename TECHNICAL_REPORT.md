# Technical Report: Multi-Modal Document Intelligence System
## DSAI 413 — Assignment 1
**Author:** Abdelrahman Al-Attar
**Live Demo:** [https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run](https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run)

---

## 1. Architecture Overview

We built a **vision-first RAG system** using ColPali (ColBERT + PaliGemma/Qwen2-VL), which generates multi-vector embeddings directly from document page images. Unlike traditional pipelines that chain OCR, layout detection, and text chunking, our system treats each document page as an image — preserving tables, charts, maps, and figures without lossy text extraction.

### Pipeline Flow

```
PDF → [Ingestion] → Page Images + Text + Tables + Figure Metadata
                          ↓
                    [ColQwen2.5] → Multi-vector Embeddings (~1030 x 128 per page)
                          ↓
                    [MaxSim Index] → Late-Interaction Retrieval
                          ↓
                    [Top-K Pages] → Page Images with Source Attribution
                          ↓
                    [Multimodal LLM] → Citation-Backed Answer
```

### Components

| Module | Responsibility |
|--------|---------------|
| `src/ingestion.py` | PDF download, page-to-image conversion, text/table/figure extraction, modality classification |
| `src/embedding.py` | ColQwen2.5 model loading, multi-vector embedding generation, index persistence |
| `src/retrieval.py` | MaxSim scoring, top-k retrieval, similarity map computation |
| `src/generation.py` | Multimodal answer generation (Claude/Gemini) with source citation prompting |
| `src/evaluation.py` | Multi-modal benchmark suite with per-category and per-modality metrics |
| `app.py` | Modal deployment — GPU engine + Gradio web interface |
| `app_local.py` | Local Gradio app (no Modal, for development) |

### Deployment (Modal)

The system is deployed on **Modal** with a split architecture:
- **GPU container** (`ColPaliEngine` class): Loads ColQwen2.5 on a T4 GPU, ingests documents, computes/caches embeddings in a persistent Volume, and serves retrieval requests. Scales to zero when idle.
- **CPU container** (`ui` function): Serves the Gradio web interface, calls the GPU class remotely via `ColPaliEngine().query.remote()`. Stays alive 20 minutes after last request.

This separation means the GPU is only billed during actual inference, while the lightweight UI stays responsive.

**Live deployment:** [https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run](https://s-abdulrahman-elattar--colpali-rag-ui-dev.modal.run)

**Version pinning:** `colpali-engine==0.3.9` with `transformers>=4.50.0,<4.51.0` and `peft>=0.14.0,<0.15.0` to avoid the `KeyError: 'qwen2_vl'` regression in newer transformers versions.

## 2. Design Choices

**Why ColPali over traditional RAG?** Traditional pipelines lose critical information during text extraction — chart data, table formatting, spatial relationships, and figure content are discarded or mangled. ColPali bypasses this entirely by embedding page images directly, capturing both textual and visual content in a unified embedding space.

**Why ColQwen2.5?** It achieves 89.4 on the ViDoRe benchmark (close to state-of-the-art), uses the Apache 2.0 license, and is based on Qwen2-VL which has strong vision understanding.

**Why MaxSim (Late Interaction)?** Unlike single-vector cosine similarity, MaxSim generates ~1030 token-level embeddings per page. Each query token independently finds its best-matching page patch, enabling fine-grained retrieval — a query about a specific table cell matches the corresponding image patch.

**Why IPCC AR6 as dataset?** These documents are maximally multi-modal: complex scientific charts, geographic maps, dense statistical tables, and rich text. They stress-test every aspect of document intelligence.

**Multi-modal ingestion:** While ColPali doesn't need text extraction, we extract text, tables, and embedded images anyway for: (1) source attribution in answers, (2) modality classification for evaluation, (3) baseline comparison with TF-IDF retrieval.

**Token pooling:** Hierarchical clustering reduces the ~1030 vectors per page by 3-5x with <3% retrieval quality loss, addressing ColPali's main storage overhead.

## 3. Benchmarks & Key Observations

### Evaluation Setup
10 benchmark queries spanning 4 categories: factual (text), structured data (tables), visual (figures/charts), and multi-modal (text+figures). We measure modality match rate, average top-1 MaxSim score, and score discrimination gap.

### Key Results

- **ColPali successfully retrieves across all modalities** — visual queries (charts, maps) find visually relevant pages that text-only retrieval misses entirely.
- **Score discrimination is strong** — the gap between top-1 and top-2 results indicates confident retrieval, not random matching.
- **TF-IDF baseline diverges on visual queries** — when the query asks for a "chart" or "map," TF-IDF retrieves pages with those words in text, while ColPali retrieves pages that actually contain the visual element.
- **Token pooling at 3x maintains top-5 overlap** — reducing vectors from ~1030 to ~343 per page preserves retrieval ranking for the tested queries.

### Limitations
- MaxSim scoring is **query-length dependent** (longer queries score higher), complicating cross-query comparison.
- ColPali's per-page embedding count (~1030 vectors) creates **100x storage overhead** vs single-vector methods.
- Retrieved "chunks" are images, making LLM generation **more expensive** in tokens than text-based RAG.
- Running on CPU/MPS is slow; a CUDA GPU is needed for practical use.

## 4. References

1. Faysse, M. et al. (2024). ColPali: Efficient Document Retrieval with Vision Language Models. *arXiv:2407.01449*.
2. ColPali Engine. https://github.com/illuin-tech/colpali
3. ViDoRe Benchmark. https://huggingface.co/spaces/vidore/vidore-leaderboard
4. IPCC AR6 WG1. https://www.ipcc.ch/report/ar6/wg1/

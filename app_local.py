"""
ColPali Multi-Modal RAG — Interactive Demo (Gradio)

Usage:
    python app.py [--provider none|claude|gemini] [--api-key KEY] [--share]

This launches a Gradio web interface for querying the indexed IPCC documents.
On first launch it downloads PDFs and computes embeddings (cached after that).
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion import ingest_documents
from src.embedding import load_model, embed_pages, save_index, load_index
from src.retrieval import retrieve_top_k
from src.generation import generate_answer

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PDF_SOURCES = {
    "IPCC_AR6_WG1_SPM": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WG1_TS": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_TS.pdf",
}

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pt"
MAX_PAGES = 25

# ──────────────────────────────────────────────
# Global state (loaded once at startup)
# ──────────────────────────────────────────────

model = None
processor = None
page_embeddings = None
all_pages = None
llm_provider = "none"
api_key = None


def initialize():
    """Load or compute everything needed for the pipeline."""
    global model, processor, page_embeddings, all_pages

    print("\n[1/3] Ingesting documents...")
    all_pages = ingest_documents(PDF_SOURCES, PDF_DIR, max_pages=MAX_PAGES)

    print("\n[2/3] Loading ColPali model...")
    model, processor, _, _ = load_model()

    print("\n[3/3] Computing embeddings...")
    if EMBEDDINGS_PATH.exists():
        page_embeddings, metadata = load_index(EMBEDDINGS_PATH)
        if len(page_embeddings) != len(all_pages):
            print("  Index stale — recomputing...")
            page_embeddings = embed_pages(all_pages, model, processor)
            save_index(page_embeddings, all_pages, EMBEDDINGS_PATH)
    else:
        page_embeddings = embed_pages(all_pages, model, processor)
        save_index(page_embeddings, all_pages, EMBEDDINGS_PATH)

    print("\nReady!")


def query_pipeline(query: str, top_k: int, provider_choice: str, user_api_key: str):
    """
    Main query handler for the Gradio interface.

    Returns: (answer_text, gallery_images, source_table)
    """
    if not query.strip():
        return "Please enter a query.", [], ""

    # Use session-level provider if user overrides in the UI
    prov = provider_choice if provider_choice != "auto" else llm_provider
    key = user_api_key.strip() if user_api_key.strip() else api_key

    # Retrieve
    results = retrieve_top_k(query, model, processor, page_embeddings, all_pages, top_k=int(top_k))

    # Generate answer
    answer = generate_answer(query, results, provider=prov, api_key=key)

    # Build gallery images with captions
    gallery = []
    for rank, (score, idx, page) in enumerate(results, 1):
        caption = f"#{rank} | {page['doc_name']} p.{page['page_num']} | score: {score:.2f} | {page.get('modality', '')}"
        gallery.append((page["image"], caption))

    # Build source attribution table
    source_lines = []
    for rank, (score, idx, page) in enumerate(results, 1):
        n_tables = len(page.get("tables", []))
        n_imgs = page.get("n_embedded_images", 0)
        source_lines.append(
            f"| {rank} | {page['doc_name']} | {page['page_num']} | "
            f"{score:.2f} | {page.get('modality', '')} | "
            f"{n_tables} table(s), {n_imgs} figure(s) |"
        )

    source_table = (
        "| Rank | Document | Page | Score | Modality | Content |\n"
        "|------|----------|------|-------|----------|----------|\n"
        + "\n".join(source_lines)
    )

    return answer, gallery, source_table


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="ColPali Multi-Modal RAG",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# ColPali Multi-Modal Document RAG\n"
            "Ask questions about **IPCC AR6 Climate Change Reports**. "
            "The system retrieves relevant document pages using vision-based embeddings "
            "(ColQwen2.5 + MaxSim) and generates answers with source citations.\n\n"
            "*DSAI 413 — Assignment 1*"
        )

        with gr.Row():
            with gr.Column(scale=2):
                query_box = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the projected temperature changes under SSP5-8.5?",
                    lines=2,
                )
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="Number of pages to retrieve",
                    )
                    provider_dropdown = gr.Dropdown(
                        choices=["auto", "none", "claude", "gemini"],
                        value="auto",
                        label="LLM Provider",
                    )
                    api_key_box = gr.Textbox(
                        label="API Key (optional)",
                        type="password",
                        placeholder="Leave empty to use default",
                    )
                submit_btn = gr.Button("Ask", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Example Queries")
                examples = gr.Examples(
                    examples=[
                        ["What is the observed global surface temperature change?"],
                        ["Show me the chart of CO2 emissions over time"],
                        ["Table showing greenhouse gas emission scenarios"],
                        ["Map of regional temperature changes across continents"],
                        ["What are the projected sea level rise scenarios?"],
                        ["How does methane contribute to global warming?"],
                    ],
                    inputs=[query_box],
                )

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_output = gr.Markdown(label="Generated Answer")

            with gr.TabItem("Retrieved Pages"):
                gallery_output = gr.Gallery(
                    label="Retrieved Document Pages",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )

            with gr.TabItem("Source Attribution"):
                source_output = gr.Markdown(label="Sources")

        submit_btn.click(
            fn=query_pipeline,
            inputs=[query_box, top_k_slider, provider_dropdown, api_key_box],
            outputs=[answer_output, gallery_output, source_output],
        )
        query_box.submit(
            fn=query_pipeline,
            inputs=[query_box, top_k_slider, provider_dropdown, api_key_box],
            outputs=[answer_output, gallery_output, source_output],
        )

    return demo


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColPali RAG Demo")
    parser.add_argument("--provider", default="none", choices=["none", "claude", "gemini"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    llm_provider = args.provider
    api_key = args.api_key

    initialize()
    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)

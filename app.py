"""
ColPali Multi-Modal RAG — Modal + Gradio Deployment

Deploy:
    modal deploy app.py

Run locally (dev mode):
    modal serve app.py

Architecture:
    - GPU class (ColPaliEngine): loads model, embeds pages, retrieves results
    - CPU function (ui): serves Gradio interface, calls GPU class remotely
    - Volume: caches model weights + pre-computed embeddings
"""

import io
import os
from pathlib import Path

import modal

# ──────────────────────────────────────────────────
# Modal App & Image Definitions
# ──────────────────────────────────────────────────

app = modal.App("colpali-rag")

# GPU image — colpali-engine 0.3.9 has ColQwen2_5 class and uses
# transformers 4.50.x which does NOT have the _MOE_TARGET_MODULE_MAPPING bug.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "colpali-engine==0.3.9",
        "transformers>=4.50.0,<4.51.0",
        "peft>=0.14.0,<0.15.0",
        "torch>=2.5.0,<2.7.0",
        "torchvision",
        "pymupdf>=1.24.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    )
)

# Lightweight web image — just Gradio + FastAPI
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.4",
    "gradio~=5.7",
    "Pillow>=10.0.0",
)

# Persistent volume for caching model weights and embeddings
volume = modal.Volume.from_name("colpali-rag-data", create_if_missing=True)
VOLUME_PATH = "/data"

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────

PDF_SOURCES = {
    "IPCC_AR6_WG1_SPM": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WG1_TS": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_TS.pdf",
}

MODEL_NAME = "vidore/colqwen2.5-v0.2"
MAX_PAGES = 25
EMBEDDINGS_FILE = "embeddings.pt"

# ──────────────────────────────────────────────────
# GPU Engine — Model Inference
# ──────────────────────────────────────────────────


@app.cls(
    image=gpu_image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=600,
    scaledown_window=300,  # keep alive 5 min after last request
)
class ColPaliEngine:
    """GPU-accelerated ColPali inference engine."""

    @modal.enter()
    def startup(self):
        """Runs once when the container starts — load model and index."""
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        print("[1/3] Loading ColQwen2.5 model...")
        self.processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
        self.model = ColQwen2_5.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        print(f"  Model loaded on CUDA")

        print("[2/3] Ingesting documents...")
        self.all_pages = self._ingest_documents()
        print(f"  {len(self.all_pages)} pages ingested")

        print("[3/3] Loading/computing embeddings...")
        self.page_embeddings = self._load_or_compute_embeddings()
        print(f"  {len(self.page_embeddings)} page embeddings ready")
        print("Engine ready!")

    def _ingest_documents(self):
        """Download PDFs and convert to page images with multi-modal extraction."""
        import requests
        import pymupdf
        from PIL import Image
        from tqdm import tqdm

        pdf_dir = Path(VOLUME_PATH) / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        all_pages = []

        for name, url in PDF_SOURCES.items():
            pdf_path = pdf_dir / f"{name}.pdf"

            # Download if not cached
            if not pdf_path.exists():
                print(f"  Downloading {name}...")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                pdf_path.write_bytes(resp.content)

            # Convert pages
            doc = pymupdf.open(str(pdf_path))
            n_pages = min(len(doc), MAX_PAGES)
            zoom = 150 / 72
            matrix = pymupdf.Matrix(zoom, zoom)

            for page_idx in range(n_pages):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=matrix)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = page.get_text("text").strip()

                # Table extraction
                tables = []
                try:
                    for table in page.find_tables().tables:
                        extracted = table.extract()
                        if extracted and len(extracted) > 1:
                            tables.append({
                                "headers": extracted[0],
                                "rows": extracted[1:],
                                "n_rows": len(extracted) - 1,
                            })
                except Exception:
                    pass

                # Image detection
                image_list = page.get_images(full=True)

                # Classify modality
                parts = []
                if len(text) > 100:
                    parts.append("text")
                if tables:
                    parts.append("table")
                if image_list:
                    parts.append("figure")
                modality = "+".join(parts) if parts else "empty"

                all_pages.append({
                    "image": img,
                    "doc_name": name,
                    "page_num": page_idx + 1,
                    "text": text,
                    "tables": tables,
                    "n_embedded_images": len(image_list),
                    "modality": modality,
                })
            doc.close()

        volume.commit()
        return all_pages

    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute from scratch."""
        import torch
        from tqdm import tqdm

        emb_path = Path(VOLUME_PATH) / EMBEDDINGS_FILE

        if emb_path.exists():
            data = torch.load(str(emb_path), weights_only=False)
            if len(data["embeddings"]) == len(self.all_pages):
                print("  Loaded cached embeddings")
                return data["embeddings"]
            print("  Cache stale — recomputing...")

        # Compute embeddings
        embeddings = []
        batch_size = 2
        for i in tqdm(range(0, len(self.all_pages), batch_size), desc="Embedding"):
            batch_imgs = [p["image"] for p in self.all_pages[i : i + batch_size]]
            processed = self.processor.process_images(batch_imgs).to(self.model.device)
            with torch.no_grad():
                batch_embs = self.model(**processed)
            for emb in batch_embs:
                embeddings.append(emb.cpu())

        # Save to volume
        metadata = [{
            "doc_name": p["doc_name"], "page_num": p["page_num"],
            "modality": p["modality"], "text": p["text"][:500],
        } for p in self.all_pages]
        torch.save({"embeddings": embeddings, "metadata": metadata}, str(emb_path))
        volume.commit()
        return embeddings

    @modal.method()
    def query(self, query_text: str, top_k: int = 3):
        """
        Run retrieval for a query. Returns serializable results.

        Returns list of dicts: {score, doc_name, page_num, modality, text_preview,
                                n_tables, n_images, image_bytes}
        """
        import torch

        # Embed query
        processed = self.processor.process_queries([query_text]).to(self.model.device)
        with torch.no_grad():
            query_emb = self.model(**processed)[0].cpu()

        # MaxSim scoring against all pages
        scores = []
        for idx, page_emb in enumerate(self.page_embeddings):
            q_norm = query_emb / query_emb.norm(dim=1, keepdim=True)
            p_norm = page_emb / page_emb.norm(dim=1, keepdim=True)
            sim = (q_norm @ p_norm.T).max(dim=1).values.sum().item()
            scores.append((sim, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_results = scores[:top_k]

        # Serialize results (images → JPEG bytes for transfer)
        results = []
        for score, idx in top_results:
            page = self.all_pages[idx]
            # Convert image to bytes for serialization
            buf = io.BytesIO()
            img = page["image"].copy()
            img.thumbnail((1024, 1024))
            img.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()

            results.append({
                "score": score,
                "doc_name": page["doc_name"],
                "page_num": page["page_num"],
                "modality": page["modality"],
                "text_preview": page["text"][:300],
                "n_tables": len(page["tables"]),
                "n_images": page["n_embedded_images"],
                "image_bytes": img_bytes,
            })

        return results


# ──────────────────────────────────────────────────
# Gradio Web Interface (CPU)
# ──────────────────────────────────────────────────


@app.function(
    image=web_image,
    max_containers=1,
    scaledown_window=60 * 20,  # keep alive 20 min after last request
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from PIL import Image

    def query_handler(query: str, top_k: int):
        """Handle a query: call GPU engine, format results."""
        if not query.strip():
            return "Please enter a query.", [], ""

        # Call the GPU engine remotely
        results = ColPaliEngine().query.remote(query, int(top_k))

        # Build answer with citations
        answer_lines = [f"**Query:** {query}\n", "**Retrieved Sources:**"]
        gallery = []
        source_lines = []

        for rank, r in enumerate(results, 1):
            # Reconstruct image from bytes
            img = Image.open(io.BytesIO(r["image_bytes"]))
            caption = (f"#{rank} | {r['doc_name']} p.{r['page_num']} | "
                       f"score: {r['score']:.2f} | {r['modality']}")
            gallery.append((img, caption))

            # Citation
            answer_lines.append(
                f"\n**#{rank}** [Doc: {r['doc_name']}, Page: {r['page_num']}] "
                f"(score: {r['score']:.2f}, modality: {r['modality']})"
            )
            if r["n_tables"] > 0:
                answer_lines.append(f"  - Contains {r['n_tables']} table(s)")
            if r["n_images"] > 0:
                answer_lines.append(f"  - Contains {r['n_images']} figure(s)/chart(s)")
            if r["text_preview"]:
                answer_lines.append(f"  - Text: _{r['text_preview'][:150]}_...")

            source_lines.append(
                f"| {rank} | {r['doc_name']} | {r['page_num']} | "
                f"{r['score']:.2f} | {r['modality']} | "
                f"{r['n_tables']} table(s), {r['n_images']} figure(s) |"
            )

        answer = "\n".join(answer_lines)
        source_table = (
            "| Rank | Document | Page | Score | Modality | Content |\n"
            "|------|----------|------|-------|----------|----------|\n"
            + "\n".join(source_lines)
        )

        return answer, gallery, source_table

    # Build Gradio UI
    with gr.Blocks(title="ColPali Multi-Modal RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# ColPali Multi-Modal Document RAG\n"
            "Ask questions about **IPCC AR6 Climate Change Reports**. "
            "The system retrieves relevant document pages using vision-based embeddings "
            "(ColQwen2.5 + MaxSim) and generates answers with source citations.\n\n"
            "*DSAI 413 — Assignment 1 | Deployed on Modal with T4 GPU*"
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
                        label="Pages to retrieve",
                    )
                submit_btn = gr.Button("Ask", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Example Queries")
                gr.Examples(
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
                    columns=3, height=500, object_fit="contain",
                )
            with gr.TabItem("Source Attribution"):
                source_output = gr.Markdown(label="Sources")

        submit_btn.click(
            fn=query_handler,
            inputs=[query_box, top_k_slider],
            outputs=[answer_output, gallery_output, source_output],
        )
        query_box.submit(
            fn=query_handler,
            inputs=[query_box, top_k_slider],
            outputs=[answer_output, gallery_output, source_output],
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")

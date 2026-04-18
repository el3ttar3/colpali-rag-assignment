"""
Multi-modal answer generation module.

Takes retrieved page images + query and generates
citation-backed answers using a multimodal LLM.
"""

import io
import base64
from typing import List, Dict, Tuple, Optional

from PIL import Image


SYSTEM_PROMPT = """You are a document analysis assistant. You are given images of document pages
retrieved from a corpus. Answer the user's question accurately using ONLY information visible
in the provided pages.

Rules:
1. Ground every claim in the document content — do not hallucinate.
2. Cite your sources using [Doc: <name>, Page: <number>] format.
3. If the answer involves data from charts/tables, describe the specific numbers.
4. If the pages don't contain enough information, say so explicitly.
5. Be concise but thorough."""


def image_to_base64(img: Image.Image, fmt: str = "JPEG", max_size: int = 1024) -> str:
    """Convert PIL Image to base64, resizing for API efficiency."""
    img_copy = img.copy()
    img_copy.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    img_copy.save(buffer, format=fmt, quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_answer_claude(
    query: str,
    retrieved: List[Tuple[float, int, Dict]],
    api_key: str,
    model_id: str = "claude-sonnet-4-20250514",
) -> str:
    """Generate answer using Anthropic Claude with vision."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content = []

    for rank, (score, idx, page) in enumerate(retrieved, 1):
        b64 = image_to_base64(page["image"])
        content.append({
            "type": "text",
            "text": f"[Page #{rank} — {page['doc_name']}, Page {page['page_num']}, Score: {score:.2f}]"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
        })

    content.append({"type": "text", "text": f"\nQuestion: {query}"})

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def generate_answer_gemini(
    query: str,
    retrieved: List[Tuple[float, int, Dict]],
    api_key: str,
    model_id: str = "gemini-2.0-flash",
) -> str:
    """Generate answer using Google Gemini with vision."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(model_id, system_instruction=SYSTEM_PROMPT)

    parts = []
    for rank, (score, idx, page) in enumerate(retrieved, 1):
        parts.append(f"[Page #{rank} — {page['doc_name']}, Page {page['page_num']}, Score: {score:.2f}]")
        img_copy = page["image"].copy()
        img_copy.thumbnail((1024, 1024), Image.LANCZOS)
        parts.append(img_copy)

    parts.append(f"\nQuestion: {query}")
    response = gmodel.generate_content(parts)
    return response.text


def generate_answer(
    query: str,
    retrieved: List[Tuple[float, int, Dict]],
    provider: str = "none",
    api_key: Optional[str] = None,
) -> str:
    """
    Route to the configured LLM provider.

    Providers: 'claude', 'gemini', 'none' (returns citation-only summary).
    """
    if provider == "claude" and api_key:
        return generate_answer_claude(query, retrieved, api_key)
    elif provider == "gemini" and api_key:
        return generate_answer_gemini(query, retrieved, api_key)
    else:
        return _fallback_summary(query, retrieved)


def _fallback_summary(query: str, retrieved: List[Tuple[float, int, Dict]]) -> str:
    """Generate a citation summary when no LLM API is available."""
    lines = [f"**Query:** {query}\n", "**Retrieved Sources:**"]
    for rank, (score, idx, page) in enumerate(retrieved, 1):
        modality = page.get("modality", "unknown")
        text_preview = page.get("text", "")[:200]
        n_tables = len(page.get("tables", []))
        n_imgs = page.get("n_embedded_images", 0)

        lines.append(f"\n**#{rank}** [Doc: {page['doc_name']}, Page: {page['page_num']}] "
                      f"(score: {score:.2f}, modality: {modality})")
        if n_tables > 0:
            lines.append(f"  - Contains {n_tables} table(s)")
        if n_imgs > 0:
            lines.append(f"  - Contains {n_imgs} embedded image(s)/figure(s)")
        if text_preview:
            lines.append(f"  - Text preview: {text_preview}...")

    lines.append("\n*(Configure an LLM API key for full answer generation)*")
    return "\n".join(lines)

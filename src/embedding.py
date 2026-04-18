"""
ColPali embedding module.

Loads ColQwen2.5 and generates multi-vector embeddings
for document page images and text queries.
"""

from typing import List, Dict, Optional
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm


def detect_device():
    """Auto-detect the best available device and dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


def load_model(model_name: str = "vidore/colqwen2.5-v0.2", device=None, dtype=None):
    """
    Load ColQwen2.5 model and processor.

    Returns (model, processor, device, dtype).
    """
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

    if device is None or dtype is None:
        device, dtype = detect_device()

    print(f"Loading {model_name} on {device} ({dtype})...")
    processor = ColQwen2_5_Processor.from_pretrained(model_name)
    model = ColQwen2_5.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {n_params:.0f}M parameters")
    return model, processor, device, dtype


def embed_pages(
    pages: List[Dict],
    model,
    processor,
    batch_size: int = 2,
) -> List[torch.Tensor]:
    """
    Generate multi-vector embeddings for page images.

    Each page produces a tensor of shape (n_tokens, 128).
    """
    all_embeddings = []

    for i in tqdm(range(0, len(pages), batch_size), desc="Embedding pages"):
        batch_imgs = [p["image"] for p in pages[i : i + batch_size]]
        batch_processed = processor.process_images(batch_imgs).to(model.device)

        with torch.no_grad():
            batch_embs = model(**batch_processed)

        for emb in batch_embs:
            all_embeddings.append(emb.cpu())

    return all_embeddings


def embed_query(query: str, model, processor) -> torch.Tensor:
    """
    Embed a text query into multi-vector representation.

    Returns tensor of shape (n_query_tokens, 128).
    """
    processed = processor.process_queries([query]).to(model.device)
    with torch.no_grad():
        emb = model(**processed)[0].cpu()
    return emb


def save_index(page_embeddings: List[torch.Tensor], pages: List[Dict], path: Path):
    """Persist embeddings and metadata to disk."""
    metadata = [{
        "doc_name": p["doc_name"],
        "page_num": p["page_num"],
        "modality": p.get("modality", "unknown"),
        "text": p.get("text", ""),
    } for p in pages]

    torch.save({"embeddings": page_embeddings, "metadata": metadata}, path)
    print(f"Saved {len(page_embeddings)} embeddings to {path}")


def load_index(path: Path):
    """Load embeddings and metadata from disk."""
    data = torch.load(path, weights_only=False)
    print(f"Loaded {len(data['embeddings'])} embeddings from {path}")
    return data["embeddings"], data["metadata"]

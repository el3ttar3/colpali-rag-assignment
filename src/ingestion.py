"""
Multi-modal document ingestion pipeline.

Handles:
- PDF → page images (for ColPali visual embeddings)
- Text extraction per page (for hybrid search & citation)
- Table detection and extraction
- Embedded image extraction
"""

import io
import requests
from pathlib import Path
from typing import List, Dict, Optional

import pymupdf  # PyMuPDF
from PIL import Image
from tqdm.auto import tqdm


def download_pdf(name: str, url: str, save_dir: Path) -> Path:
    """Download a PDF from a URL if not already cached locally."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{name}.pdf"
    if save_path.exists():
        print(f"  [cached] {name}.pdf")
        return save_path

    print(f"  Downloading {name}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"  Saved {save_path} ({save_path.stat().st_size / 1e6:.1f} MB)")
    return save_path


def pdf_to_page_images(
    pdf_path: Path,
    max_pages: Optional[int] = None,
    dpi: int = 150,
) -> List[Dict]:
    """
    Convert PDF pages to PIL Images using PyMuPDF.

    Returns a list of dicts:
        {image, doc_name, page_num, text, tables, has_images}
    """
    doc = pymupdf.open(str(pdf_path))
    doc_name = pdf_path.stem
    n_pages = min(len(doc), max_pages) if max_pages else len(doc)
    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    pages = []
    for page_idx in tqdm(range(n_pages), desc=f"Ingesting {doc_name}"):
        page = doc[page_idx]

        # --- Image rendering ---
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # --- Text extraction ---
        text = page.get_text("text")

        # --- Table extraction (structural blocks) ---
        tables = _extract_tables(page)

        # --- Detect embedded images ---
        image_list = page.get_images(full=True)

        pages.append({
            "image": img,
            "doc_name": doc_name,
            "page_num": page_idx + 1,
            "text": text.strip(),
            "tables": tables,
            "n_embedded_images": len(image_list),
            "has_images": len(image_list) > 0,
            "modality": _classify_modality(text, tables, image_list),
        })

    doc.close()
    return pages


def _extract_tables(page) -> List[Dict]:
    """
    Extract table-like structures from a PDF page using PyMuPDF's
    block-level layout analysis. Returns a list of table dicts.
    """
    tables = []
    try:
        # PyMuPDF >= 1.23 has built-in table finding
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            extracted = table.extract()
            if extracted and len(extracted) > 1:
                headers = extracted[0]
                rows = extracted[1:]
                tables.append({
                    "headers": headers,
                    "rows": rows,
                    "n_rows": len(rows),
                    "n_cols": len(headers) if headers else 0,
                    "bbox": list(table.bbox),
                })
    except Exception:
        # Fallback: no table extraction support in this PyMuPDF version
        pass
    return tables


def _classify_modality(text: str, tables: List, images: List) -> str:
    """Classify the dominant modality of a page."""
    has_text = len(text.strip()) > 100
    has_tables = len(tables) > 0
    has_images = len(images) > 0

    modalities = []
    if has_text:
        modalities.append("text")
    if has_tables:
        modalities.append("table")
    if has_images:
        modalities.append("figure")

    return "+".join(modalities) if modalities else "empty"


def extract_embedded_images(
    pdf_path: Path,
    max_pages: Optional[int] = None,
) -> List[Dict]:
    """Extract all embedded images from a PDF."""
    doc = pymupdf.open(str(pdf_path))
    n_pages = min(len(doc), max_pages) if max_pages else len(doc)
    extracted = []

    for page_idx in range(n_pages):
        page = doc[page_idx]
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # Skip tiny images (icons, bullets, etc.)
                if img.width >= 50 and img.height >= 50:
                    extracted.append({
                        "image": img,
                        "doc_name": pdf_path.stem,
                        "page_num": page_idx + 1,
                        "img_idx": img_idx,
                        "size": (img.width, img.height),
                    })
            except Exception:
                continue

    doc.close()
    return extracted


def ingest_documents(
    pdf_sources: Dict[str, str],
    pdf_dir: Path,
    max_pages: Optional[int] = None,
    dpi: int = 150,
) -> List[Dict]:
    """
    Full ingestion pipeline: download PDFs and extract all modalities.

    Returns a list of page dicts with image, text, tables, and metadata.
    """
    all_pages = []
    for name, url in pdf_sources.items():
        pdf_path = download_pdf(name, url, pdf_dir)
        pages = pdf_to_page_images(pdf_path, max_pages=max_pages, dpi=dpi)
        all_pages.extend(pages)
        print(f"  {name}: {len(pages)} pages | "
              f"modalities: {set(p['modality'] for p in pages)}")

    print(f"\nTotal: {len(all_pages)} pages ingested")
    return all_pages

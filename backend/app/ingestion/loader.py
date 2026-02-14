from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Dict, List, Literal

from PIL import Image

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore[assignment]

from pypdf import PdfReader

from app.config import settings
from app.errors import TesseractError

logger = logging.getLogger(__name__)

SupportedFileType = Literal["txt", "md", "pdf", "image"]


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read text file %s: %s", path, exc)
        raise


def _read_pdf_file(path: Path) -> str:
    text_parts: List[str] = []
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # best-effort for empty password
            except Exception:
                logger.warning("Encrypted PDF cannot be decrypted: %s", path)
                return ""
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to extract text from page in %s: %s", path, exc)
                page_text = ""
            if page_text:
                text_parts.append(page_text)
    except Exception as exc:
        logger.exception("Failed to read PDF file %s: %s", path, exc)
        return ""
    return "\n".join(text_parts)


def _read_image_file(path: Path) -> str:
    if pytesseract is None:
        logger.warning(
            "pytesseract is not installed; skipping OCR for image file: %s", path
        )
        raise TesseractError("Tesseract not installed")

    def _ocr() -> str:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img) or ""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_ocr)
        try:
            text = future.result(timeout=settings.ocr_timeout_seconds)
            return text
        except FuturesTimeout as exc:
            logger.error(
                "ocr_timeout",
                extra={"path": str(path), "timeout": settings.ocr_timeout_seconds},
            )
            raise TesseractError("OCR timed out") from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to run OCR on image %s: %s", path, exc)
            raise TesseractError("OCR failed") from exc


def detect_file_type(path: Path) -> SupportedFileType | None:
    suffix = path.suffix.lower()
    if suffix in {".txt"}:
        return "txt"
    if suffix in {".md"}:
        return "md"
    if suffix in {".pdf"}:
        return "pdf"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    return None


def load_file(path: Path) -> List[Dict]:
    """
    Load a single file and return a list of normalized document dicts.

    Each document dict has:
      - "text": str
      - "metadata": dict with at least:
          - "source_file": POSIX path
          - "file_type": one of SupportedFileType
    """
    path = path.resolve()
    ftype = detect_file_type(path)
    if ftype is None:
        raise ValueError(f"Unsupported file type for {path}")

    if ftype in {"txt", "md"}:
        text = _read_text_file(path)
    elif ftype == "pdf":
        text = _read_pdf_file(path)
    else:  # image
        text = _read_image_file(path)

    text = text.strip()
    if not text:
        logger.info("No text extracted from %s (type=%s)", path, ftype)
        return []

    metadata = {
        "source_file": path.as_posix(),
        "file_type": ftype,
    }
    return [{"text": text, "metadata": metadata}]


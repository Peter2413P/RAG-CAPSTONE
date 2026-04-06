# loaders/document_loader.py
"""
Document loader supporting PDF, DOCX, TXT, and HTML formats.
Extracts text and metadata from each uploaded document.
"""

import os
from typing import List, Dict, Any

# ── optional heavy deps ──────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (preferred) or pdfplumber."""
    text = ""
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception:
            pass

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text.strip()
        except Exception:
            pass

    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    if not DOCX_AVAILABLE:
        return ""
    try:
        doc = DocxDocument(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)
    except Exception:
        return ""


def extract_text_from_html(file_path: str) -> str:
    """Extract text from HTML using BeautifulSoup."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        if BS4_AVAILABLE:
            soup = BeautifulSoup(raw, "html.parser")
            return soup.get_text(separator="\n").strip()
        # Fallback: strip tags naively
        import re
        clean = re.sub(r"<[^>]+>", " ", raw)
        return clean.strip()
    except Exception:
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from plain text files."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""


def load_document(file_path: str) -> Dict[str, Any]:
    """
    Load a document and return its text and metadata.

    Returns:
        dict with keys: doc_id, filename, category, source, text, char_count
    """
    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
    elif ext in (".html", ".htm"):
        text = extract_text_from_html(file_path)
    elif ext == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        text = extract_text_from_txt(file_path)  # Best-effort for unknown types

    # Infer category from filename
    name_lower = filename.lower()
    if any(k in name_lower for k in ["fee", "tuition", "payment", "scholarship"]):
        category = "Fees & Finance"
    elif any(k in name_lower for k in ["hostel", "accommodation", "dorm"]):
        category = "Hostel"
    elif any(k in name_lower for k in ["exam", "academic", "regulation", "grade"]):
        category = "Academic"
    elif any(k in name_lower for k in ["syllabus", "course", "curriculum"]):
        category = "Syllabus"
    elif any(k in name_lower for k in ["admission", "enroll"]):
        category = "Admissions"
    elif any(k in name_lower for k in ["library", "book"]):
        category = "Library"
    elif any(k in name_lower for k in ["welfare", "health", "counsel", "grievance"]):
        category = "Student Welfare"
    else:
        category = "General"

    return {
        "doc_id": filename,
        "filename": filename,
        "category": category,
        "source": file_path,
        "text": text,
        "char_count": len(text),
    }


def load_documents_from_uploaded(uploaded_files) -> List[Dict[str, Any]]:
    """
    Load multiple documents from Streamlit uploaded file objects.
    Writes to a temp location, loads, then cleans up.
    """
    import tempfile

    documents = []
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            doc = load_document(tmp_path)
            doc["filename"] = uploaded_file.name
            doc["doc_id"] = uploaded_file.name
            documents.append(doc)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return documents


def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Load all supported documents from a folder (for sample dataset)."""
    supported = {".pdf", ".docx", ".txt", ".html", ".htm"}
    documents = []
    for fname in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in supported:
            full_path = os.path.join(folder_path, fname)
            doc = load_document(full_path)
            if doc["text"]:
                documents.append(doc)
    return documents

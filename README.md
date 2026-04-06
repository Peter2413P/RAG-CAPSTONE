# 🎓 University Handbook Search System

A production-quality **Information Retrieval (IR)** system for university handbooks, built with Streamlit, TF-IDF ranking, a LangChain-compatible retriever, and **Phi-3 via Ollama** for RAG-based answer generation.

---

## 📸 Features

| Feature | Description |
|---|---|
| **Document Upload** | PDF, DOCX, TXT, HTML — up to 10 docs |
| **Preprocessing** | Tokenization · Stopword removal · Lemmatization (toggleable) |
| **Inverted Index** | Term → {doc_id: frequency} with TF-IDF scoring |
| **Keyword Search** | Ranked results, no embeddings used |
| **Explainability** | Per-result score breakdown: TF · IDF · TF-IDF per term |
| **LangChain Retriever** | Custom `get_relevant_documents()` — zero embeddings |
| **Search Coach** | Rule-based query advisor (no results / broad / ambiguous) |
| **Evaluation** | Precision · Recall · F1 · MAP on 5 ground-truth queries |
| **Phi-3 RAG** | Grounded answers via Ollama — context-only, no hallucination |
| **Dark UI** | Black + Purple theme with animated cards, badges, metrics |

---

## 🗂️ Project Structure

```
university_handbook_search/
├── app.py                         ← Main Streamlit application
├── requirements.txt
├── README.md
│
├── loaders/
│   └── document_loader.py         ← PDF/DOCX/TXT/HTML text extraction
│
├── preprocessing/
│   └── text_preprocessor.py       ← Tokenize · Stopwords · Lemmatize
│
├── indexing/
│   └── inverted_index.py          ← Inverted index + TF-IDF scoring
│
├── retrieval/
│   └── retriever.py               ← KeywordRetriever + LangChainRetriever
│
├── evaluation/
│   └── evaluator.py               ← Precision / Recall / F1 / MAP
│
├── agent/
│   └── search_coach.py            ← Rule-based query advisor
│
├── rag/
│   └── answer_generator.py        ← Ollama / Phi-3 integration
│
├── utils/
│   └── session_state.py           ← Streamlit session-state manager
│
└── sample_docs/                   ← 7 pre-built university documents
    ├── academic_regulations.txt
    ├── fee_structure.txt
    ├── hostel_rules.txt
    ├── cse_syllabus.txt
    ├── admission_policy.txt
    ├── student_welfare.txt
    └── library_rules.txt
```

---

## ⚙️ Setup Instructions (Windows)

### Step 1 — Prerequisites

- **Python 3.10 or 3.11** (recommended)
  Download from: https://www.python.org/downloads/
  ✅ Check "Add Python to PATH" during installation

- **Git** (optional, for cloning)
  Download from: https://git-scm.com/download/win

---

### Step 2 — Create a Virtual Environment

Open **Command Prompt** or **PowerShell** in the project folder:

```bat
cd university_handbook_search

python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

---

### Step 3 — Install Python Dependencies

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, PyMuPDF, NLTK, pandas, requests, etc.

---

### Step 4 — Install Ollama + Phi-3

#### 4a. Download Ollama for Windows
Go to: https://ollama.com/download/windows
Run the installer.

#### 4b. Start Ollama (keep this terminal open)
```bat
ollama serve
```

#### 4c. Pull Phi-3 model (in a NEW terminal)
```bat
ollama pull phi3
```

> ⚠️ The Phi-3 model is ~2.2 GB. Ensure you have sufficient disk space and RAM (8 GB+ recommended).

Verify it's working:
```bat
ollama run phi3 "Hello, who are you?"
```

---

### Step 5 — Run the Application

With `(venv)` active and Ollama running:

```bat
streamlit run app.py
```

The app will open at: **http://localhost:8501**

---

## 🚀 Quick Start Guide

### 1. Upload Documents
- Go to **📤 Upload & Index** tab
- Click **"Load Sample Dataset"** to load all 7 pre-built university documents
  — or upload your own PDF/DOCX/TXT/HTML files
- Click **"⚡ Build Inverted Index"**
- The index stats (vocab size, doc count) appear in the sidebar

### 2. Search
- Go to **🔍 Search** tab
- Type a query (e.g., `examination attendance grading system`)
- Click **"🔍 Search Documents"**
- Review ranked results with highlighted matched terms
- Check the **Explanation Panel** for TF-IDF score breakdowns
- Check the **Search Coach** for query refinement tips
- Click **"🤖 Generate Answer (Phi-3)"** for a RAG-synthesized answer

### 3. Evaluate
- Go to **📊 Evaluation** tab
- Click **"▶ Run Evaluation"**
- View Precision, Recall, F1, MAP metrics for 5 predefined queries
- Inspect per-query error analysis (TP / FP / FN)

---

## 📐 System Architecture

```
Document Upload
    ↓
Text Extraction (PyMuPDF / python-docx / BeautifulSoup)
    ↓
Preprocessing (Tokenize → Remove Stopwords → Lemmatize)
    ↓
Inverted Index (term → {doc_id: frequency})
    ↓
TF-IDF Scoring & Ranking
    ↓
Explainability (per-term breakdown)
    ↓ ──────────────────────────────────────────────────
    ├── Evaluation Module (Precision / Recall / F1 / MAP)
    ├── LangChain Retriever (get_relevant_documents)
    ├── Agentic Search Coach (rule-based suggestions)
    └── Phi-3 via Ollama (RAG answer generation)
```

---

## 🧩 Module Descriptions

### `loaders/document_loader.py`
Extracts raw text from PDF (PyMuPDF → pdfplumber fallback), DOCX (python-docx), HTML (BeautifulSoup), and TXT files. Infers document category from filename. Handles Streamlit uploaded files via temp file buffering.

### `preprocessing/text_preprocessor.py`
Full NLP pipeline: lowercase tokenization, punctuation/digit removal, stopword filtering (NLTK + custom English stopwords), and WordNet lemmatization. Includes fallback suffix-stripping when NLTK is unavailable.

### `indexing/inverted_index.py`
Builds an in-memory inverted index: `term → {doc_id: raw_frequency}`. Computes TF (raw_count / doc_length), IDF (log-smoothed), and TF-IDF scores. Supports incremental document addition and efficient top-k retrieval.

### `retrieval/retriever.py`
`KeywordRetriever` wraps the inverted index and provides ranked results with explainability data (snippet extraction, matched term highlighting). `LangChainKeywordRetriever` implements the `get_relevant_documents(query)` interface for LangChain compatibility — zero embeddings.

### `evaluation/evaluator.py`
5 predefined evaluation queries with ground-truth relevance labels. Computes Precision@K, Recall@K, F1, and Average Precision. Returns per-query error analysis (TP, FP, FN).

### `agent/search_coach.py`
Rule-based advisor that inspects the query and results and gives actionable feedback: no results → suggest better terms; short query → add context; ambiguous → clarifying question; broad → suggest filters; good results → confirm + refinement tip.

### `rag/answer_generator.py`
Checks Ollama health, builds a grounded RAG prompt (context-only instruction), calls `phi3` model, and returns the answer with source attribution. Handles connection errors, timeouts, and model fallbacks gracefully.

### `utils/session_state.py`
Centralizes all Streamlit session state key names. Prevents key-name collisions and makes state management consistent across all modules.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure `(venv)` is active and run `pip install -r requirements.txt` |
| Ollama offline in sidebar | Run `ollama serve` in a separate terminal |
| Phi-3 not found | Run `ollama pull phi3` |
| NLTK errors | Run `python -c "import nltk; nltk.download('all')"` once |
| Slow answer generation | Normal — Phi-3 takes 30–90s locally without GPU |
| PDF text empty | Some scanned PDFs need OCR (not included). Use text-based PDFs. |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## 📊 Sample Evaluation Results (expected)

With the 7 sample documents loaded:

| Query | Precision@5 | Recall@5 | F1 |
|---|---|---|---|
| Q1: exam attendance grading | ~80% | ~100% | ~89% |
| Q2: hostel fees payment refund | ~80% | ~100% | ~89% |
| Q3: library books digital databases | ~20% | ~100% | ~33% |
| Q4: admission eligibility JEE reservation | ~20% | ~100% | ~33% |
| Q5: counseling mental health grievance | ~20% | ~100% | ~33% |

---

## 🛠️ Technologies Used

- **Streamlit** — UI framework
- **NLTK** — NLP preprocessing
- **PyMuPDF / pdfplumber** — PDF extraction
- **python-docx** — DOCX extraction
- **BeautifulSoup4** — HTML extraction
- **Pandas** — Tabular display
- **Ollama** — Local LLM runtime
- **Phi-3** — Microsoft's small language model
- **Requests** — Ollama HTTP API calls

---

## 📝 Notes

- **No embeddings** are used anywhere in the retrieval pipeline
- The LangChain retriever is custom-implemented — no `langchain` package required
- All processing is **fully local** — no cloud API calls
- Phi-3 answers are grounded to retrieved context only
#   R A G - C A P S T O N E  
 
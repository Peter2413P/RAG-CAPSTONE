# rag/answer_generator.py
"""
Answer generation using Phi-3 via Ollama.
Retrieves top-k documents and generates a grounded answer
using ONLY the retrieved context (no hallucination).
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "phi3"          # matches `ollama pull phi3`
FALLBACK_MODELS = ["phi3:mini", "phi3:medium", "llama3", "mistral"]


# ── Ollama health check ───────────────────────────────────────────────────────

def check_ollama_available() -> Tuple[bool, str]:
    """
    Check whether Ollama is running and Phi-3 (or a fallback) is available.
    Returns (is_available: bool, model_name: str)
    """
    if not REQUESTS_AVAILABLE:
        return False, ""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False, ""
        data = resp.json()
        available_models = [m["name"].split(":")[0] for m in data.get("models", [])]
        raw_names        = [m["name"] for m in data.get("models", [])]

        # Prefer exact phi3 match first
        for candidate in [DEFAULT_MODEL] + FALLBACK_MODELS:
            if candidate in available_models or candidate in raw_names:
                return True, candidate
        # If any model is available use the first one
        if raw_names:
            return True, raw_names[0]
        return False, ""
    except Exception:
        return False, ""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, context_docs: List[Dict[str, Any]]) -> str:
    """
    Build a RAG prompt that instructs the model to answer ONLY from context.
    """
    context_parts = []
    for i, doc in enumerate(context_docs, start=1):
        fname    = doc.get("filename", f"Document {i}")
        category = doc.get("category", "General")
        text     = doc.get("text", "")
        # Truncate long docs to avoid context-window overflow
        max_chars = 1200
        excerpt   = text[:max_chars] + ("..." if len(text) > max_chars else "")
        context_parts.append(
            f"[Document {i}] {fname} (Category: {category})\n{excerpt}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a university handbook assistant. Answer the student's question using ONLY the information provided in the context documents below. Do NOT use any outside knowledge.

If the answer is not found in the context, say: "I could not find this information in the provided documents."

Be concise, clear, and cite which document(s) you used.

=== CONTEXT DOCUMENTS ===

{context_block}

=== STUDENT QUESTION ===
{query}

=== YOUR ANSWER ==="""
    return prompt


# ── Ollama API caller ─────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Tuple[bool, str]:
    """
    Call Ollama /api/generate endpoint.
    Returns (success: bool, response_text: str)
    """
    if not REQUESTS_AVAILABLE:
        return False, "❌ 'requests' library not installed."
    try:
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":   temperature,
                "num_predict":   max_tokens,
                "stop":          ["=== STUDENT QUESTION ===", "=== CONTEXT"],
            },
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("response", "").strip()
            return True, answer
        else:
            return False, f"❌ Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.ConnectionError:
        return False, (
            "❌ Cannot connect to Ollama. Make sure Ollama is running:\n"
            "   Run `ollama serve` in a terminal, then try again."
        )
    except requests.exceptions.Timeout:
        return False, "❌ Ollama request timed out (>120s). Try a smaller model or shorter context."
    except Exception as e:
        return False, f"❌ Unexpected error: {str(e)}"


# ── Main answer generator ─────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Orchestrates RAG: takes retrieved docs + query → calls Phi-3 → returns answer.
    """

    def __init__(self, model: Optional[str] = None):
        available, detected_model = check_ollama_available()
        self.ollama_available = available
        self.model = model or detected_model or DEFAULT_MODEL

    def refresh_status(self) -> Tuple[bool, str]:
        """Re-check Ollama availability (call before each generation)."""
        available, model = check_ollama_available()
        self.ollama_available = available
        if model:
            self.model = model
        return available, self.model

    def generate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 4,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate an answer from retrieved documents.

        Args:
            query:       Original user query
            results:     List of retrieval result dicts
            top_k:       How many docs to include in context
            temperature: Sampling temperature for the model

        Returns:
            {
                "success":   bool,
                "answer":    str,
                "model":     str,
                "sources":   list of filenames used,
                "prompt":    str  (for transparency),
                "error":     str | None,
            }
        """
        available, current_model = self.refresh_status()

        if not available:
            return {
                "success": False,
                "answer":  "",
                "model":   self.model,
                "sources": [],
                "prompt":  "",
                "error": (
                    "Ollama is not running or Phi-3 is not installed.\n\n"
                    "Fix: Open a terminal and run:\n"
                    "  ollama serve\n"
                    "  ollama pull phi3\n"
                    "Then refresh this page."
                ),
            }

        if not results:
            return {
                "success": False,
                "answer":  "",
                "model":   current_model,
                "sources": [],
                "prompt":  "",
                "error":   "No documents were retrieved. Please search first.",
            }

        # Use top-k docs
        context_docs = results[:top_k]
        sources       = [d.get("filename", "") for d in context_docs]

        prompt = build_prompt(query, context_docs)
        success, answer = call_ollama(
            prompt,
            model=current_model,
            temperature=temperature,
            max_tokens=600,
        )

        return {
            "success": success,
            "answer":  answer,
            "model":   current_model,
            "sources": sources,
            "prompt":  prompt,
            "error":   None if success else answer,
        }

# evaluation/evaluator.py
"""
Evaluation module for the IR system.
Computes Precision, Recall, F1, and Average Precision for 5 predefined queries.
Includes error analysis.
"""

from typing import List, Dict, Any, Tuple


# ── Ground-truth relevance labels ─────────────────────────────────────────────
# Format: query_id → { "query": str, "relevant_docs": [list of relevant filenames] }

EVALUATION_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "Q1",
        "query": "examination attendance rules grading system",
        "relevant_docs": [
            "academic_regulations.txt",
            "cse_syllabus.txt",
        ],
        "description": "Tests retrieval of academic/exam rules",
    },
    {
        "query_id": "Q2",
        "query": "hostel fees payment refund policy",
        "relevant_docs": [
            "fee_structure.txt",
            "hostel_rules.txt",
        ],
        "description": "Tests retrieval of fee and hostel docs",
    },
    {
        "query_id": "Q3",
        "query": "library books borrowing digital databases",
        "relevant_docs": [
            "library_rules.txt",
        ],
        "description": "Tests specific library retrieval",
    },
    {
        "query_id": "Q4",
        "query": "admission eligibility JEE reservation quota",
        "relevant_docs": [
            "admission_policy.txt",
        ],
        "description": "Tests admissions doc retrieval",
    },
    {
        "query_id": "Q5",
        "query": "counseling mental health grievance complaint ragging",
        "relevant_docs": [
            "student_welfare.txt",
        ],
        "description": "Tests student welfare retrieval",
    },
]


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Precision@K: fraction of top-k retrieved docs that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@K: fraction of relevant docs found in top-k results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """
    Average Precision (AP): average of precision values at each relevant hit.
    """
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            sum_precision += hits / i
    if hits == 0:
        return 0.0
    return sum_precision / len(relevant)


def run_evaluation(
    retriever,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run evaluation on all 5 predefined queries.

    Args:
        retriever: KeywordRetriever instance
        top_k: Number of results per query

    Returns:
        Dict with per-query results and aggregate metrics
    """
    per_query_results = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_aps = []

    for q in EVALUATION_QUERIES:
        query_text = q["query"]
        relevant_docs = q["relevant_docs"]

        # Retrieve
        try:
            raw_results = retriever.retrieve(query_text, top_k=top_k)
            retrieved_docs = [r["filename"] for r in raw_results]
        except Exception as e:
            retrieved_docs = []

        # Compute metrics
        prec = precision_at_k(retrieved_docs, relevant_docs, top_k)
        rec = recall_at_k(retrieved_docs, relevant_docs, top_k)
        f1 = f1_score(prec, rec)
        ap = average_precision(retrieved_docs, relevant_docs)

        # Error analysis
        true_positives = [d for d in retrieved_docs[:top_k] if d in relevant_docs]
        false_positives = [d for d in retrieved_docs[:top_k] if d not in relevant_docs]
        false_negatives = [d for d in relevant_docs if d not in retrieved_docs[:top_k]]

        error_analysis = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

        result = {
            "query_id": q["query_id"],
            "query": query_text,
            "description": q["description"],
            "relevant_docs": relevant_docs,
            "retrieved_docs": retrieved_docs,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "ap": round(ap, 4),
            "error_analysis": error_analysis,
        }
        per_query_results.append(result)
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1s.append(f1)
        all_aps.append(ap)

    # Aggregate
    n = len(EVALUATION_QUERIES)
    aggregate = {
        "mean_precision": round(sum(all_precisions) / n, 4),
        "mean_recall": round(sum(all_recalls) / n, 4),
        "mean_f1": round(sum(all_f1s) / n, 4),
        "map": round(sum(all_aps) / n, 4),  # Mean Average Precision
    }

    return {
        "per_query": per_query_results,
        "aggregate": aggregate,
        "top_k": top_k,
    }

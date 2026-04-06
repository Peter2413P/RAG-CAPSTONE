# utils/logger.py
"""
JSONL query logger for tracking search queries, results, and coach feedback.
Logs are written to query_log.jsonl in the project root.
"""

import json
import datetime
import pathlib
from typing import List, Dict, Any

LOG_PATH = pathlib.Path("query_log.jsonl")


def log_query(query: str, results: List[Dict[str, Any]], coach_advice: Dict[str, Any]) -> None:
    """
    Log a search query, its results, and coach feedback to JSONL file.
    
    Args:
        query: The user's search query
        results: List of retrieval result dicts
        coach_advice: Dictionary containing coach status and suggestions
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "num_results": len(results),
        "top_results": [r.get("filename", "") for r in results[:3]],
        "coach_status": coach_advice.get("status", "unknown"),
        "coach_message": coach_advice.get("message", ""),
    }
    
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Silently fail to avoid disrupting the UI
        pass

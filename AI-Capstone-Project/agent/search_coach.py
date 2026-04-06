# agent/search_coach.py
"""
Agentic Search Coach — rule-based system that analyzes search results
and provides actionable query improvement suggestions.

Rules:
  1. No results        → suggest richer query terms
  2. Too few results   → broaden query or try synonyms
  3. Too many results  → suggest focused filters
  4. Ambiguous query   → ask clarifying question
  5. Short query       → prompt for more detail
  6. Good results      → affirm and show refinement tip
"""

from typing import List, Dict, Any, Optional
import re
from preprocessing.text_preprocessor import preprocess


# ── Domain knowledge for suggestions ─────────────────────────────────────────

TOPIC_KEYWORDS = {
    "exam": ["examination", "grading", "marks", "semester", "backlog", "supplementary", "attendance"],
    "fees": ["tuition", "payment", "scholarship", "refund", "fine", "late fee", "waiver"],
    "hostel": ["accommodation", "room", "curfew", "warden", "mess", "outpass", "hostel"],
    "library": ["books", "borrowing", "databases", "journal", "overdue", "renewal", "IEEE"],
    "admission": ["eligibility", "JEE", "GATE", "CAT", "reservation", "quota", "counseling"],
    "welfare": ["counseling", "mental health", "grievance", "ragging", "complaint", "scholarship"],
    "syllabus": ["course", "curriculum", "credits", "unit", "subject", "lab", "semester"],
    "placement": ["placement", "internship", "recruiter", "package", "career", "CDC"],
}

CLARIFYING_QUESTIONS = {
    "fee": "Are you asking about tuition fees, hostel fees, exam fees, or late payment fines?",
    "exam": "Are you asking about end-semester examination rules, grading, or supplementary exams?",
    "admission": "Are you asking about undergraduate (B.Tech/B.Sc), postgraduate (M.Tech/MBA), or PhD admissions?",
    "hostel": "Are you asking about hostel room allotment, rules and timings, or hostel fees?",
    "course": "Are you looking for a specific department syllabus (e.g., CSE, ECE) or general course registration rules?",
    "scholarship": "Are you asking about merit scholarships, government scholarships, or sports quota fee waivers?",
    "library": "Are you asking about borrowing rules, digital databases, or library timings?",
    "complaint": "Are you filing a complaint about academic issues, hostel, sexual harassment, or ragging?",
}

AMBIGUOUS_TERMS = [
    "course", "fee", "exam", "hostel", "library", "admission",
    "scholarship", "complaint", "register", "form", "card",
]

NARROW_THRESHOLD = 1  # If retrieved ≤ this many docs, query might be too narrow


class SearchCoach:
    """Rule-based search coach providing query refinement suggestions."""

    def __init__(self):
        pass

    def analyze(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        query_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze query and results, return coaching advice.

        Args:
            query: User's search query
            results: List of retrieved result dicts
            top_k: The top-k used for retrieval (to determine if results are broad)
            query_tokens: Optional pre-processed query tokens

        Returns:
            {
                "status": str,          # "no_results" | "few_results" | "broad" | "ambiguous" | "good"
                "message": str,         # Main coaching message
                "suggestions": list,    # List of actionable suggestions
                "clarifying_question": str | None,
                "suggested_queries": list,
            }
        """
        num_results = len(results)
        query_words = query.strip().split()
        query_lower = query.lower()

        # ── Rule 1: No results at all ────────────────────────────────────────
        if num_results == 0:
            return self._no_results_advice(query, query_lower, query_words)

        # ── Rule 2: Very short query (1 word) ────────────────────────────────
        if len(query_words) == 1:
            return self._short_query_advice(query, query_lower)

        # ── Rule 3: Ambiguous query ───────────────────────────────────────────
        clarification = self._detect_ambiguity(query_lower, query_words, num_results)
        if clarification:
            return clarification

        # ── Rule 4: Too many results (broad query) ────────────────────────────
        if num_results >= top_k:
            return self._broad_query_advice(query, query_lower, results)

        # ── Rule 5: Few results ───────────────────────────────────────────────
        if num_results <= NARROW_THRESHOLD:
            return self._few_results_advice(query, query_lower, results)

        # ── Rule 6: Good results ──────────────────────────────────────────────
        return self._good_results_advice(query, results)

    # ── Rule Handlers ─────────────────────────────────────────────────────────

    def _no_results_advice(self, query: str, query_lower: str, words: List[str]) -> Dict:
        suggestions = [
            "Check spelling of query terms",
            "Use more common keywords (e.g., 'exam' instead of 'examination procedure')",
            "Try a shorter, focused query with 2–3 key terms",
        ]
        suggested_queries = self._build_suggested_queries(query_lower)
        if not suggested_queries:
            suggested_queries = [
                "examination grading attendance",
                "fees payment scholarship",
                "hostel rules curfew",
                "library books borrowing",
                "admission eligibility documents",
            ]
        return {
            "status": "no_results",
            "message": "⚠️ No documents found for your query.",
            "suggestions": suggestions,
            "clarifying_question": None,
            "suggested_queries": suggested_queries[:3],
        }

    def _short_query_advice(self, query: str, query_lower: str) -> Dict:
        suggestions = [
            f"Add more context to '{query}' — try combining it with related terms",
            "Specify a topic area: academic, hostel, fees, library, admissions",
        ]
        suggested_queries = self._build_suggested_queries(query_lower)
        return {
            "status": "short_query",
            "message": f"🔍 Your query '{query}' is very short. Adding more terms improves accuracy.",
            "suggestions": suggestions,
            "clarifying_question": self._get_clarifying_question(query_lower),
            "suggested_queries": suggested_queries[:3],
        }

    def _detect_ambiguity(
        self,
        query_lower: str,
        words: List[str],
        num_results: int,
    ) -> Optional[Dict]:
        for amb_term in AMBIGUOUS_TERMS:
            preprocessed = preprocess(query_lower)
            if (amb_term in words or amb_term in preprocessed) and len(words) <= 2:
                cq = self._get_clarifying_question(query_lower)
                if cq:
                    return {
                        "status": "ambiguous",
                        "message": f"🤔 '{amb_term}' can mean different things. Could you clarify?",
                        "suggestions": [
                            "Specify the context (e.g., academic fee vs hostel fee)",
                            "Combine with another term for a more precise search",
                        ],
                        "clarifying_question": cq,
                        "suggested_queries": self._build_suggested_queries(query_lower)[:3],
                    }
        return None

    def _broad_query_advice(
        self,
        query: str,
        query_lower: str,
        results: List[Dict],
    ) -> Dict:
        categories = list({r.get("category", "") for r in results if r.get("category")})
        filter_suggestions = [f"Filter by category: {cat}" for cat in categories[:3]]
        suggestions = [
            "Your query returned many results — try adding a specific keyword",
            *filter_suggestions,
        ]
        focused = self._make_focused_queries(query_lower)
        return {
            "status": "broad",
            "message": f"📋 Your query returned {len(results)} results. Consider narrowing it down.",
            "suggestions": suggestions,
            "clarifying_question": None,
            "suggested_queries": focused[:3],
        }

    def _few_results_advice(
        self,
        query: str,
        query_lower: str,
        results: List[Dict],
    ) -> Dict:
        synonyms = self._get_synonyms(query_lower)
        suggestions = [
            "Try synonyms or related terms",
            "Remove highly specific terms that may not appear in documents",
            "Check if the topic falls under a different category",
        ]
        if synonyms:
            suggestions.append(f"Synonyms to try: {', '.join(synonyms)}")
        return {
            "status": "few_results",
            "message": f"🔎 Only {len(results)} result(s) found. You might get better coverage with a broader query.",
            "suggestions": suggestions,
            "clarifying_question": None,
            "suggested_queries": self._build_suggested_queries(query_lower)[:3],
        }

    def _good_results_advice(self, query: str, results: List[Dict]) -> Dict:
        top_doc = results[0].get("filename", "") if results else ""
        categories = list({r.get("category", "") for r in results})
        return {
            "status": "good",
            "message": f"✅ Good results found! Top match: {top_doc}",
            "suggestions": [
                f"Results span categories: {', '.join(categories)}",
                "Use the Explanation Panel below for score breakdown details",
                "Try the Generated Answer section for a synthesized response",
            ],
            "clarifying_question": None,
            "suggested_queries": [],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_clarifying_question(self, query_lower: str) -> Optional[str]:
        for key, question in CLARIFYING_QUESTIONS.items():
            if key in query_lower:
                return question
        return None

    def _build_suggested_queries(self, query_lower: str) -> List[str]:
        suggestions = []
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(k in query_lower for k in keywords) or topic in query_lower:
                if topic == "exam":
                    suggestions.append("examination rules grading attendance")
                elif topic == "fees":
                    suggestions.append("tuition fee payment scholarship refund")
                elif topic == "hostel":
                    suggestions.append("hostel accommodation rules curfew warden")
                elif topic == "library":
                    suggestions.append("library books borrowing database renewal")
                elif topic == "admission":
                    suggestions.append("admission eligibility JEE reservation documents")
                elif topic == "welfare":
                    suggestions.append("student welfare counseling grievance ragging")
                elif topic == "syllabus":
                    suggestions.append("course syllabus credits curriculum semester")
                elif topic == "placement":
                    suggestions.append("placement internship package career CDC")
        return list(dict.fromkeys(suggestions))  # Deduplicate preserving order

    def _make_focused_queries(self, query_lower: str) -> List[str]:
        return [
            query_lower + " exam rules",
            query_lower + " fees payment",
            query_lower + " hostel accommodation",
        ]

    def _get_synonyms(self, query_lower: str) -> List[str]:
        synonym_map = {
            "exam": ["test", "assessment", "evaluation"],
            "fee": ["charge", "payment", "tuition", "cost"],
            "hostel": ["accommodation", "dormitory", "residence"],
            "library": ["books", "resources", "reading room"],
            "grade": ["marks", "score", "gpa", "cgpa"],
            "admission": ["enrollment", "joining", "intake"],
            "scholarship": ["financial aid", "waiver", "stipend"],
        }
        synonyms = []
        for word, syns in synonym_map.items():
            if word in query_lower:
                synonyms.extend(syns)
        return synonyms[:4]

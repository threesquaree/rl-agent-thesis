"""
Fact selection for ExplainNewFact.

Selection policy:
- candidates are only from new_facts
- if at least two facts score >= t_high: pick top1 + top2
- otherwise: pick top1
- optional force_top1_below_low keeps one fact even when score < t_low
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "at",
    "for", "and", "or", "but", "with", "as", "by", "from", "that", "this", "it",
    "be", "do", "does", "did", "can", "could", "would", "should", "i", "you", "we",
    "they", "he", "she", "what", "how", "why", "when", "where", "who", "which",
    "tell", "me", "about", "more",
}


@dataclass
class ScoredFact:
    fact_id: str
    fact_text: str
    fact_raw: str
    score: float


@dataclass
class SelectionResult:
    selected_facts: List[ScoredFact]
    top_score: float
    second_score: float
    no_fact_added_due_to_low_confidence: int
    new_facts_empty: int
    selection_mode: str


def _extract_fact_id(fact_with_id: str) -> str:
    m = re.search(r"\[([A-Z]{2}_\d{3})\]", fact_with_id or "")
    return m.group(1) if m else ""


def _strip_fact_id(fact_with_id: str) -> str:
    return re.sub(r"^\s*\[[A-Z]{2}_\d{3}\]\s*", "", fact_with_id or "").strip()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _score(user_text: str, fact_text: str) -> float:
    u_tokens = set(_tokenize(user_text))
    f_tokens = set(_tokenize(fact_text))
    if not u_tokens or not f_tokens:
        overlap = 0.0
    else:
        overlap = len(u_tokens & f_tokens) / max(1, len(u_tokens))
    seq = SequenceMatcher(None, (user_text or "").lower(), (fact_text or "").lower()).ratio()
    # weighted blend, deterministic and cheap
    return 0.7 * overlap + 0.3 * seq


def select_candidate_facts(
    user_text: str,
    candidate_facts: List[str],
    *,
    t_low: float = 0.18,
    t_high: float = 0.35,
    allow_top2_high: bool = False,
    force_top1_below_low: bool = False,
) -> SelectionResult:
    if not candidate_facts:
        return SelectionResult([], 0.0, 0.0, 0, 1, "empty_no_fact")

    scored: List[ScoredFact] = []
    for fact in candidate_facts:
        fid = _extract_fact_id(fact)
        txt = _strip_fact_id(fact)
        if not fid or not txt:
            continue
        scored.append(
            ScoredFact(
                fact_id=fid,
                fact_text=txt,
                fact_raw=fact,
                score=_score(user_text, txt),
            )
        )

    if not scored:
        return SelectionResult([], 0.0, 0.0, 1, 0, "empty_no_fact")

    scored.sort(key=lambda x: x.score, reverse=True)
    top_score = scored[0].score
    second_score = scored[1].score if len(scored) > 1 else 0.0

    if allow_top2_high and len(scored) > 1 and top_score >= t_high and second_score >= t_high:
        return SelectionResult(scored[:2], top_score, second_score, 0, 0, "high_two")

    if top_score >= t_low:
        return SelectionResult(scored[:1], top_score, second_score, 0, 0, "single_top1")

    if force_top1_below_low:
        return SelectionResult(scored[:1], top_score, second_score, 0, 0, "single_top1_low")

    return SelectionResult([], top_score, second_score, 1, 0, "low_no_fact")


def select_new_facts(
    user_text: str,
    new_facts: List[str],
    *,
    t_low: float = 0.18,
    t_high: float = 0.35,
    allow_top2_high: bool = False,
    force_top1_below_low: bool = False,
) -> SelectionResult:
    """Backward-compatible wrapper for new-fact selection."""
    return select_candidate_facts(
        user_text=user_text,
        candidate_facts=new_facts,
        t_low=t_low,
        t_high=t_high,
        allow_top2_high=allow_top2_high,
        force_top1_below_low=force_top1_below_low,
    )

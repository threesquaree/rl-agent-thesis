"""
Prompt builder for RepeatFact with AOI-summary handling.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from prompt.fact_selector import SelectionResult, select_candidate_facts


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _is_aoi_summary_fact(fact_text: str) -> bool:
    stripped = (fact_text or "").strip().lower()
    return stripped.startswith("aoi details -") or stripped.startswith("aoi details —")


def _detect_question_type(user_text: str, aois: List[Dict[str, str]]) -> Tuple[str, str]:
    user_lower = (user_text or "").lower()
    user_tokens = set(_normalize_tokens(user_text))
    best_name = ""
    best_score = 0.0

    for aoi in aois or []:
        name = str(aoi.get("name", "")).strip()
        desc = str(aoi.get("description", "")).strip()
        if not name:
            continue
        name_lower = name.lower()
        score = 0.0
        if name_lower in user_lower:
            score = 1.0
        else:
            name_tokens = set(_normalize_tokens(name))
            desc_tokens = set(_normalize_tokens(desc))
            token_overlap = 0.0
            if user_tokens and name_tokens:
                token_overlap = max(token_overlap, len(user_tokens & name_tokens) / max(1, len(name_tokens)))
            if user_tokens and desc_tokens:
                token_overlap = max(token_overlap, len(user_tokens & desc_tokens) / max(1, min(5, len(desc_tokens))))
            seq = SequenceMatcher(None, user_lower, f"{name_lower} {desc.lower()}".strip()).ratio()
            score = max(token_overlap, seq * 0.5)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name and best_score >= 0.2:
        return "aoi_targeted", best_name
    return "general", ""


def build_repeat_fact_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    facts_used: List[str],
    user_text: str,
    current_completion: float = 0.0,
    t_low: float = 0.18,
    t_high: float = 0.35,
    auxiliary_context: Optional[dict] = None,
) -> Tuple[str, Dict[str, object]]:
    aois = list((auxiliary_context or {}).get("aois", []) or [])
    question_type, matched_aoi_name = _detect_question_type(user_text, aois)

    selection: SelectionResult = select_candidate_facts(
        user_text=user_text,
        candidate_facts=facts_used or [],
        t_low=t_low,
        t_high=t_high,
        allow_top2_high=False,
        force_top1_below_low=False,
    )

    selected_fact_ids = [f.fact_id for f in selection.selected_facts]
    selected = selection.selected_facts[0] if selection.selected_facts else None
    selected_fact_is_aoi_summary = 1 if (selected and _is_aoi_summary_fact(selected.fact_text)) else 0

    if selection.new_facts_empty == 1:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. No previously shared fact is available to repeat.
2. Answer the user's question directly.
3. Do not include any [FACT_ID].
4. Maximum two sentences.

Response:"""
        selection_mode = "empty_no_fact"
    elif question_type == "aoi_targeted" and selected_fact_is_aoi_summary == 1 and selected is not None:
        aoi_line = f' "{matched_aoi_name}"' if matched_aoi_name else ""
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. Sentence 1: answer the user's current AOI-related question.
2. Sentence 2: restate only the part of the selected AOI-summary fact that is most relevant to AOI{aoi_line}.
3. Do not restate the entire AOI-summary fact.
4. Use the exact [FACT_ID] for the selected repeated fact.
5. Do not add any new facts or new IDs.
6. Do not quote the visitor verbatim.
7. Make the two sentences connect naturally.
8. Maximum two sentences.

Selected repeated fact:
- [{selected.fact_id}] {selected.fact_text}

Response:"""
        selection_mode = "aoi_repeat"
    elif question_type == "aoi_targeted":
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. The user's question is AOI-specific, but no suitable previously shared AOI-summary fact is available to repeat.
2. Answer the user's question directly.
3. Do not force a repeated fact in this turn.
4. Do not include any [FACT_ID].
5. Maximum two sentences.

Response:"""
        selection_mode = "aoi_answer_only"
    elif selection.no_fact_added_due_to_low_confidence == 1 or selected is None:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. Relevance confidence is low (top_score={selection.top_score:.3f} < t_low={t_low}).
2. Answer the user's question directly.
3. Do not force a repeated fact in this turn.
4. Do not include any [FACT_ID].
5. Maximum two sentences.

Response:"""
        selection_mode = "low_no_fact"
    else:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. Sentence 1: answer the user's current question.
2. Sentence 2: restate the single most relevant previously shared fact in fresh words.
3. Use the exact [FACT_ID] for the selected repeated fact.
4. Do not add any new facts or new IDs.
5. Do not quote the visitor verbatim.
6. Make the two sentences connect naturally.
7. Maximum two sentences.

Selected repeated fact:
- [{selected.fact_id}] {selected.fact_text}

Response:"""
        selection_mode = "general_repeat"

    meta = {
        "selected_fact_ids": selected_fact_ids,
        "top_score": selection.top_score,
        "selection_mode": selection_mode,
        "question_type": question_type,
        "selected_fact_is_aoi_summary": selected_fact_is_aoi_summary,
        "matched_aoi_name": matched_aoi_name,
        "facts_used_empty": selection.new_facts_empty,
    }
    return prompt, meta

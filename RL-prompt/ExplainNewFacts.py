"""
Prompt builders for explain strategies.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from prompt.fact_selector import SelectionResult, select_new_facts


def _extract_used_ids(facts_used: List[str]) -> List[str]:
    ids = set()
    for fact in facts_used or []:
        m = re.search(r"\[([A-Z]{2}_\d{3})\]", fact)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def build_explain_new_fact_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    new_facts: List[str],
    facts_used: List[str],
    user_text: str,
    current_completion: float = 0.0,
    t_low: float = 0.18,
    t_high: float = 0.35,
    allow_top2_high: bool = False,
) -> (str, Dict[str, object]):
    """
    Build ExplainNewFact prompt using the agreed policy.
    Returns prompt text + selection metadata for logging.
    """
    used_ids = _extract_used_ids(facts_used)
    selection: SelectionResult = select_new_facts(
        user_text=user_text,
        new_facts=new_facts,
        t_low=t_low,
        t_high=t_high,
        allow_top2_high=allow_top2_high,
        force_top1_below_low=True,
    )

    selected_fact_ids = [f.fact_id for f in selection.selected_facts]
    selected_fact_lines = "\n".join(
        [f"- [{f.fact_id}] {f.fact_text}" for f in selection.selected_facts]
    )
    used_warning = ", ".join(used_ids) if used_ids else "None"

    if selection.new_facts_empty == 1:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ExplainNewFact policy for this turn:
1) First, directly answer the user's question in a natural way.
2) No new facts are available now. Do not force a new fact.
3) You lightly suggest moving to another exhibit.
4) Do not output any [FACT_ID].
5) Do not quote the visitor verbatim.
6) Maximum two sentences.

Response:"""
    elif selection.selection_mode == "high_two":
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ExplainNewFact policy for this turn:
1) First, answer the user's question naturally.
2) Then use the two selected facts only: top1 first, then top2.
3) Make the two sentences connect naturally.
4) Include the exact [FACT_ID] for each selected fact you use.
5) Never repeat previously used IDs: {used_warning}.
6) Never invent IDs.
7) Never use facts outside the selected list.
8) Do not quote the visitor verbatim.
9) Maximum two sentences.

Selected facts (use only these):
{selected_fact_lines}

Response:"""
    else:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ExplainNewFact policy for this turn:
1) Sentence 1: briefly answer the user's question.
2) Sentence 2: add the single most relevant selected fact.
3) The second sentence must connect naturally to the first.
4) Use only the selected fact.
5) Include its exact [FACT_ID].
6) Never repeat previously used IDs: {used_warning}.
7) Never invent IDs.
8) Never use facts outside the selected list.
9) Do not quote the visitor verbatim.
10) Maximum two sentences.

Selected facts (use only these):
{selected_fact_lines}

Response:"""

    meta = {
        "selected_fact_ids": selected_fact_ids,
        "top_score": selection.top_score,
        "second_score": selection.second_score,
        "no_fact_added_due_to_low_confidence": selection.no_fact_added_due_to_low_confidence,
        "new_facts_empty": selection.new_facts_empty,
        "selection_mode": selection.selection_mode,
    }
    return prompt, meta

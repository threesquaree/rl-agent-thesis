"""
Prompt builder for RepeatFact using relevance-based selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from prompt.fact_selector import SelectionResult, select_candidate_facts


def build_repeat_fact_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    facts_used: List[str],
    user_text: str,
    current_completion: float = 0.0,
    t_low: float = 0.18,
    t_high: float = 0.35,
) -> Tuple[str, Dict[str, object]]:
    """
    Build RepeatFact prompt with the same selector mechanism as ExplainNewFact.
    """
    selection: SelectionResult = select_candidate_facts(
        user_text=user_text,
        candidate_facts=facts_used or [],
        t_low=t_low,
        t_high=t_high,
        allow_top2_high=False,
    )

    selected_fact_ids = [f.fact_id for f in selection.selected_facts]
    selected = selection.selected_facts[0] if selection.selected_facts else None

    if selection.new_facts_empty == 1:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. No previously shared fact is available to repeat.
2. Answer the user's question directly without adding any [FACT_ID].
3. Do not invent fact IDs.
4. Maximum two sentences.

Response:"""
    elif selection.no_fact_added_due_to_low_confidence == 1 or selected is None:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. Relevance confidence is low (top_score={selection.top_score:.3f} < t_low={t_low}).
2. Do not force repeating a fact in this turn.
3. Answer the user's question directly.
4. Do not include any [FACT_ID].
5. Maximum two sentences.

Response:"""
    else:
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

RepeatFact policy:
1. First answer the user's question naturally.
2. Then restate the selected previously-shared fact in fresh words.
3. Use the exact fact ID [{selected.fact_id}] for that repeated fact.
4. Do not add any new facts or new IDs.
5. Do not quote the visitor verbatim.
6. Maximum two sentences.

Selected repeated fact:
- [{selected.fact_id}] {selected.fact_text}

Response:"""

    meta = {
        "selected_fact_ids": selected_fact_ids,
        "top_score": selection.top_score,
        "no_fact_added_due_to_low_confidence": selection.no_fact_added_due_to_low_confidence,
        "facts_used_empty": selection.new_facts_empty,
    }
    return prompt, meta


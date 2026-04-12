"""
Prompt builder for OfferTransition/SummarizeAndSuggest strategy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple


def build_summarize_and_suggest_prompt(
    *,
    current_exhibit: Optional[str],
    target_exhibit: Optional[str],
    context_section: str,
) -> Tuple[str, Dict[str, object]]:
    current_name = (current_exhibit or "current exhibit").replace("_", " ")

    if target_exhibit:
        target_name = target_exhibit.replace("_", " ")
        prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {current_name}
---

{context_section}

SummarizeAndSuggest policy:
1. Keep exactly two sentences.
2. Sentence 1: briefly summarize what has been discussed so far.
3. Sentence 2: suggest moving to "{target_name}" and name it explicitly.
4. This is only a suggestion; do not imply the visitor already moved.
5. Do not introduce new facts about the target exhibit.
6. Do not include any [FACT_ID].

Response:"""
        meta = {
            "transition_target": target_exhibit,
            "transition_has_target": 1,
            "transition_mode": "summarize_and_suggest",
        }
        return prompt, meta

    prompt = f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {current_name}
---

{context_section}

SummarizeAndSuggest policy:
1. Keep exactly two sentences.
2. Sentence 1: briefly summarize what has been discussed so far.
3. Sentence 2: provide a neutral suggestion to choose any other exhibit.
4. This is only a suggestion; do not imply the visitor already moved.
5. Do not include any [FACT_ID].

Response:"""
    meta = {
        "transition_target": None,
        "transition_has_target": 0,
        "transition_mode": "fallback",
    }
    return prompt, meta

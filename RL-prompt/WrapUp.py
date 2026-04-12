"""
Prompt builder for conclude/wrap-up strategy.
"""

from __future__ import annotations

from typing import Optional


def build_wrap_up_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    current_completion: float = 0.0,
) -> str:
    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

WrapUp policy:
1. Keep exactly two sentences.
2. If the user's latest input contains a clear question:
   - Sentence 1: briefly answer that question directly.
   - Sentence 2: politely close the visit (thanks + warm ending).
3. If there is no clear question:
   - Sentence 1: briefly summarize 1-2 key points already discussed.
   - Sentence 2: politely close the visit (thanks + warm ending).
4. Do not introduce new facts.
5. Do not include any [FACT_ID].
6. Do not ask a new question.
7. Do not mention system state (coverage, policy, strategy, completion).

Response:"""

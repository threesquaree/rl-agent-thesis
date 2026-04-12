"""
Prompt builder for AskOpinion.
"""

from __future__ import annotations

from typing import Optional


def build_ask_opinion_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    current_completion: float = 0.0,
) -> str:
    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskOpinion policy:
1. First, respond naturally to the user's current input without quoting them verbatim.
2. Then ask exactly one opinion question.
3. The question must include at least one keyword from the current exhibit or current AOI.
4. The question must be opinion/feeling-oriented.
5. Do not introduce new facts.
6. Do not include any [FACT_ID].
7. Maximum two sentences.

Response:"""


"""
Prompt builders for ask-question strategies.
"""

from __future__ import annotations

from typing import Optional


def build_ask_clarification_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    current_completion: float = 0.0,
) -> str:
    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskClarification policy:
1. Follow this priority order: Answer > Summary > Clarify question.
2. Only ask a clarification question if ambiguity is genuine.
3. Do not include any [FACT_ID].
4. Do not quote the visitor verbatim.
5. Maximum two sentences.

Response:"""


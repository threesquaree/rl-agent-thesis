"""
Prompt builder for ClarifyFact.
"""

from __future__ import annotations

import re
from typing import List, Optional


def _pick_fact_to_clarify(facts_used: List[str], selected_fact: Optional[str]) -> str:
    if selected_fact and str(selected_fact).strip():
        return selected_fact
    return facts_used[-1] if facts_used else ""


def _strip_fact_id(fact_with_id: str) -> str:
    return re.sub(r'^\s*\[[A-Z]{2}_\d{3}\]\s*', '', fact_with_id or '').strip()


def build_clarify_fact_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    facts_used: List[str],
    selected_fact: Optional[str] = None,
    current_completion: float = 0.0,
) -> str:
    if facts_used:
        fact_to_clarify = _pick_fact_to_clarify(facts_used, selected_fact)
        fact_plain = _strip_fact_id(fact_to_clarify)
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ClarifyFact policy:
1. First, respond naturally to the user's question or confusion.
2. Clarify this previously discussed fact in simpler words or with an everyday analogy:
   "{fact_plain}"
3. Do not introduce new facts.
4. Do not include any [FACT_ID].
5. Do not quote the visitor verbatim.
6. Maximum two sentences.

Response:"""

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

ClarifyFact policy:
1. No previously discussed fact is available to clarify.
2. Answer the user's question directly in simple language.
3. Do not include any [FACT_ID].
4. Maximum two sentences.

Response:"""


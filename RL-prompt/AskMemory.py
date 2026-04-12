"""
Prompt builder for AskMemory.
"""

from __future__ import annotations

from typing import List, Optional


def build_ask_memory_prompt(
    *,
    ex_id: Optional[str],
    context_section: str,
    facts_used: List[str],
    current_completion: float = 0.0,
) -> str:
    if facts_used:
        recent_facts = facts_used[-3:]
        facts_block = "\n".join([f"- {f}" for f in recent_facts])
        return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskMemory policy:
1. First respond naturally to the user's current input.
2. Then ask one short memory-check question.
3. Memory target priority:
   - Highest priority: recall from facts already used in this exhibit.
   - If no usable fact, recall from recently mentioned points in conversation history.
4. Do not include [FACT_ID].
5. Maximum two sentences.

Recent used facts (priority memory source):
{facts_block}

Response:"""

    return f"""[CONTEXT - DO NOT REPEAT]
Museum guide at: {ex_id} | Progress: {current_completion:.1%} covered
---

{context_section}

AskMemory policy:
1. First respond naturally to the user's current input.
2. Then ask one short memory-check question.
3. Memory target: recently mentioned points in conversation history.
4. Do not include [FACT_ID].
5. Maximum two sentences.

Response:"""

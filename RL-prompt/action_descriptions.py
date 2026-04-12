"""
Action description registry for RL prompt guidance.

This file intentionally tracks the 8 subactions in the current H3 checkpoint:
Explain: ExplainNewFact, RepeatFact, ClarifyFact
AskQuestion: AskOpinion, AskMemory, AskClarification
OfferTransition: SummarizeAndSuggest
Conclude: WrapUp
"""

from typing import Dict


ACTION_DESCRIPTIONS: Dict[str, str] = {
    "ExplainNewFact": "Explain: ExplainNewFact",
    "RepeatFact": "Explain: RepeatFact",
    "ClarifyFact": "Explain: ClarifyFact",
    "AskOpinion": "AskQuestion: AskOpinion",
    "AskMemory": "AskQuestion: AskMemory",
    "AskClarification": "AskQuestion: AskClarification",
    "SummarizeAndSuggest": "OfferTransition: SummarizeAndSuggest",
    "WrapUp": "Conclude: WrapUp",
}


def get_action_description(subaction: str) -> str:
    sub = (subaction or "").strip()
    if not sub:
        return ""
    return ACTION_DESCRIPTIONS.get(sub, sub)

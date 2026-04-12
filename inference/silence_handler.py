"""
Silence Handler for Inference-Time Visitor Silence Detection

Detects prolonged visitor silence during live VR museum tours and returns
a rule-based action override. See docs/superpowers/specs/2026-04-12-silence-handler-design.md.
"""

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class ConversationSnapshot:
    last_agent_option: str
    last_agent_subaction: str
    facts_mentioned_count: int
    total_facts_at_exhibit: int


class SilenceHandler:
    def __init__(self, threshold_sec: float = 40.0, max_triggers: int = 2):
        self.threshold_sec = threshold_sec
        self.max_triggers = max_triggers
        self._last_visitor_time: Optional[float] = None
        self._triggers_used: int = 0

    def check(self, current_time: float, snapshot: ConversationSnapshot) -> Optional[Dict[str, str]]:
        """Returns action dict {"option": ..., "subaction": ...} if silence triggered, else None."""
        if self._last_visitor_time is None:
            return None
        if self._triggers_used >= self.max_triggers:
            return None
        elapsed = current_time - self._last_visitor_time
        if elapsed < self.threshold_sec:
            return None
        return self._select_action(snapshot)

    def _select_action(self, snapshot: ConversationSnapshot) -> Dict[str, str]:
        """State-dependent rule table. First matching rule wins."""
        return {"option": "AskQuestion", "subaction": "AskOpinion"}  # Placeholder — Task 3

    def notify_visitor_spoke(self, current_time: float) -> None:
        """Resets silence tracking when visitor provides input."""
        self._last_visitor_time = current_time
        self._triggers_used = 0

    def notify_action_taken(self) -> None:
        """Increments trigger counter after a silence-triggered action is executed."""
        self._triggers_used += 1

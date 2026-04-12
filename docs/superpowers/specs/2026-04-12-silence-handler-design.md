# Silence Handler — Inference-Time Rule-Based Visitor Silence Detection

**Date:** 2026-04-12
**Scope:** Inference pipeline only (no training changes)
**Simulator:** Hybrid (training); VR deployment (inference)

## Problem

During live VR museum tours, a visitor may go silent (e.g., 40+ seconds without speaking). The RL agent currently has no mechanism to detect this at inference time and proactively re-engage.

## Solution

A standalone `SilenceHandler` module that intercepts the inference tick, detects prolonged visitor silence, and returns a rule-based action override — bypassing the trained model for that turn.

## Design Decisions

- **Rule-based, not learned**: The agent was not trained on silence events. A hardcoded state-dependent rule table selects the action.
- **Capped at 2 triggers**: After 2 consecutive silence-triggered actions without visitor response, the handler stops and waits for the visitor to speak.
- **Prompt generation still runs**: The overridden action goes through `dialogue_planner.build_prompt()` + LLM as normal — only the action selection is bypassed.
- **No changes to env.py, training, or model weights.**

## Components

### 1. `inference/silence_handler.py`

Single class with three methods:

```python
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class ConversationSnapshot:
    last_agent_option: str          # e.g., "Explain", "AskQuestion"
    last_agent_subaction: str       # e.g., "ExplainNewFact", "AskOpinion"
    facts_mentioned_count: int      # total facts shared at current exhibit
    total_facts_at_exhibit: int     # total available facts at current exhibit

class SilenceHandler:
    def __init__(self, threshold_sec: float = 40.0, max_triggers: int = 2):
        self.threshold_sec = threshold_sec
        self.max_triggers = max_triggers
        self._last_visitor_time: Optional[float] = None
        self._triggers_used: int = 0

    def check(self, current_time: float, snapshot: ConversationSnapshot) -> Optional[Dict[str, str]]:
        """Returns action dict if silence triggered, else None."""

    def notify_visitor_spoke(self, current_time: float) -> None:
        """Resets silence tracking on visitor input."""

    def notify_action_taken(self) -> None:
        """Increments trigger counter after a silence-triggered action."""
```

### 2. Action Selection Rules

Evaluated in order inside `check()`. The first matching rule wins.

| Priority | Condition | Option | Subaction | Rationale |
|----------|-----------|--------|-----------|-----------|
| 1 | `trigger_number == 2` | AskQuestion | AskClarification | Second silence — gentle, least intrusive check-in |
| 2 | `last_agent_option == "Explain"` | AskQuestion | AskOpinion | Was lecturing — flip to question to re-engage |
| 3 | `last_agent_option == "AskQuestion"` | Explain | ExplainNewFact | Already asked and got silence — try offering content (only if facts available) |
| 4 | `facts_mentioned_count == 0` | Explain | ExplainNewFact | Nothing shared yet — start the conversation |
| 5 | `facts_mentioned >= total_facts` | OfferTransition | SummarizeAndSuggest | Exhibit exhausted — suggest moving on |
| 6 | Fallback | AskQuestion | AskOpinion | Safe default |

Rule 3 falls through to rule 6 if no new facts are available (`facts_mentioned_count >= total_facts_at_exhibit`).

### 3. Integration Point

The handler wraps the existing inference tick. One integration site in the VR inference loop:

```python
silence_handler = SilenceHandler(threshold_sec=40.0, max_triggers=2)

def on_visitor_input(message, timestamp):
    silence_handler.notify_visitor_spoke(timestamp)
    # ... normal model inference via get_agent_response()

def on_tick(timestamp, conversation_snapshot):
    override = silence_handler.check(timestamp, conversation_snapshot)
    if override:
        silence_handler.notify_action_taken()
        # override = {"option": "AskQuestion", "subaction": "AskOpinion"}
        # Skip model, go straight to build_prompt() + LLM generation
        return generate_utterance(override)
    return None  # No silence — normal model inference
```

## What Does NOT Change

- `src/environment/env.py` — no training-time changes
- `src/simulator/hybrid_simulator.py` — no simulator changes
- Trained model weights — not retrained
- `src/utils/dialogue_planner.py` — prompts are reused as-is for silence-triggered actions

## File Changes

| File | Change |
|------|--------|
| `inference/silence_handler.py` | New file — SilenceHandler class + ConversationSnapshot dataclass |
| `inference/test_model.py` | Add silence check wrapper around `get_agent_response()` |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold_sec` | 40.0 | Seconds of silence before first trigger |
| `max_triggers` | 2 | Max consecutive silence actions before stopping |

Both configurable via constructor arguments. No environment variables needed (inference-only feature).

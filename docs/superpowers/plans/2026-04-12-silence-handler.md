# Silence Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add inference-time visitor silence detection that picks a state-dependent action when the visitor is silent for 40+ seconds, capped at 2 triggers per silence window.

**Architecture:** A standalone `SilenceHandler` class in `inference/silence_handler.py` intercepts the inference tick. When silence exceeds the threshold, it returns a rule-based action override (bypassing the model). The override still flows through `dialogue_planner.build_prompt()` + LLM for utterance generation. No training code changes.

**Tech Stack:** Python stdlib only (dataclasses, typing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-12-silence-handler-design.md`

---

## File Structure

| File | Role |
|------|------|
| `inference/silence_handler.py` | New — `ConversationSnapshot` dataclass + `SilenceHandler` class with 6-rule action selection |
| `inference/test_silence_handler.py` | New — unit tests for silence handler |
| `inference/test_model.py` | Modify — add `get_agent_response_with_silence()` wrapper |

---

### Task 1: ConversationSnapshot dataclass + SilenceHandler skeleton

**Files:**
- Create: `inference/test_silence_handler.py`
- Create: `inference/silence_handler.py`

- [ ] **Step 1: Write test for ConversationSnapshot creation**

```python
# inference/test_silence_handler.py
"""Tests for inference/silence_handler.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from silence_handler import ConversationSnapshot, SilenceHandler


def test_conversation_snapshot_creation():
    snap = ConversationSnapshot(
        last_agent_option="Explain",
        last_agent_subaction="ExplainNewFact",
        facts_mentioned_count=3,
        total_facts_at_exhibit=8,
    )
    assert snap.last_agent_option == "Explain"
    assert snap.facts_mentioned_count == 3


def test_silence_handler_defaults():
    handler = SilenceHandler()
    assert handler.threshold_sec == 40.0
    assert handler.max_triggers == 2


def test_silence_handler_custom_params():
    handler = SilenceHandler(threshold_sec=30.0, max_triggers=1)
    assert handler.threshold_sec == 30.0
    assert handler.max_triggers == 1


if __name__ == "__main__":
    test_conversation_snapshot_creation()
    test_silence_handler_defaults()
    test_silence_handler_custom_params()
    print("All Task 1 tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python inference/test_silence_handler.py`
Expected: `ModuleNotFoundError: No module named 'silence_handler'`

- [ ] **Step 3: Implement ConversationSnapshot and SilenceHandler skeleton**

```python
# inference/silence_handler.py
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
        return None  # Placeholder — implemented in Task 2

    def notify_visitor_spoke(self, current_time: float) -> None:
        """Resets silence tracking when visitor provides input."""
        self._last_visitor_time = current_time
        self._triggers_used = 0

    def notify_action_taken(self) -> None:
        """Increments trigger counter after a silence-triggered action is executed."""
        self._triggers_used += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python inference/test_silence_handler.py`
Expected: `All Task 1 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add inference/silence_handler.py inference/test_silence_handler.py
git commit -m "feat: add SilenceHandler skeleton and ConversationSnapshot dataclass"
```

---

### Task 2: Silence detection logic (check + notify)

**Files:**
- Modify: `inference/test_silence_handler.py`
- Modify: `inference/silence_handler.py`

- [ ] **Step 1: Write tests for silence detection timing**

Append to `inference/test_silence_handler.py`:

```python
def _make_snapshot(**overrides):
    defaults = dict(
        last_agent_option="Explain",
        last_agent_subaction="ExplainNewFact",
        facts_mentioned_count=3,
        total_facts_at_exhibit=8,
    )
    defaults.update(overrides)
    return ConversationSnapshot(**defaults)


def test_no_trigger_before_threshold():
    handler = SilenceHandler(threshold_sec=40.0)
    handler.notify_visitor_spoke(100.0)
    result = handler.check(130.0, _make_snapshot())  # 30s < 40s
    assert result is None


def test_trigger_at_threshold():
    handler = SilenceHandler(threshold_sec=40.0)
    handler.notify_visitor_spoke(100.0)
    result = handler.check(141.0, _make_snapshot())  # 41s > 40s
    assert result is not None
    assert "option" in result
    assert "subaction" in result


def test_no_trigger_before_first_input():
    handler = SilenceHandler(threshold_sec=40.0)
    # Never called notify_visitor_spoke — _last_visitor_time is None
    result = handler.check(999.0, _make_snapshot())
    assert result is None


def test_cap_at_max_triggers():
    handler = SilenceHandler(threshold_sec=40.0, max_triggers=2)
    handler.notify_visitor_spoke(100.0)

    # First trigger
    result1 = handler.check(141.0, _make_snapshot())
    assert result1 is not None
    handler.notify_action_taken()

    # Second trigger
    result2 = handler.check(182.0, _make_snapshot())
    assert result2 is not None
    handler.notify_action_taken()

    # Third — should be capped
    result3 = handler.check(223.0, _make_snapshot())
    assert result3 is None


def test_reset_on_visitor_spoke():
    handler = SilenceHandler(threshold_sec=40.0, max_triggers=2)
    handler.notify_visitor_spoke(100.0)

    # Trigger twice
    handler.check(141.0, _make_snapshot())
    handler.notify_action_taken()
    handler.check(182.0, _make_snapshot())
    handler.notify_action_taken()

    # Visitor speaks — resets
    handler.notify_visitor_spoke(200.0)
    result = handler.check(241.0, _make_snapshot())
    assert result is not None  # Triggers again after reset
```

Add these calls to the `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    test_conversation_snapshot_creation()
    test_silence_handler_defaults()
    test_silence_handler_custom_params()
    test_no_trigger_before_threshold()
    test_trigger_at_threshold()
    test_no_trigger_before_first_input()
    test_cap_at_max_triggers()
    test_reset_on_visitor_spoke()
    print("All Task 1+2 tests passed.")
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python inference/test_silence_handler.py`
Expected: `test_trigger_at_threshold` fails — `check()` always returns `None`

- [ ] **Step 3: Implement check() detection logic**

Replace the `check` method in `inference/silence_handler.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python inference/test_silence_handler.py`
Expected: `All Task 2 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add inference/silence_handler.py inference/test_silence_handler.py
git commit -m "feat: implement silence detection timing logic in SilenceHandler.check()"
```

---

### Task 3: State-dependent action selection rules

**Files:**
- Modify: `inference/test_silence_handler.py`
- Modify: `inference/silence_handler.py`

- [ ] **Step 1: Write tests for all 6 rules**

Append to `inference/test_silence_handler.py`:

```python
def _trigger(handler, snapshot, spoke_at=100.0, check_at=141.0):
    """Helper: set up timing so silence triggers, return the action."""
    handler.notify_visitor_spoke(spoke_at)
    return handler.check(check_at, snapshot)


def test_rule1_second_trigger_returns_ask_clarification():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(last_agent_option="Explain")

    # First trigger
    _trigger(handler, snap)
    handler.notify_action_taken()

    # Second trigger — rule 1
    result = handler.check(182.0, snap)
    assert result == {"option": "AskQuestion", "subaction": "AskClarification"}


def test_rule2_after_explain_returns_ask_opinion():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(last_agent_option="Explain", last_agent_subaction="ExplainNewFact")
    result = _trigger(handler, snap)
    assert result == {"option": "AskQuestion", "subaction": "AskOpinion"}


def test_rule3_after_ask_question_returns_explain():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(
        last_agent_option="AskQuestion",
        last_agent_subaction="AskOpinion",
        facts_mentioned_count=3,
        total_facts_at_exhibit=8,
    )
    result = _trigger(handler, snap)
    assert result == {"option": "Explain", "subaction": "ExplainNewFact"}


def test_rule3_fallthrough_when_exhausted():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(
        last_agent_option="AskQuestion",
        facts_mentioned_count=8,
        total_facts_at_exhibit=8,
    )
    # All facts mentioned — rule 3 can't fire, falls to rule 5 (exhausted)
    result = _trigger(handler, snap)
    assert result == {"option": "OfferTransition", "subaction": "SummarizeAndSuggest"}


def test_rule4_no_facts_yet():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(
        last_agent_option="OfferTransition",
        facts_mentioned_count=0,
        total_facts_at_exhibit=8,
    )
    result = _trigger(handler, snap)
    assert result == {"option": "Explain", "subaction": "ExplainNewFact"}


def test_rule5_exhibit_exhausted():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(
        last_agent_option="Explain",
        facts_mentioned_count=8,
        total_facts_at_exhibit=8,
    )
    # last_agent_option is Explain but exhibit is exhausted
    # Rule 2 would fire (AskOpinion) because it checks before rule 5.
    # This is correct — asking opinion at an exhausted exhibit is fine.
    result = _trigger(handler, snap)
    assert result == {"option": "AskQuestion", "subaction": "AskOpinion"}


def test_rule6_fallback():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(
        last_agent_option="Conclude",
        last_agent_subaction="WrapUp",
        facts_mentioned_count=3,
        total_facts_at_exhibit=8,
    )
    result = _trigger(handler, snap)
    assert result == {"option": "AskQuestion", "subaction": "AskOpinion"}
```

Add these calls to the `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    test_conversation_snapshot_creation()
    test_silence_handler_defaults()
    test_silence_handler_custom_params()
    test_no_trigger_before_threshold()
    test_trigger_at_threshold()
    test_no_trigger_before_first_input()
    test_cap_at_max_triggers()
    test_reset_on_visitor_spoke()
    test_rule1_second_trigger_returns_ask_clarification()
    test_rule2_after_explain_returns_ask_opinion()
    test_rule3_after_ask_question_returns_explain()
    test_rule3_fallthrough_when_exhausted()
    test_rule4_no_facts_yet()
    test_rule5_exhibit_exhausted()
    test_rule6_fallback()
    print("All Task 3 tests passed.")
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python inference/test_silence_handler.py`
Expected: `test_rule1_second_trigger_returns_ask_clarification` fails — `_select_action()` always returns AskOpinion

- [ ] **Step 3: Implement _select_action() with 6 rules**

Replace the `_select_action` method in `inference/silence_handler.py`:

```python
    def _select_action(self, snapshot: ConversationSnapshot) -> Dict[str, str]:
        """State-dependent rule table. First matching rule wins."""
        trigger_number = self._triggers_used + 1
        has_new_facts = snapshot.facts_mentioned_count < snapshot.total_facts_at_exhibit
        exhibit_exhausted = snapshot.facts_mentioned_count >= snapshot.total_facts_at_exhibit

        # Rule 1: Second silence trigger — gentle check-in
        if trigger_number >= 2:
            return {"option": "AskQuestion", "subaction": "AskClarification"}

        # Rule 2: Was lecturing — flip to question
        if snapshot.last_agent_option == "Explain":
            return {"option": "AskQuestion", "subaction": "AskOpinion"}

        # Rule 3: Already asked, got silence — try offering content
        if snapshot.last_agent_option == "AskQuestion" and has_new_facts:
            return {"option": "Explain", "subaction": "ExplainNewFact"}

        # Rule 4: Nothing shared yet — start the conversation
        if snapshot.facts_mentioned_count == 0 and has_new_facts:
            return {"option": "Explain", "subaction": "ExplainNewFact"}

        # Rule 5: Exhibit exhausted — suggest moving on
        if exhibit_exhausted:
            return {"option": "OfferTransition", "subaction": "SummarizeAndSuggest"}

        # Rule 6: Fallback
        return {"option": "AskQuestion", "subaction": "AskOpinion"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python inference/test_silence_handler.py`
Expected: `All Task 3 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add inference/silence_handler.py inference/test_silence_handler.py
git commit -m "feat: implement 6-rule state-dependent action selection for silence handler"
```

---

### Task 4: Integrate into inference pipeline

**Files:**
- Modify: `inference/test_model.py` (add `get_agent_response_with_silence()`)

- [ ] **Step 1: Write the wrapper function**

Add to the end of `inference/test_model.py` (before the final newline), importing at the top:

Add this import after the existing imports (after line 22):
```python
from silence_handler import SilenceHandler, ConversationSnapshot
```

Add this function at the end of the file (after `test_conversation`):

```python
def get_agent_response_with_silence(
    agent,
    user_message: Optional[str],
    exhibit: str,
    dialogue_history: List[Tuple[str, str, int]],
    knowledge_graph: SimpleKnowledgeGraph,
    options: List[str],
    subactions: Dict[str, List[str]],
    facts_mentioned: Dict[str, set],
    option_counts: Dict[str, int],
    turn_number: int,
    projection_matrix: np.ndarray,
    bert_recognizer,
    silence_handler: SilenceHandler,
    current_time: float,
    last_agent_option: str = "",
    last_agent_subaction: str = "",
    state_dim: Optional[int] = None
) -> Dict[str, Any]:
    """
    Wrapper around get_agent_response that checks for visitor silence first.

    If user_message is not None, notifies the silence handler and runs normal inference.
    If user_message is None, checks for silence override.

    Returns the same dict as get_agent_response, plus 'silence_triggered': bool.
    """
    if user_message is not None:
        silence_handler.notify_visitor_spoke(current_time)
        result = get_agent_response(
            agent=agent,
            user_message=user_message,
            exhibit=exhibit,
            dialogue_history=dialogue_history,
            knowledge_graph=knowledge_graph,
            options=options,
            subactions=subactions,
            facts_mentioned=facts_mentioned,
            option_counts=option_counts,
            turn_number=turn_number,
            projection_matrix=projection_matrix,
            bert_recognizer=bert_recognizer,
            state_dim=state_dim,
        )
        result['silence_triggered'] = False
        return result

    # No visitor input — check silence
    exhibit_facts = knowledge_graph.get_exhibit_facts(exhibit)
    mentioned_ids = facts_mentioned.get(exhibit, set())
    snapshot = ConversationSnapshot(
        last_agent_option=last_agent_option,
        last_agent_subaction=last_agent_subaction,
        facts_mentioned_count=len(mentioned_ids),
        total_facts_at_exhibit=len(exhibit_facts),
    )

    override = silence_handler.check(current_time, snapshot)
    if override is None:
        return {'action': '', 'option': '', 'subaction': '', 'state_vector': None,
                'action_dict': {}, 'silence_triggered': False}

    silence_handler.notify_action_taken()
    return {
        'action': f"{override['option']}/{override['subaction']}",
        'option': override['option'],
        'subaction': override['subaction'],
        'state_vector': None,
        'action_dict': override,
        'silence_triggered': True,
    }
```

- [ ] **Step 2: Verify the module imports correctly**

Run: `python -c "import sys; sys.path.insert(0, 'inference'); from test_model import get_agent_response_with_silence; print('Import OK')"`
Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add inference/test_model.py
git commit -m "feat: add get_agent_response_with_silence() wrapper to inference pipeline"
```

---

### Task 5: Run full test suite and final commit

**Files:**
- All files from Tasks 1-4

- [ ] **Step 1: Run all silence handler tests**

Run: `python inference/test_silence_handler.py`
Expected: `All Task 3 tests passed.`

- [ ] **Step 2: Verify test_model.py still imports cleanly**

Run: `python -c "import sys; sys.path.insert(0, 'inference'); from test_model import load_trained_model, get_agent_response, get_agent_response_with_silence; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify silence_handler.py has no syntax issues**

Run: `python -c "import ast; ast.parse(open('inference/silence_handler.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 4: Final commit with all files**

```bash
git add inference/silence_handler.py inference/test_silence_handler.py inference/test_model.py
git commit -m "feat: complete silence handler — detection, rules, inference integration"
```

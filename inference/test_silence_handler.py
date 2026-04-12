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
    result = handler.check(999.0, _make_snapshot())
    assert result is None


def test_cap_at_max_triggers():
    handler = SilenceHandler(threshold_sec=40.0, max_triggers=2)
    handler.notify_visitor_spoke(100.0)
    result1 = handler.check(141.0, _make_snapshot())
    assert result1 is not None
    handler.notify_action_taken()
    result2 = handler.check(182.0, _make_snapshot())
    assert result2 is not None
    handler.notify_action_taken()
    result3 = handler.check(223.0, _make_snapshot())
    assert result3 is None


def test_reset_on_visitor_spoke():
    handler = SilenceHandler(threshold_sec=40.0, max_triggers=2)
    handler.notify_visitor_spoke(100.0)
    handler.check(141.0, _make_snapshot())
    handler.notify_action_taken()
    handler.check(182.0, _make_snapshot())
    handler.notify_action_taken()
    handler.notify_visitor_spoke(200.0)
    result = handler.check(241.0, _make_snapshot())
    assert result is not None


def _trigger(handler, snapshot, spoke_at=100.0, check_at=141.0):
    """Helper: set up timing so silence triggers, return the action."""
    handler.notify_visitor_spoke(spoke_at)
    return handler.check(check_at, snapshot)


def test_rule1_second_trigger_returns_ask_clarification():
    handler = SilenceHandler(threshold_sec=40.0)
    snap = _make_snapshot(last_agent_option="Explain")
    _trigger(handler, snap)
    handler.notify_action_taken()
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

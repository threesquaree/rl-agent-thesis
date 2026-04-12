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

# Silence Handler — Integration Guide

This document explains how to integrate visitor silence detection into your VR inference pipeline.

## What It Does

When a visitor is silent for 40+ seconds, the system picks an agent action using a simple rule table (bypassing the RL model). This re-engages the visitor with a contextually appropriate response. After 2 consecutive silence-triggered actions without visitor input, it stops and waits.

## Action Vocabulary

The silence handler and the RL model share the same action vocabulary:

| Option | Subaction | Description |
|--------|-----------|-------------|
| Explain | ExplainNewFact | Share a new fact about the current exhibit |
| Explain | RepeatFact | Restate a previously shared fact |
| Explain | ClarifyFact | Simplify a previously shared fact |
| AskQuestion | AskOpinion | Ask the visitor's opinion |
| AskQuestion | AskMemory | Quiz the visitor on something discussed |
| AskQuestion | AskClarification | Ask what the visitor is interested in |
| OfferTransition | SummarizeAndSuggest | Summarize and suggest moving to another exhibit |
| Conclude | WrapUp | End the tour |

The silence handler only uses: **AskOpinion**, **AskClarification**, **ExplainNewFact**, and **SummarizeAndSuggest**.

## The Rule Table

When silence is detected, evaluate these rules **in order**. Use the first match.

| # | Condition | Action |
|---|-----------|--------|
| 1 | This is the 2nd consecutive silence trigger | AskQuestion / AskClarification |
| 2 | Last agent action was Explain (any subaction) | AskQuestion / AskOpinion |
| 3 | Last agent action was AskQuestion AND exhibit has unshared facts | Explain / ExplainNewFact |
| 4 | Zero facts have been shared AND exhibit has facts | Explain / ExplainNewFact |
| 5 | All exhibit facts have been shared | OfferTransition / SummarizeAndSuggest |
| 6 | None of the above | AskQuestion / AskOpinion |

## State You Need to Track

To evaluate the rules, you need 4 values at each tick:

```
last_agent_option:       string  — the Option of the last action the agent took (e.g., "Explain")
last_agent_subaction:    string  — the Subaction (e.g., "ExplainNewFact")
facts_mentioned_count:   int     — how many facts have been shared at the current exhibit
total_facts_at_exhibit:  int     — total available facts for the current exhibit
```

Plus the silence tracking state:

```
last_visitor_spoke_time: float   — timestamp when the visitor last said something
triggers_used:           int     — how many silence actions have fired since the visitor last spoke
```

## Integration Pseudocode (any language)

```
THRESHOLD = 40.0  // seconds
MAX_TRIGGERS = 2

// State
last_visitor_spoke_time = null
triggers_used = 0

function on_visitor_speaks(timestamp):
    last_visitor_spoke_time = timestamp
    triggers_used = 0
    // ... proceed with normal RL model inference ...

function on_tick(timestamp):
    if last_visitor_spoke_time == null:
        return  // no interaction yet

    if triggers_used >= MAX_TRIGGERS:
        return  // capped, wait for visitor

    elapsed = timestamp - last_visitor_spoke_time
    if elapsed < THRESHOLD:
        return  // not silent long enough

    // Silence detected — pick action from rule table
    action = select_action_from_rules(triggers_used, last_agent_option, facts_mentioned, total_facts)
    triggers_used += 1

    // Feed the selected action into your prompt builder + LLM
    // (same pipeline as model-selected actions, just skip the model step)
    prompt = build_prompt(action.option, action.subaction, ...)
    utterance = llm.generate(prompt)
    agent_speaks(utterance)
```

## Pipeline Flow

```
Normal:   visitor speaks → RL model selects action → build prompt → LLM generates utterance → agent speaks
Silence:  40s silence    → rule table selects action → build prompt → LLM generates utterance → agent speaks
```

The only difference is **who picks the action** — the model or the rule table. Everything downstream (prompt building, LLM generation, speaking) is identical.

## Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Threshold | 40 seconds | Time before first trigger |
| Max triggers | 2 | After 2 silence actions, stops until visitor speaks |

## Python Reference Implementation

See `inference/silence_handler.py` for a ready-to-use Python implementation (~75 lines). It can be used directly if your VR pipeline calls Python, or ported to C#/C++ using the pseudocode above.

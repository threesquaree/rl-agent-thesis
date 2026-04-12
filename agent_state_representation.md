# Agent State Representation & Decision-Making

## What Does the Agent Actually Consider?

The agent makes decisions from a **153-d state vector** (147-d base + 6-d optional response type). It is heavily weighted toward **structured museum state**, not raw visitor speech.

---

## State Vector Breakdown

| Component | Dims | What It Encodes | Role in Decisions |
|---|---|---|---|
| **Focus snapshot** | 6-d | One-hot of which exhibit the visitor is at | "Where am I?" |
| **History vector** | 9-d | Per-exhibit completion ratios (0-1) + normalized option usage counts | "What have I covered? What actions have I been using?" |
| **Subaction availability** | 4-d | Binary flags: ExplainNewFact, ClarifyFact, RepeatFact available + exhibit exhausted | "What can I still do here?" |
| **Intent embedding** | 64-d | DialogueBERT projection of the latest user utterance | Compressed language signal |
| **Dialogue context** | 64-d | DialogueBERT projection of last 3 turns | Compressed conversation flow |
| **Response type** (optional) | 6-d | One-hot: acknowledgment, follow_up, question, statement, confusion, silence | Categorical reaction label |

> Defined in `src/environment/env.py` lines 970-1072

---

## Primary Decision Drivers (Structured Signals)

1. **Exhibit exhaustion** (`subaction_availability[3]`) -- direct binary flag: "there's nothing left to explain here." Strongest transition trigger.
2. **Completion ratios** (`history[0:5]`) -- per-exhibit coverage fractions. Agent sees which exhibits are untouched vs. fully covered.
3. **Option usage distribution** (`history[5:9]`) -- normalized counts of Explain / AskQuestion / OfferTransition / Conclude usage. Helps avoid action spam.

## Visitor Speech (Indirect Signal)

The visitor's actual words are compressed into **128 dimensions** of DialogueBERT embeddings (64-d intent + 64-d context). These encode semantic meaning -- "Can we move on?" produces a different embedding than "That's fascinating!" -- but it's a **lossy, learned representation**, not an explicit "visitor wants to transition" flag.

---

## Visitor-Initiated Transitions

**Visitors never directly trigger transitions.** Transitions are always agent-initiated via the `OfferTransition` option.

### How visitors signal readiness (simulator side)

The state machine (`state_machine_simulator.py`) transitions the visitor to `READY_TO_MOVE` when:
- Current exhibit coverage >= 80% AND 4+ turns spent there

The visitor then says things like: *"What's next?"*, *"Can we move on?"*, *"Should we keep going?"*

These are **hints, not commands**.

### Escalation if agent ignores signals

```
READY_TO_MOVE  -->  (agent does Explain)   -->  FATIGUED
READY_TO_MOVE  -->  (3 turns ignored)      -->  DISENGAGED
```

### Transition success probability (when agent does OfferTransition)

| Facts covered at current exhibit | Success probability |
|---|---|
| 0 facts | 20% |
| 1 fact | 50% |
| 2 facts | 80% |
| 3+ facts | 95% |

> Even if the visitor says "Can we move on?", if coverage is low, the transition can fail.

---

## Practical Implication

The agent learns a policy like:

> *"Exhibit exhausted + high coverage + been explaining for a while --> OfferTransition"*

rather than:

> *"The visitor said 'let's move on' --> OfferTransition"*

Structured features are more reliable learning signals than decoding natural language intent from compressed embeddings.

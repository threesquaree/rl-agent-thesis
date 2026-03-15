# Literature Review — Thesis Directions & Relevant Papers

## Overview

This document covers relevant literature across the three thesis pillars:

1. **Pillar 2 (Main):** User/visitor simulator design and fidelity — how simulator quality shapes RL policy learning
2. **Pillar 1 (Supporting):** Reward shaping theory; novelty/curiosity as intrinsic reward; reward engineering for dialogue RL
3. **Pillar 3 (Exploratory):** Gaze and eye-tracking features as engagement signals; multimodal cognitive load estimation
4. **Contextual:** Actor-Critic methods for dialogue; engagement-adaptive conversational agents; museum guide systems

---

## Pillar 2 — Simulator Design & Fidelity

### 1. How to Build User Simulators to Train RL-based Dialog Systems
**Authors:** Kreyssig et al.
**Venue:** EMNLP 2019
**Source:** https://arxiv.org/abs/1909.01388

The most directly relevant paper to the main thesis contribution. Systematically evaluates different simulator design decisions (transition structure, noise, state representation) and their downstream impact on RL dialogue policy quality. Directly supports the "simulator improvement ladder" experimental design (SM-v1 → SM-v4).

---

### 2. A User Simulator for Task-Completion Dialogues
**Authors:** Li et al.
**Venue:** 2016
**Source:** https://arxiv.org/abs/1612.05688

Foundational work introducing agenda-based user simulators for task-oriented dialogue RL. The conceptual ancestor of the State Machine simulator used in this thesis. Essential background for the methodology chapter — establishes the agenda/state-machine paradigm that this work extends and improves.

---

### 3. Adversarial Learning of Neural User Simulators for Dialogue Policy Optimisation
**Authors:** Shi et al.
**Venue:** 2023
**Source:** https://arxiv.org/abs/2306.00858

Demonstrates empirically that simulator realism directly causes measurable policy gains (8.3% higher task success rate when using a more realistic simulator). Strong evidence for the central thesis claim that simulator fidelity determines policy quality, not merely training speed.

---

## Pillar 1 — Reward Shaping

### 4. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping
**Authors:** Ng, Harada & Russell
**Venue:** ICML 1999
**Source:** https://www.semanticscholar.org/paper/Policy-Invariance-Under-Reward-Transformations:-and-Ng-Harada/94066dc12fe31e96af7557838159bde598cb4f10

The foundational paper establishing potential-based reward shaping and the policy invariance theorem. Provides the theoretical backbone for the centred engagement reward (Eq. 3) and its policy-invariance property — the key theoretical contribution of Pillar 1.

---

### 5. Curiosity-driven Exploration by Self-supervised Prediction
**Authors:** Pathak, Agrawal, Efros & Darrell
**Venue:** ICML 2017
**Source:** https://arxiv.org/abs/1705.05363

The seminal Intrinsic Curiosity Module (ICM) paper. Formalises novelty as an intrinsic reward signal — the conceptual basis for the broadened novelty reward (Eq. 5) that penalises stale/repetitive action sequences (the ExplainNewFact dominance pathology).

---

### 6. Reward Shaping with Recurrent Neural Networks for Speeding up On-Line Policy Learning in Spoken Dialogue Systems
**Authors:** Weisz et al.
**Venue:** 2015
**Source:** https://arxiv.org/abs/1508.03391

Directly applies reward shaping to spoken dialogue systems to address the sparse reward problem. Close parallel to Pillar 1 — applying reward shaping in a dialogue RL setting. Useful for situating the contribution in prior dialogue-specific reward engineering work.

---

## Pillar 3 — Gaze Features & Engagement

### 7. Using Fixed and Mobile Eye Tracking to Understand How Visitors View Art in a Museum
**Venue:** 2025
**Source:** https://arxiv.org/abs/2504.19881

A direct hit for the museum context: studies real museum visitor gaze patterns (fixation duration, saccade patterns) in a physical gallery. Directly motivates the real-world validity of dwell time and gaze metrics as engagement signals, and supports the multi-feature gaze comparison study (Pillar 3).

---

### 8. CLARE: Cognitive Load Assessment in REaltime with Multimodal Data
**Venue:** 2024
**Source:** https://arxiv.org/abs/2404.17098

Demonstrates that gaze features — pupil dilation, fixation duration, saccade rate — are strong and discriminative indicators of cognitive load and engagement state. Directly supports the thesis's use of gaze entropy and saccade span as engagement signals beyond dwell time alone (Pillar 3 motivation).

---

## Contextual — Actor-Critic & Engagement-Adaptive Dialogue

### 9. Sample-efficient Actor-Critic Reinforcement Learning with Supervised Data for Dialogue Management
**Authors:** Weisz et al.
**Venue:** 2017
**Source:** https://arxiv.org/abs/1707.00130

Introduces Actor-Critic methods for dialogue management with trust region and natural gradient techniques. Methodological grounding for the flat A2C architecture used in the codebase inherited from Daniel's thesis.

---

### 10. Continuous Learning Conversational AI: A Personalized Agent Framework via A2C Reinforcement Learning
**Venue:** arXiv 2025
**Source:** https://arxiv.org/abs/2502.12876

Uses Advantage Actor-Critic to optimise dialogue actions via engagement and value delivery metrics in a continuous, session-level learning setup. A close parallel to this thesis's setup — useful for framing and positioning the contribution in current work.

---

### 11. AURA: A Reinforcement Learning Framework for AI-Driven Adaptive Conversational Surveys
**Venue:** arXiv 2025
**Source:** https://arxiv.org/abs/2510.27126

Frames within-session engagement adaptation as an RL problem with a multi-dimensional quality reward, using an ε-greedy policy updated during each session. Directly relevant to framing RQ1 and RQ2 — demonstrates that engagement-adaptive RL dialogue is an active and well-motivated research area.

---

## Museum Context

### 12. A Conversational Agent as Museum Guide — Design and Evaluation of a Real-World Application
**Authors:** Kopp et al.
**Venue:** Lecture Notes in Computer Science
**Source:** https://www.semanticscholar.org/paper/A-Conversational-Agent-as-Museum-Guide-Design-and-a-Kopp-Alvincz/f4e98f391ef1c24577c5cda317eae0d3f69a4a35

The classic real-world deployment of a museum guide agent (the "Max" system). Establishes face-to-face museum dialogue as a research problem and situates this thesis in an established tradition. Essential background context for the introduction and related work chapters.

---

## Coverage Summary

| Pillar | Papers |
|---|---|
| Pillar 2 — Simulator Fidelity (Main) | #1, #2, #3 |
| Pillar 1 — Reward Shaping (Supporting) | #4, #5, #6 |
| Pillar 3 — Gaze & Engagement (Exploratory) | #7, #8 |
| Actor-Critic & Engagement-Adaptive Dialogue | #9, #10, #11 |
| Museum Context | #12 |

---

## Notes

- Papers **#4** (Ng et al.) and **#5** (Pathak et al.) are foundational classics — cite in theory/background sections.
- Papers **#1** and **#3** are the strongest empirical supports for the main thesis claim (Pillar 2).
- Paper **#7** is very recent (2025) — verify accessibility before citing.
- Paper **#12** is behind an LNCS paywall but available via the Semantic Scholar abstract link above.

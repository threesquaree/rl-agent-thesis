# Research Results & Key Findings

This document explains the key concepts, experimental conditions, and results from training the HRL Museum Dialogue Agent.

## Table of Contents

1. [The Option Collapse Problem](#the-option-collapse-problem)
2. [Simulator Differences](#simulator-differences)
3. [Model Architectures](#model-architectures)
4. [Experimental Conditions](#experimental-conditions)
5. [Interpreting Metrics](#interpreting-metrics)
6. [Key Findings](#key-findings)

---

## The Option Collapse Problem

### What is Option Collapse?

Option collapse is a well-known failure mode in hierarchical reinforcement learning where the agent converges to using a single dominant option, effectively losing the benefits of hierarchical structure.

In our museum dialogue agent:
- **Healthy behavior**: Agent uses all 4 options (Explain, AskQuestion, OfferTransition, Conclude) appropriately
- **Collapsed behavior**: Agent uses "Explain" 90%+ of the time, ignoring other strategies

### How We Measure It: Option Collapse Index (OCI)

```
OCI = max_option_percentage / 25.0
```

| OCI Value | Interpretation |
|-----------|----------------|
| 1.0 | Perfect balance (each option = 25%) |
| 2.0-2.5 | Mild preference, acceptable |
| 3.0+ | **Collapsed** - single option dominates |
| 4.0 | Complete collapse (one option = 100%) |

### Why Does It Happen?

Option collapse in our task stems from **reward-induced vanishing gradients**:

1. **Dense rewards dominate**: The engagement reward (dwell time) is received every turn, creating strong immediate feedback for "Explain" actions
2. **Temporal credit assignment fails**: Long-term benefits of transitions and questions are discounted away (γ^k → 0)
3. **Termination gradient vanishes**: The option-level advantage A_Ω approaches zero when one option consistently receives higher rewards

**Mathematical intuition**: If Explain always yields reward ~0.7 while other options yield ~0.5, the Option-Critic learns to never terminate the Explain option.

### Visual Evidence

See `figures/fig_h3_action_distribution_oci.jpg` - Shows option distribution over training:
- SMDP (hierarchical): Often shows Explain dominating (collapsed)
- MDP (flat): More balanced action distribution

---

## Simulator Differences

The training package includes **three visitor simulators** that model how museum visitors respond to the agent:

### 1. Sim8 (Default)

```bash
python train.py --simulator sim8
```

**Design**: Lightweight adapter with template-based responses and statistical gaze synthesis.

| Aspect | Implementation |
|--------|----------------|
| Dialogue | Template-based responses with LLM enhancement |
| Gaze/Engagement | Statistical sampling from persona distributions |
| Transitions | Probability-based (depends on exhibit completion) |
| Speed | Fast (~0.5s per turn with LLM) |

**Best for**: Development, hyperparameter tuning, fast iteration.

### 2. Sim8 Original (Neural)

```bash
python train.py --simulator sim8_original
```

**Design**: Full neural simulator from original research with trained T5 dialogue model and conditional VAE for gaze.

| Aspect | Implementation |
|--------|----------------|
| Dialogue | Fine-tuned T5 model on visitor data |
| Gaze/Engagement | Conditional VAE (persona × AOI × parent) |
| Realism | High - trained on real visitor behavior |
| Speed | Slower (~1-2s per turn) |

**Best for**: Final evaluation, realistic behavior testing.

### 3. State Machine (Literature-Grounded)

```bash
python train.py --simulator state_machine
```

**Design**: Deterministic state machine based on museum visitor behavior literature.

| Aspect | Implementation |
|--------|----------------|
| States | ENGAGED, CONFUSED, OVERLOADED, CURIOUS, FATIGUED, READY_TO_MOVE |
| Transitions | Deterministic based on agent actions |
| Dwell Ranges | Non-overlapping per state (clear reward signals) |
| Interpretability | High - states map to literature constructs |

**Literature basis**:
- Working memory limits (Bitgood 2013, Cowan 2001)
- Question spacing effects (Woo et al. 2024)
- Conversational repair (Schegloff et al. 1977)

**Best for**: Interpretable experiments, hypothesis testing.

### Simulator Comparison

| Feature | sim8 | sim8_original | state_machine |
|---------|------|---------------|---------------|
| Speed | Fast | Slow | Fast |
| Realism | Medium | High | Medium |
| Interpretability | Low | Low | High |
| Determinism | Stochastic | Stochastic | Deterministic |
| Research Use | Development | Evaluation | Hypothesis testing |

---

## Model Architectures

### SMDP (Hierarchical Option-Critic)

The **Semi-Markov Decision Process** model uses hierarchical reinforcement learning:

```
High-level: Option policy π_Ω(ω|s) selects among 4 options
Low-level: Intra-option policy π_ω(a|s) selects primitive actions
Termination: β_ω(s) decides when to switch options
```

**Options**:
| Option | Purpose | Subactions |
|--------|---------|------------|
| Explain | Share exhibit information | ExplainNewFact, ClarifyFact, ConnectToHistory |
| AskQuestion | Engage visitor | AskOpinion, AskMemory, AskPreference |
| OfferTransition | Move to new exhibit | SuggestNearby, SuggestRelated, OfferChoice |
| Conclude | End tour | SummarizeVisit, ThankVisitor, InviteReturn |

**Advantages**:
- Temporal abstraction (multi-step strategies)
- Interpretable option structure
- Transfer potential

**Challenges**:
- Option collapse (see above)
- Complex credit assignment

### MDP (Flat Actor-Critic)

The **Markov Decision Process** model uses standard actor-critic:

```
Single policy: π(a|s) selects among all 13 primitive actions
No hierarchy: Actions are selected directly each turn
```

**Advantages**:
- Simpler optimization
- More balanced action distribution
- Better coverage in practice

**Challenges**:
- No temporal abstraction
- Less interpretable strategies

---

## Experimental Conditions

### Pre-trained Models Included

| Model | Architecture | Simulator | Episodes | Seed |
|-------|--------------|-----------|----------|------|
| `H3_SMDP_StateMachine.pt` | SMDP (hierarchical) | state_machine | 1000 | 1 |
| `H3_MDP_StateMachine.pt` | MDP (flat) | state_machine | 1000 | 1 |

### Hyperparameters Used

```python
# Common settings
episodes = 1000
turns_per_episode = 50
learning_rate = 1e-4
gamma = 0.99
entropy_coef = 0.08

# SMDP-specific
termination_reg = 0.01
value_clip = 10.0
```

### Reward Modes

**Baseline** (default):
```
R_t = r_engagement + r_novelty
    = dwell_time + α × |new_facts|
```

**Augmented** (optional):
```
R_t = r_engagement + r_novelty + r_responsiveness + r_transition + r_conclude
```

---

## Interpreting Metrics

### Coverage (Primary Success Metric)

**Definition**: Percentage of exhibits where the agent shared at least one fact.

```
Coverage = |exhibits_with_facts| / |total_exhibits| × 100%
```

| Coverage | Interpretation |
|----------|----------------|
| < 40% | Poor - agent stuck at single exhibit |
| 40-60% | Moderate - some exploration |
| 60-80% | Good - reasonable coverage |
| > 80% | Excellent - comprehensive tour |

**Key insight**: MDP typically achieves 80-95% coverage; SMDP often stuck at 40-50% due to option collapse.

### Return (Cumulative Reward)

**Definition**: Sum of all rewards in an episode.

```
Return = Σ_t γ^t × r_t
```

Higher is better, but must be interpreted with coverage:
- High return + low coverage = exploitation (stuck at one exhibit)
- High return + high coverage = good policy

### Option Collapse Index (OCI)

See [Option Collapse section](#the-option-collapse-problem).

### Entropy

**Definition**: Policy entropy measures action diversity.

```
H(π) = -Σ_a π(a|s) log π(a|s)
```

| Entropy | Interpretation |
|---------|----------------|
| < 0.5 | Low diversity, deterministic |
| 0.5-1.0 | Moderate diversity |
| > 1.0 | High diversity, exploratory |

**Note**: Entropy should decrease during training as policy converges, but not collapse to zero.

---

## Key Findings

### 1. MDP Outperforms SMDP on Coverage

Despite theoretical advantages of hierarchy, the flat MDP model achieves significantly better exhibit coverage:

| Model | Coverage | Return | OCI |
|-------|----------|--------|-----|
| SMDP | 45-55% | 20-25 | 3.0+ (collapsed) |
| MDP | 85-95% | 45-55 | 1.5-2.0 (balanced) |

**Why?** Option collapse in SMDP prevents effective exploration.

### 2. State Machine Simulator Reveals Issues

The deterministic state machine simulator with non-overlapping dwell ranges makes option collapse more visible:
- Clear reward signals per state
- Deterministic transitions enable hypothesis testing
- Reveals that collapse is a learning problem, not simulator noise

### 3. Option Collapse is a Reward Structure Problem

Our experiments show that option collapse stems from:
1. **Reward magnitude**: Explain consistently yields higher immediate reward
2. **Temporal structure**: Long-term benefits of transitions are discounted away
3. **Credit assignment**: Hard to attribute tour success to individual option choices

### Figures

| Figure | Description |
|--------|-------------|
| `fig_h3_learning_curves.jpg` | Training curves comparing SMDP vs MDP |
| `fig_h3_action_distribution_oci.jpg` | Option/action distribution showing collapse |
| `fig_h3_coverage.jpg` | Exhibit coverage over training |
| `fig_h3_smdp_options.jpg` | SMDP option usage breakdown |

---

## Simulator Limitations & Deployment Challenges

### The Simulator Gap

A critical limitation of this work is that **none of the three simulators adequately capture real visitor behavior**. While they enable controlled experiments and hypothesis testing, they fail to model the complexity and unpredictability of actual museum visitors.

### Observed Behaviors in Simulated Training

**SMDP (Hierarchical)**:
- Tends to collapse to spamming "Explain" option
- Rarely uses AskQuestion or OfferTransition effectively
- Gets stuck at single exhibits due to option collapse

**MDP (Flat)**:
- More balanced than SMDP, but still shows patterns
- Spams "ExplainNewFact" subaction when facts are available
- Transitions only when exhibit is exhausted (no more new facts)
- Lacks nuanced visitor engagement modeling

### Why This Happens

The simulators, despite their differences, share fundamental limitations:

1. **Simplified visitor responses**: Real visitors have complex, context-dependent reactions that our templates and state machines cannot capture
2. **Deterministic or template-based**: Real behavior is highly variable and adaptive
3. **Missing social dynamics**: Real visitors respond to guide personality, group dynamics, fatigue, and external factors
4. **Limited feedback signals**: Simulators provide structured rewards (dwell time, confusion states) but miss subtle engagement cues

### Deployment Reality Check

**Critical concern**: Models trained on these simulators will likely exhibit problematic behaviors when deployed with real visitors:

- **SMDP models** will spam "Explain" actions, overwhelming visitors with information
- **MDP models** will spam "ExplainNewFact" until exhibits are exhausted, then abruptly transition
- Both architectures lack the nuanced understanding needed for natural, adaptive dialogue

The simulators are **research tools**, not production-ready proxies for real visitor behavior. They reveal algorithmic issues (like option collapse) but cannot validate that learned policies will work with actual humans.

### The Need for Real Data

To move from research to deployment, we need:

1. **Real visitor interaction data**: Recordings of actual museum tours with human guides
2. **Visitor behavior datasets**: Gaze patterns, engagement signals, and dialogue responses from real visits
3. **Human-in-the-loop evaluation**: Test trained policies with real visitors before deployment
4. **Iterative refinement**: Use real feedback to improve both simulators and training procedures

---

## Future Work

### Primary Direction: Real-World Data Collection

The most critical next step is **gathering data that better proxies real people**:

#### 1. Visitor Interaction Datasets

**Collect**:
- Audio/video recordings of museum tours with human guides
- Visitor gaze tracking data (eye movements, attention patterns)
- Dialogue transcripts with timing and context
- Visitor feedback (surveys, ratings, engagement measures)

**Use for**:
- Training more realistic simulators
- Validating learned policies
- Understanding real visitor behavior patterns

#### 2. Human-in-the-Loop Training

**Approach**:
- Deploy partially-trained agents with real visitors
- Collect interaction data and feedback
- Use human feedback as reward signal
- Iteratively improve policies based on real-world performance

**Challenges**:
- Ethical considerations (visitor consent, data privacy)
- Cost and logistics of real-world deployment
- Safety (ensuring agent doesn't frustrate visitors)

#### 3. Improved Simulator Development

**Goals**:
- Train neural simulators on real visitor data
- Model visitor personality, fatigue, and engagement more accurately
- Capture long-term visitor satisfaction, not just immediate rewards

**Methods**:
- Fine-tune language models on real visitor dialogue
- Train generative models for visitor responses
- Incorporate visitor demographic and contextual factors

#### 4. Transfer Learning from Simulation to Reality

**Research questions**:
- Can policies trained on simulators transfer to real visitors?
- What simulator features are most important for transfer?
- How much real data is needed to fine-tune simulated policies?

#### 5. Evaluation Frameworks

**Develop**:
- Metrics that correlate with real visitor satisfaction
- Evaluation protocols using human evaluators
- Long-term engagement measures (return visits, recommendations)

### Secondary Directions

- **Algorithm improvements**: Better credit assignment, reward shaping, termination learning
- **Architecture exploration**: Alternative hierarchical structures, attention mechanisms
- **Multi-modal inputs**: Incorporate visual cues, visitor body language, environmental context


# HRL Museum Dialogue Agent - Training Package

A hierarchical reinforcement learning system for training adaptive museum dialogue agents. This package implements the Option-Critic architecture for learning dialogue policies that guide museum visitors through exhibits.

## Features

- **Hierarchical RL (SMDP)**: Option-Critic architecture with learned termination
- **Flat RL Baseline (MDP)**: Standard actor-critic for comparison
- **Multiple Simulators**: 3 visitor simulation backends (sim8, sim8_original, state_machine)
- **Pre-trained Models**: Ready-to-use trained agents included
- **Comprehensive Evaluation**: Auto-generated plots and metrics
- **LLM Integration**: Groq API for natural language generation
- **Research Results**: Documentation of key findings including option collapse analysis

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

You need a Groq API key for LLM inference. Get one free at: https://console.groq.com/

**Option A: Environment Variable (Recommended)**

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_key_here"

# Windows CMD
set GROQ_API_KEY=your_key_here

# Linux/Mac
export GROQ_API_KEY="your_key_here"
```

**Option B: .env File**

```bash
# Copy the example file
cp env.example .env

# Edit .env and add your key:
GROQ_API_KEY=your_actual_key_here
```

**Option C: key.txt File**

Create a file named `key.txt` in the project root containing only your API key.

### 3. Train

```bash
# Basic training (500 episodes, SMDP hierarchical)
python train.py --episodes 500

# Flat MDP baseline (H1 hypothesis)
python train.py --variant h1 --episodes 500

# With GPU acceleration
python train.py --episodes 500 --device cuda

# Quick test (20 episodes)
python train.py --episodes 20 --name quick_test
```

### 4. Results

Training results are saved to `training_logs/experiments/YYYYMMDD/`:

```
experiment_folder/
├── models/           # Trained model checkpoints
├── logs/             # Training logs
├── evaluation/       # Auto-generated evaluation plots
├── maps/             # Museum visualization snapshots
├── metadata.json     # Experiment configuration
└── summary.json      # Training summary
```

## Simulator Options

Select the visitor simulator with `--simulator`:

| Simulator | Description | Use Case |
|-----------|-------------|----------|
| `sim8` (default) | Lightweight adapter with templates + statistical gaze | Fast training, development |
| `sim8_original` | Full neural simulator (T5 + VAE) | Research, realistic behavior |
| `state_machine` | Literature-grounded state machine | Deterministic, interpretable |

```bash
# Use state machine simulator
python train.py --simulator state_machine --episodes 500

# Use original neural simulator
python train.py --simulator sim8_original --episodes 500
```

## Pre-trained Models

We include pre-trained models from our H3 experiments (State Machine simulator, 1000 episodes):

| Model | Architecture | Coverage | Description |
|-------|--------------|----------|-------------|
| `H3_SMDP_StateMachine.pt` | Hierarchical (Option-Critic) | ~50% | Shows option collapse behavior |
| `H3_MDP_StateMachine.pt` | Flat (Actor-Critic) | ~90% | Better coverage, balanced actions |

### Using Pre-trained Models

```python
import torch

# Load SMDP model
checkpoint = torch.load('pretrained_models/H3_SMDP_StateMachine.pt')
print(f"Options: {checkpoint['options']}")
print(f"Avg reward: {checkpoint['avg_reward']:.2f}")

# Load MDP model  
checkpoint = torch.load('pretrained_models/H3_MDP_StateMachine.pt')
print(f"Actions: {checkpoint['subactions']}")
```

### Key Findings

**MDP outperforms SMDP on coverage** due to option collapse in hierarchical models:
- SMDP tends to converge to using "Explain" 90%+ of the time
- MDP maintains balanced action distribution
- See [results/RESULTS.md](results/RESULTS.md) for detailed analysis

## Research Results

The `results/` folder contains:
- **RESULTS.md**: Comprehensive documentation of findings
- **figures/**: Key visualization from experiments

### Option Collapse Problem

A critical finding is that hierarchical RL (SMDP) suffers from **option collapse** - the agent converges to using a single dominant option. This is measured by the Option Collapse Index (OCI):

| OCI | Interpretation |
|-----|----------------|
| 1.0 | Balanced (ideal) |
| 2.0-2.5 | Acceptable |
| 3.0+ | Collapsed (problematic) |

For more details, see [results/RESULTS.md](results/RESULTS.md).

## Training Modes

### Hierarchical RL (SMDP) - Default

The Option-Critic architecture with 4 high-level options:
- **Explain**: Share facts about exhibits
- **AskQuestion**: Engage visitors with questions
- **OfferTransition**: Suggest moving to another exhibit
- **Conclude**: End the tour

```bash
python train.py --episodes 500
```

### Flat RL (MDP) - Baseline

Standard actor-critic without hierarchical structure:

```bash
python train.py --variant h1 --episodes 500
```

### Reward Modes

```bash
# Baseline rewards: engagement + novelty only
python train.py --reward_mode baseline --episodes 500

# Augmented rewards: + responsiveness, transition, conclude bonuses
python train.py --reward_mode augmented --episodes 500
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 500 | Number of training episodes |
| `--turns` | 50 | Max turns per episode |
| `--lr` | 1e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--device` | cpu | Training device (cpu/cuda) |
| `--simulator` | sim8 | Simulator backend |
| `--reward_mode` | baseline | Reward function (baseline/augmented) |
| `--variant` | None | Model variant (baseline/h1) |
| `--seed` | 1 | Random seed |

For the complete argument reference, see [ARGUMENTS.md](ARGUMENTS.md).

## Project Structure

```
training_export/
├── train.py                    # Main training entry point
├── LLM_CONFIG.py               # LLM configuration
├── museum_knowledge_graph.json # Museum exhibit data
├── README.md                   # This file
├── ARGUMENTS.md                # Full CLI argument reference
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── env.example                 # API key template
├── src/
│   ├── agent/                  # RL agents (Actor-Critic, networks)
│   ├── environment/            # Gym environment
│   ├── simulator/              # Visitor simulators (sim8, sim8_original, state_machine)
│   ├── training/               # Training loops (HRL)
│   ├── flat_rl/                # Flat MDP implementation
│   ├── visualization/          # Logging and visualization
│   └── utils/                  # Utilities (LLM, DialogueBERT, etc.)
├── models/
│   └── sim8_original/          # Neural models for sim8_original simulator
│       ├── gaze_vae_with_parent.pth
│       └── visitor_dialogue/
├── pretrained_models/          # Ready-to-use trained agents
│   ├── H3_SMDP_StateMachine.pt # Hierarchical model (shows option collapse)
│   └── H3_MDP_StateMachine.pt  # Flat model (better coverage)
├── results/                    # Research results and analysis
│   ├── RESULTS.md              # Detailed findings documentation
│   └── figures/                # Key visualizations
└── tools/
    └── create_evaluation_plots.py
```

## LLM Configuration

Edit `LLM_CONFIG.py` to change models:

```python
from LLM_CONFIG import LLMConfig

# Use cheaper 8B model (default)
LLMConfig.preset_cheap()

# Use high quality 70B model
LLMConfig.preset_high_quality()

# Custom configuration
LLMConfig.AGENT_LLM_MODEL = "llama-3.3"
LLMConfig.SIMULATOR_LLM_MODEL = "llama-3.1-8b"
```

Available Groq models:
- `llama-3.1-8b`: Fast, cheap (~$1-2 for 500 episodes)
- `llama-3.1`: Higher quality 70B
- `llama-3.3`: Latest 70B model

## Example Commands

```bash
# Standard training with evaluation
python train.py --episodes 500 --name my_experiment

# Compare SMDP vs MDP
python train.py --compare --episodes 500

# Hyperparameter tuning
python train.py --episodes 500 --lr 5e-5 --gamma 0.95

# Advanced: termination tuning
python train.py --episodes 500 --termination-reg 0.05 --entropy-coef 0.15

# Debug mode with verbose output
python train.py --episodes 50 --verbose --show-prompts
```

## Troubleshooting

### "GROQ_API_KEY not set"

Ensure your API key is configured using one of the three methods above. Verify with:

```bash
# PowerShell
echo $env:GROQ_API_KEY

# Linux/Mac
echo $GROQ_API_KEY
```

### CUDA out of memory

Use CPU or reduce batch size:

```bash
python train.py --device cpu --episodes 500
```

### Slow training

Use the lightweight simulator:

```bash
python train.py --simulator sim8 --episodes 500
```

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{hrl_museum_dialogue,
  title={Hierarchical Reinforcement Learning for Adaptive Museum Dialogue},
  author={[Author Name]},
  year={2026},
  school={[University Name]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""
Train HRL Museum Dialogue Agent

Unified training script with all features:
- Configurable reward mode (baseline vs augmented)
- Configurable simulator (sim8 vs state_machine)
- Comprehensive evaluation (always enabled)
- Map visualization at specified turn numbers
- Detailed logging and metrics

Usage:
    # Train baseline SMDP with baseline rewards (default)
    python train.py --episodes 500 --reward_mode baseline
    
    # Train flat MDP (H1)
    python train.py --variant h1 --episodes 500 --reward_mode baseline
    
    # Train with augmented rewards (H2)
    python train.py --episodes 500 --reward_mode augmented
    
    # Train with state machine simulator (H3)
    python train.py --episodes 500 --simulator state_machine
    
    # Quick validation test (20 episodes)
    python train.py --episodes 20 --reward_mode baseline --name test_validation
"""

import argparse
import subprocess
import torch
import os
import json
import sys
from datetime import datetime
from pathlib import Path
from src.training.training_loop import HRLTrainingLoop
from src.flat_rl.training_loop import FlatTrainingLoop


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility (standard RL practice)."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_folder(name, seed, episodes, experiment_type='minor'):
    """Create experiment folder with semantic naming: {name}_S{seed}_{episodes}ep
    
    Follows standard RL research conventions for ablation studies.
    Folders are organized by date: training_logs/experiments/YYYYMMDD/{name}_S{seed}_{episodes}ep/
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    
    # Create date-based folder structure: training_logs/experiments/YYYYMMDD/
    exp_base = Path("training_logs/experiments")
    date_folder = exp_base / date_str
    date_folder.mkdir(parents=True, exist_ok=True)
    
    # Semantic folder name: {name}_S{seed}_{episodes}ep
    # This matches major_results convention (e.g., H1_MDP_Flat_S1_1000ep)
    exp_name = f"{name}_S{seed}_{episodes}ep"
    exp_dir = date_folder / exp_name
    
    # Handle existing folder (append timestamp if collision)
    if exp_dir.exists():
        timestamp = now.strftime("%H%M%S")
        exp_name = f"{name}_S{seed}_{episodes}ep_{timestamp}"
    exp_dir = date_folder / exp_name
    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "maps").mkdir(exist_ok=True)
    (exp_dir / "models").mkdir(exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "detailed_logs").mkdir(exist_ok=True)
    (exp_dir / "parameterization_results").mkdir(exist_ok=True)
    
    return exp_dir




def main():
    parser = argparse.ArgumentParser(
        description='Train HRL Museum Dialogue Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reward Modes (per paper.tex Section 4.7):
  BASELINE:   R_t = r^eng + r^nov  (engagement + novelty ONLY)
  AUGMENTED:  R_t = r^eng + r^nov + r^resp + r^trans + r^conclude + r^ask

Examples:
  # Baseline SMDP (default)
  python train.py --episodes 500 --reward_mode baseline
  
  # Flat MDP with baseline rewards (H1)
  python train.py --variant h1 --reward_mode baseline --episodes 500
  
  # SMDP with augmented rewards (H2)
  python train.py --reward_mode augmented --episodes 500
  
  # State machine simulator (H3)
  python train.py --simulator state_machine --episodes 500
  
  # Quick test
  python train.py --episodes 20 --reward_mode baseline --name quick_test
        """
    )
    
    # ===== CORE TRAINING ARGUMENTS =====
    parser.add_argument('--mode', type=str, default='hrl',
                       choices=['hrl', 'flat'],
                       help='Training mode: hierarchical (hrl) or flat (flat). Default: hrl')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name of experiment for logging and display (e.g., "Exp1_LR5e5_Baseline")')
    parser.add_argument('--variant', type=str, default=None,
                       choices=['baseline', 'h1'],
                       help='Model variant: baseline (SMDP hierarchical), h1 (flat MDP). Use --reward_mode for baseline vs augmented. Overrides --mode if specified.')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes (default: 500)')
    parser.add_argument('--turns', type=int, default=50,
                       help='Max turns per episode (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4, reduced for stability)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for training (default: cpu)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (optional identifier, auto-set from variant)')
    parser.add_argument('--experiment-type', type=str, default='minor',
                       choices=['major', 'minor'],
                       help='Experiment type: major (significant changes) or minor (default: minor)')
    parser.add_argument('--compare', action='store_true',
                       help='Run both HRL (SMDP) and flat (MDP) back-to-back with identical settings')
    parser.add_argument('--seed', type=int, default=1,
                       help='Random seed for reproducibility (default: 1). Use different seeds (1,2,3) for statistical significance.')
    
    # ===== REWARD PARAMETERS (per paper formalization, Section 4.7) =====
    # Note: Per paper.tex line 637, r^eng_t = dwell_t and r^nov_t = α × |new facts|
    # Reward parameters (per paper.tex Section 4.7, lines 681-709)
    # All weights configurable with paper.tex defaults
    
    parser.add_argument('--w-engagement', type=float, default=1.0,
                       help='Engagement reward weight: r^eng_t = dwell_t × w_engagement (default: 1.0)')
    parser.add_argument('--centred-engagement', action='store_true',
                       help='Use centred engagement reward: r^ceng_t = w_e * (dwell_t - EMA(dwell)) (thesis Eq. 2). Reduces bias from visitor baseline dwell differences.')
    parser.add_argument('--dwell-ema-alpha', type=float, default=0.1,
                       help='EMA decay factor for centred engagement baseline (default: 0.1, effective memory ~10 steps)')
    parser.add_argument('--novelty-per-fact', type=float, default=1.0,
                       help='Novelty reward scale: r^nov_t = novelty_per_fact × |new_facts| (default: 1.0)')
    parser.add_argument('--engagement-gated-novelty', action='store_true',
                       help='Use engagement-gated novelty: r_t = dwell * novelty_credit. Novelty only pays off when visitor is engaged. Replaces additive engagement+novelty.')
    parser.add_argument('--broadened-novelty', action='store_true',
                       help='Use broadened novelty reward (thesis Eq. 5). Replaces standard novelty with multi-action credit: ExplainNewFact + RepeatFact + ClarifyFact + AskQuestion - staleness.')
    parser.add_argument('--alpha-new', type=float, default=1.0,
                       help='Broadened novelty: reward for ExplainNewFact with new facts (default: 1.0)')
    parser.add_argument('--alpha-rep', type=float, default=0.3,
                       help='Broadened novelty: reward for RepeatFact (default: 0.3)')
    parser.add_argument('--alpha-clar', type=float, default=0.3,
                       help='Broadened novelty: reward for ClarifyFact (default: 0.3)')
    parser.add_argument('--alpha-ask', type=float, default=0.2,
                       help='Broadened novelty: reward for AskOpinion/AskMemory/AskClarification (default: 0.2)')
    parser.add_argument('--alpha-stale', type=float, default=1.0,
                       help='Broadened novelty: staleness penalty at exhausted exhibit for non-Explain actions (default: 1.0)')
    parser.add_argument('--alpha-transition', type=float, default=0.4,
                       help='Broadened novelty: reward for transition actions (default: 0.4)')
    parser.add_argument('--action-repeat-penalty', type=float, default=0.15,
                       help='Penalty per consecutive same-subaction over threshold (default: 0.15)')
    parser.add_argument('--action-repeat-threshold', type=int, default=2,
                       help='Number of consecutive repeats before penalty kicks in (default: 2)')
    parser.add_argument('--w-responsiveness', type=float, default=0.5,
                       help='Responsiveness reward: +w_responsiveness (answer) / -0.6*w_responsiveness (deflect) (default: 0.5, increased for H2)')
    parser.add_argument('--w-conclude', type=float, default=0.4,
                       help='Conclude bonus: w_conclude × |exhibits_covered| (default: 0.4, increased for H2)')
    parser.add_argument('--w-ask', type=float, default=0.5,
                       help='Question-asking incentive: hybrid reward considering spacing, engagement impact, and response quality (default: 0.5, increased for H2)')
    
    # ===== TERMINATION TUNING PARAMETERS (H1 Termination Tuning) =====
    parser.add_argument('--termination-reg', type=float, default=0.01,
                       help='Termination regularization coefficient (default: 0.01, try 0.05-0.1 to encourage option switching)')
    parser.add_argument('--intra-option-threshold', type=float, default=0.1,
                       help='Threshold for intra-option advantage termination signal (default: 0.1)')
    parser.add_argument('--intra-option-weight', type=float, default=0.5,
                       help='Weight for intra-option termination signal (default: 0.5)')
    parser.add_argument('--deliberation-cost', type=float, default=0.0,
                       help='Per-step cost for staying in option - Harb et al. 2018 (default: 0.0, try 0.01-0.05)')
    parser.add_argument('--entropy-floor', type=float, default=0.02,
                       help='Minimum entropy coefficient for option exploration (default: 0.02, try 0.05 for more exploration)')
    parser.add_argument('--max-option-duration', type=int, default=None,
                       help='Maximum steps in an option before forced termination - Sutton et al. 1999 (default: None=disabled, try 8 for museum dialogue)')
    parser.add_argument('--entropy-coef', type=float, default=0.08,
                       help='Initial entropy coefficient for exploration (default: 0.08, try 0.15-0.25 for aggressive exploration)')
    parser.add_argument('--entropy-decay-start', type=int, default=0,
                       help='Episode to start entropy decay (default: 0, try 100 to allow initial exploration)')
    parser.add_argument('--entropy-decay-end', type=int, default=None,
                       help='Episode to finish entropy decay (default: max_episodes, try shorter for faster convergence)')
    parser.add_argument('--adaptive-entropy', action='store_true',
                       help='Enable OCI-aware adaptive entropy that boosts exploration when collapse detected')
    parser.add_argument('--adaptive-entropy-threshold', type=float, default=2.5,
                       help='OCI threshold for adaptive entropy boost (default: 2.5)')
    parser.add_argument('--adaptive-entropy-multiplier', type=float, default=1.5,
                       help='Multiplier for entropy boost when collapse detected (default: 1.5)')
    
    # ===== H1 ADVANCED: SEPARATE LEARNING RATES & ENTROPIES =====
    parser.add_argument('--lr-intra-option', type=float, default=None,
                       help='Learning rate for intra-option policies (default: same as --lr). H1 Advanced: Allows faster learning for subactions.')
    parser.add_argument('--entropy-coef-option', type=float, default=None,
                       help='Entropy coefficient for option policy (default: same as --entropy-coef). H1 Advanced: Separate exploration for option selection.')
    parser.add_argument('--entropy-coef-intra', type=float, default=None,
                       help='Entropy coefficient for intra-option policies (default: same as --entropy-coef). H1 Advanced: Separate exploration for subaction selection.')
    parser.add_argument('--diversity-reward-coef', type=float, default=0.0,
                       help='Diversity reward coefficient for option diversity (default: 0.0). H1 Advanced: Kamat & Precup 2020 diversity-enriched Option-Critic.')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value loss coefficient (default: 0.5). H1 Advanced: Balances policy vs value learning.')
    parser.add_argument('--target-update-interval', type=int, default=20,
                       help='Episodes between target network updates (default: 20). H1 Advanced: More frequent updates may improve learning.')
    parser.add_argument('--value-clip', type=float, default=10.0,
                       help='Value clipping range: clip values and targets to [-value-clip, value-clip] (default: 10.0). Use 50.0 to allow higher value estimates for high-return episodes.')
    
    # ===== SMDP COVERAGE FIX: REWARD STRUCTURE PARAMETERS =====
    parser.add_argument('--exhaustion-penalty', type=float, default=-1.0,
                       help='Penalty for Explain actions at exhausted exhibits (default: -0.5, try -2.0 for stronger signal)')
    parser.add_argument('--transition-bonus', type=float, default=0.3,
                       help='Immediate bonus for successful exhibit transitions (default: 0.0, try 1.5)')
    parser.add_argument('--zero-engagement-exhausted', action='store_true',
                       help='Zero engagement reward for Explain at exhausted exhibits (creates Q-value separation)')
    parser.add_argument('--beta-supervision-weight', type=float, default=0.0,
                       help='Beta supervision weight for termination guidance (default: 0.0 = pure Option-Critic, try 0.5-1.0 for heuristic guidance)')

    # ===== RESPONSE TYPE FEATURES (thesis extension) =====
    parser.add_argument('--response-type-feature', action='store_true',
                       help='Add visitor response type as 6-dim one-hot to state vector (acknowledgment, follow_up_question, question, statement, confusion, silence)')
    parser.add_argument('--response-type-reward', action='store_true',
                       help='Add response-type reward component: positive for engaged reactions, negative for confusion/silence')
    parser.add_argument('--w-response-type', type=float, default=0.3,
                       help='Weight for response-type reward component (default: 0.3)')
    
    # ===== EVALUATION & VISUALIZATION =====
    parser.add_argument('--map-interval', type=int, default=50,
                       help='Save map visualization every N episodes (default: 50, set to 0 to disable)')
    parser.add_argument('--save-map-frames', action='store_true',
                       help='Save map snapshots at EVERY turn (for all episodes)')
    parser.add_argument('--live-map-display', action='store_true',
                       help='Show live map windows during training (default: save only)')
    
    # ===== SIMULATOR SELECTION =====
    parser.add_argument('--simulator', type=str, default='sim8',
                       choices=['sim8', 'sim8_original', 'state_machine', 'hybrid'],
                       help='Simulator type: sim8 (adapted), sim8_original (neural T5+VAE), state_machine, or hybrid (state_machine + sim8 dynamics). Default: sim8')
    parser.add_argument('--stochasticity', type=float, default=0.5,
                       help='Hybrid simulator: sim8 influence (0.0=pure state machine, 0.5=balanced, 1.0=max sim8). Default: 0.5')
    
    # ===== REWARD MODE (per paper.tex) =====
    parser.add_argument('--reward_mode', type=str, default='baseline',
                       choices=['baseline', 'augmented'],
                       help='Reward mode: baseline (engagement+novelty only) or augmented (+responsiveness, transition, conclude, ask). Default: baseline')
    
    # ===== HYPOTHESIS-SPECIFIC ARGUMENTS =====
    parser.add_argument('--termination', type=str, default='learned',
                       choices=['learned', 'fixed-3', 'threshold'],
                       help='Termination strategy (H5): learned (Option-Critic), fixed-3 (always 3 turns), threshold (dwell < 0.5). Default: learned')
    parser.add_argument('--state-representation', type=str, default='dialoguebert',
                       choices=['dialoguebert', 'dialogue_act'],
                       help='State representation variant (H4): dialoguebert (149-d baseline), dialogue_act (23-d compact). Default: dialoguebert')
    parser.add_argument('--option-granularity', type=str, default='medium',
                       choices=['medium', 'coarse', 'coarse_3opt', 'coarse_4opt'],
                       help='Option granularity (H6): medium (4 options - function-based baseline), coarse (2 options - Explain/Engage), coarse_3opt (3 options - deprecated, no Conclude), coarse_4opt (4 options - reward-aligned with Conclude). Default: medium')
    
    # ===== DEBUGGING & TESTING =====
    parser.add_argument('--show-prompts', action='store_true',
                       help='Show LLM prompts during training (for debugging)')
    parser.add_argument('--force-option', type=str, default=None,
                       choices=['Explain', 'AskQuestion', 'OfferTransition', 'Conclude'],
                       help='Force agent to always choose this option (testing only)')
    parser.add_argument('--force-subaction', type=str, default=None,
                       help='Force agent to always choose this subaction (testing only)')
    parser.add_argument('--enable-live-monitor', action='store_true',
                       help='Enable live training monitor with turn-by-turn visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output mode')
    
    args = parser.parse_args()

    # Warn if broadened novelty conflicts with novelty-per-fact
    if args.broadened_novelty and args.novelty_per_fact != 1.0:
        print("WARNING: --broadened-novelty is active; --novelty-per-fact is ignored (use --alpha-new instead)")

    # Handle variant selection - override mode and name if variant is specified
    # Note: Use --reward_mode to switch between baseline/augmented rewards
    variant_configs = {
        'baseline': {
            'mode': 'hrl',
            'name': 'baseline_smdp',
            'loop_class': HRLTrainingLoop,
            'description': 'Baseline: SMDP (Hierarchical Option-Critic)'
        },
        'h1': {
            'mode': 'flat',
            'name': 'h1_flat_mdp',
            'loop_class': FlatTrainingLoop,
            'description': 'H1: Flat MDP (no hierarchical structure)'
        },
    }
    
    # If variant is specified, override mode and name
    if args.variant:
        variant_config = variant_configs[args.variant]
        args.mode = variant_config['mode']
        if args.name is None:
            args.name = variant_config['name']
        if args.experiment_type == 'minor':
            args.experiment_type = 'major'  # Variants are major experiments
    
    def run_subprocess_for_mode(base_args, mode_label, name_suffix):
        script_path = Path(__file__).resolve()
        cmd = [sys.executable, str(script_path)] + base_args + ["--mode", mode_label, "--name", name_suffix]
        print(f"\n[COMPARE] Running {mode_label.upper()} configuration -> {name_suffix}\n")
        subprocess.run(cmd, check=True)

    if args.compare:
        def remove_option(seq, flag, has_value=True):
            result = []
            skip = False
            for item in seq:
                if skip:
                    skip = False
                    continue
                if item == flag:
                    if has_value:
                        skip = True
                    continue
                result.append(item)
            return result
        
        base_argv = sys.argv[1:]
        base_argv = remove_option(base_argv, "--compare", has_value=False)
        base_argv = remove_option(base_argv, "--mode", has_value=True)
        base_argv = remove_option(base_argv, "--name", has_value=True)
        base_name = args.name or "compare"
        run_subprocess_for_mode(base_argv, "hrl", f"{base_name}_hrl")
        run_subprocess_for_mode(base_argv, "flat", f"{base_name}_flat")
        return

    # Check for Groq API key - try to load from key.txt if not set
    if "GROQ_API_KEY" not in os.environ:
        key_file = Path("key.txt")
        if key_file.exists():
            try:
                api_key = key_file.read_text().strip()
                os.environ["GROQ_API_KEY"] = api_key
                print("[OK] Loaded Groq API key from key.txt")
            except Exception as e:
                print("=" * 80)
                print("⚠️  WARNING: GROQ_API_KEY not set and failed to read key.txt!")
                print("=" * 80)
                print(f"Error: {e}")
                print("Set it with:")
                print("  Windows PowerShell: $env:GROQ_API_KEY='your_key_here'")
                print("  Linux/Mac: export GROQ_API_KEY='your_key_here'")
                print("  Or create key.txt in project root with your API key")
                print("=" * 80)
                print()
        else:
            print("=" * 80)
            print("⚠️  WARNING: GROQ_API_KEY not set and key.txt not found!")
            print("=" * 80)
            print("Set it with:")
            print("  Windows PowerShell: $env:GROQ_API_KEY='your_key_here'")
            print("  Linux/Mac: export GROQ_API_KEY='your_key_here'")
            print("  Or create key.txt in project root with your API key")
            print("=" * 80)
            print()
    
    # Set random seeds for reproducibility (standard RL practice)
    set_random_seeds(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Create experiment folder with semantic naming: {name}_S{seed}_{episodes}ep
    folder_name = args.name
    if args.mode == 'flat':
        if folder_name:
            folder_name = f"flat_{folder_name}"
        else:
            folder_name = "flat"
    elif not folder_name:
        # Default name based on mode if not provided
        folder_name = "hrl" if args.mode == 'hrl' else args.mode
    
    exp_dir = create_experiment_folder(folder_name, args.seed, args.episodes, args.experiment_type)
    
    # Convert to absolute path for consistency
    exp_dir = exp_dir.resolve()
    
    # Set experiment directory environment variable (use absolute path)
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    
    # Set BERT mode for baseline: use standard BERT (no turn/role embeddings)
    # This can be overridden by setting HRL_BERT_MODE environment variable before running
    if os.environ.get('HRL_BERT_MODE') is None:
        os.environ["HRL_BERT_MODE"] = "standard"
        print("Baseline: Using standard BERT (HRL_BERT_MODE=standard)")
    else:
        print(f"Using BERT mode from environment: {os.environ.get('HRL_BERT_MODE')}")
    
    # Save experiment metadata with reward weights
    metadata = {
        "experiment_name": args.name or folder_name,
        "seed": args.seed,
        "experiment_type": args.experiment_type,
        "mode": args.mode,
        "reward_mode": args.reward_mode,
        "simulator": args.simulator,
        "stochasticity": args.stochasticity if args.simulator == "hybrid" else None,
        "timestamp": datetime.now().isoformat(),
        "episodes": args.episodes,
        "max_turns_per_episode": args.turns,
        "device": args.device,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "bert_mode": os.environ.get("HRL_BERT_MODE", "standard"),
        "termination_strategy": args.termination,
        "state_representation": args.state_representation,
        "option_granularity": args.option_granularity,
        "reward_parameters": {
            "reward_mode": args.reward_mode,
            "w_engagement": args.w_engagement,
            "centred_engagement": args.centred_engagement,
            "engagement_gated_novelty": args.engagement_gated_novelty,
            "dwell_ema_alpha": args.dwell_ema_alpha,
            "novelty_per_fact": args.novelty_per_fact,
            "broadened_novelty": args.broadened_novelty,
            "alpha_new": args.alpha_new,
            "alpha_rep": args.alpha_rep,
            "alpha_clar": args.alpha_clar,
            "alpha_ask": args.alpha_ask,
            "alpha_stale": args.alpha_stale,
            "alpha_transition": args.alpha_transition,
            "action_repeat_penalty": args.action_repeat_penalty,
            "action_repeat_threshold": args.action_repeat_threshold,
            "w_responsiveness": args.w_responsiveness,
            "w_conclude": args.w_conclude,
            "response_type_feature": args.response_type_feature,
            "response_type_reward": args.response_type_reward,
            "w_response_type": args.w_response_type
        },
        "note": f"Reward mode: {args.reward_mode}. Baseline = engagement + novelty only. Augmented adds responsiveness, transition, conclude, question-asking.",
        "map_interval": args.map_interval,
        "save_map_frames": args.save_map_frames
    }
    
    with open(exp_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print configuration
    print("=" * 80)
    if args.variant:
        variant_config = variant_configs[args.variant]
        print(f"TRAINING: {variant_config['description']}")
    else:
        print(f"TRAINING {args.mode.upper()} MUSEUM DIALOGUE AGENT")
    print("=" * 80)
    print(f"EXPERIMENT: {exp_dir.name}")
    print(f"Architecture: Actor-Critic (per paper.tex)")
    print(f"Options: Explain, Ask, Transition, Conclude")
    print(f"State: 149-dim (Focus + History + DialogueBERT Intent + Context)")
    print(f"LLM: Groq API (Llama 3.1 8B)")
    print("=" * 80)
    print(f"Episodes: {args.episodes}")
    print(f"Turns/episode: {args.turns}")
    print(f"Learning rate: {args.lr}")
    print(f"Gamma: {args.gamma}")
    print(f"Device: {args.device}")
    print(f"Simulator: {args.simulator}" + (f" (stochasticity={args.stochasticity})" if args.simulator == "hybrid" else ""))
    print(f"Reward Mode: {args.reward_mode.upper()}")
    print()
    if args.reward_mode == "baseline":
        print("REWARD FUNCTION (BASELINE - engagement + novelty ONLY):")
    else:
        print("REWARD FUNCTION (AUGMENTED - includes all components):")
    if args.centred_engagement:
        print(f"  Engagement:         r^ceng_t = {args.w_engagement:.2f} * (dwell_t - EMA(dwell))  [CENTRED, alpha={args.dwell_ema_alpha}]")
    else:
        print(f"  Engagement:         r^eng_t = dwell_t × {args.w_engagement:.2f}  [STANDARD]")
    if args.broadened_novelty:
        print(f"  Novelty:            BROADENED (Eq. 5): α_new={args.alpha_new}, α_rep={args.alpha_rep}, α_clar={args.alpha_clar}, α_ask={args.alpha_ask}, α_stale={args.alpha_stale}")
    else:
        print(f"  Novelty:            r^nov_t = {args.novelty_per_fact:.2f} × |new_facts|  [STANDARD]")
    if args.engagement_gated_novelty:
        print(f"  Combination:        ENGAGEMENT-GATED: r_t = dwell × novelty_credit (multiplicative)")
    print(f"  Responsiveness:     +{args.w_responsiveness:.2f} (answer) / -{args.w_responsiveness*0.6:.2f} (deflect)")
    print(f"  Conclude:           {args.w_conclude:.2f} × |exhibits_covered|")
    print("  Transition:         -0.20 (0 facts) / -0.16 (1 fact)")
    print()
    if args.response_type_reward:
        print(f"  Response Type:      w={args.w_response_type:.2f} × rtype_value (ack=+0.3, follow_up=+0.25, q=+0.1, stmt=0, confusion=-0.3, silence=-0.2)")
    if args.response_type_feature:
        print(f"  State Feature:      +6-dim one-hot response type vector")
    print("  Note: Simplified reward function - no exhausted exhibit penalty, no repeat bonuses")
    print("        Behavioral shaping (exhausted exhibits, spam) via simulator dwell adjustments")
    print("        Transition insufficiency has 3-turn exemption after successful transition")
    print()
    print("EVALUATION & VISUALIZATION:")
    if args.map_interval > 0:
        print(f"  Map interval:        every {args.map_interval} episodes")
    if args.save_map_frames:
        print(f"  Map frames:         saving EVERY turn (all episodes)")
    print(f"  Evaluation:         ALWAYS ENABLED")
    print("=" * 80)
    print()
    
    # Pass reward parameters to environment via environment variables
    os.environ["HRL_REWARD_MODE"] = args.reward_mode  # baseline or augmented
    os.environ["HRL_W_ENGAGEMENT"] = str(args.w_engagement)
    os.environ["HRL_NOVELTY_PER_FACT"] = str(args.novelty_per_fact)
    os.environ["HRL_W_RESPONSIVENESS"] = str(args.w_responsiveness)
    os.environ["HRL_W_CONCLUDE"] = str(args.w_conclude)
    os.environ["HRL_CENTRED_ENGAGEMENT"] = "1" if args.centred_engagement else "0"
    os.environ["HRL_ENGAGEMENT_GATED_NOVELTY"] = "1" if args.engagement_gated_novelty else "0"
    os.environ["HRL_DWELL_EMA_ALPHA"] = str(args.dwell_ema_alpha)

    # Broadened novelty reward (thesis Eq. 5)
    os.environ["HRL_BROADENED_NOVELTY"] = "1" if args.broadened_novelty else "0"
    os.environ["HRL_ALPHA_NEW"] = str(args.alpha_new)
    os.environ["HRL_ALPHA_REP"] = str(args.alpha_rep)
    os.environ["HRL_ALPHA_CLAR"] = str(args.alpha_clar)
    os.environ["HRL_ALPHA_ASK"] = str(args.alpha_ask)
    os.environ["HRL_ALPHA_STALE"] = str(args.alpha_stale)
    os.environ["HRL_ALPHA_TRANSITION"] = str(args.alpha_transition)

    # Anti-spam: action repetition penalty
    os.environ["HRL_ACTION_REPEAT_PENALTY"] = str(args.action_repeat_penalty)
    os.environ["HRL_ACTION_REPEAT_THRESHOLD"] = str(args.action_repeat_threshold)

    # H1 Advanced: Pass diversity reward coefficient
    os.environ["HRL_DIVERSITY_REWARD_COEF"] = str(args.diversity_reward_coef)
    
    # SMDP Coverage Fix: Pass reward structure parameters
    os.environ["HRL_EXHAUSTION_PENALTY"] = str(args.exhaustion_penalty)
    os.environ["HRL_TRANSITION_BONUS"] = str(args.transition_bonus)
    os.environ["HRL_ZERO_ENGAGEMENT_EXHAUSTED"] = "1" if args.zero_engagement_exhausted else "0"

    # Response type features
    os.environ["HRL_RESPONSE_TYPE_FEATURE"] = "1" if args.response_type_feature else "0"
    os.environ["HRL_RESPONSE_TYPE_REWARD"] = "1" if args.response_type_reward else "0"
    os.environ["HRL_W_RESPONSE_TYPE"] = str(args.w_response_type)
    
    # Initialize training loop based on variant or mode
    if args.variant:
        variant_config = variant_configs[args.variant]
        loop_cls = variant_config['loop_class']
    else:
        # Default behavior: use mode
        loop_cls = HRLTrainingLoop if args.mode == 'hrl' else FlatTrainingLoop
    
    training_loop = loop_cls(
        max_episodes=args.episodes,
        max_turns_per_episode=args.turns,
        knowledge_graph_path="museum_knowledge_graph.json",
        learning_rate=args.lr,
        gamma=args.gamma,
        use_actor_critic=True,
        device=args.device,
        turn_delay=0.0,
        show_prompts=args.show_prompts,
        force_option=args.force_option,
        force_subaction=args.force_subaction,
        enable_live_monitor=args.enable_live_monitor,
        save_metrics=True,  # Always enabled
        enable_map_viz=args.map_interval > 0 or args.save_map_frames,
        save_map_frames=args.save_map_frames,
        live_map_display=args.live_map_display,
        map_interval=args.map_interval,
        verbose=args.verbose,
        simulator_type=args.simulator,
        stochasticity=args.stochasticity,
        termination_strategy=args.termination,
        state_representation=args.state_representation,
        option_granularity=args.option_granularity,
        experiment_name=args.experiment_name,
        # H1 Termination Tuning Parameters
        termination_reg=args.termination_reg,
        intra_option_threshold=args.intra_option_threshold,
        intra_option_weight=args.intra_option_weight,
        deliberation_cost=args.deliberation_cost,
        entropy_floor=args.entropy_floor,
        max_option_duration=args.max_option_duration,
        # H1 Phase 2: Entropy Control Parameters
        entropy_coef=args.entropy_coef,
        entropy_decay_start=args.entropy_decay_start,
        entropy_decay_end=args.entropy_decay_end if args.entropy_decay_end else args.episodes,
        adaptive_entropy=args.adaptive_entropy,
        adaptive_entropy_threshold=args.adaptive_entropy_threshold,
        adaptive_entropy_multiplier=args.adaptive_entropy_multiplier,
        # H1 Advanced: Separate learning rates and entropies
        lr_intra_option=args.lr_intra_option,
        entropy_coef_option=args.entropy_coef_option,
        entropy_coef_intra=args.entropy_coef_intra,
        # H1 Advanced: Value learning parameters
        value_loss_coef=args.value_loss_coef,
        target_update_interval=args.target_update_interval,
        value_clip=args.value_clip,
        # SMDP Coverage Fix: Beta supervision weight
        beta_supervision_weight=args.beta_supervision_weight
    )
    
    # Train
    print("Starting training...")
    print()
    training_loop.run_training()
    
    # Save model
    model_path = exp_dir / "models" / "trained_agent.pt"
    os.makedirs(model_path.parent, exist_ok=True)
    
    checkpoint = {
        'agent_state_dict': training_loop.agent.network.state_dict(),
        'options': training_loop.env.options,
        'subactions': training_loop.env.subactions,
        'state_dim': training_loop.env.observation_space.shape[0],
        'config': {
            'episodes': args.episodes,
            'turns': args.turns,
            'lr': args.lr,
            'gamma': args.gamma,
            'reward_parameters': metadata['reward_parameters']
        },
        'timestamp': datetime.now().isoformat(),
        'total_episodes': training_loop.total_episodes,
        'avg_reward': sum(training_loop.episode_rewards) / len(training_loop.episode_rewards) 
                     if training_loop.episode_rewards else 0
    }
    
    torch.save(checkpoint, model_path)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Model saved: {model_path}")
    print(f"✓ Episodes: {training_loop.total_episodes}")
    print(f"✓ Avg reward: {checkpoint['avg_reward']:.3f}")
    print("=" * 80)
    
    # ALWAYS run evaluation (mandatory)
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION (mandatory)")
    print("=" * 80)
    
    try:
        from tools.create_evaluation_plots import HRLEvaluationPlotter
        
        plotter = HRLEvaluationPlotter(exp_dir)
        plotter.load_data()
        plotter.generate_all_plots()
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        print(f"✓ All plots saved to: {exp_dir / 'evaluation'}")
        print(f"✓ Summary saved to: {exp_dir / 'evaluation' / 'EVALUATION_SUMMARY.txt'}")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate evaluation plots: {e}")
        print("   This is a critical error - evaluation is mandatory!")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run parameterization analysis
    print("\n" + "=" * 80)
    print("RUNNING PARAMETERIZATION ANALYSIS")
    print("=" * 80)
    
    try:
        from src.utils.parameterization_analyzer import ParameterizationAnalyzer
        
        analyzer = ParameterizationAnalyzer(exp_dir)
        analyzer.generate_full_report()
        
        print(f"✓ Analysis saved to: {exp_dir / 'parameterization_results'}")
        print("=" * 80)
    except Exception as e:
        print(f"⚠️  Warning: Parameterization analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate training duration
    training_duration_seconds = getattr(training_loop, 'training_duration_seconds', 0)
    training_duration_hours = training_duration_seconds / 3600
    training_duration_formatted = f"{int(training_duration_seconds // 3600)}h {int((training_duration_seconds % 3600) // 60)}m {int(training_duration_seconds % 60)}s"
    
    # Save final summary
    summary = {
        **metadata,
        "status": "completed",
        "completion_time": datetime.now().isoformat(),
        "total_episodes": training_loop.total_episodes,
        "avg_reward": checkpoint['avg_reward'],
        "training_duration_seconds": training_duration_seconds,
        "training_duration_hours": round(training_duration_hours, 2),
        "training_duration_formatted": training_duration_formatted
    }
    
    with open(exp_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT SUMMARY: {exp_dir.name}")
    print("=" * 80)
    print(f"Results saved to: {exp_dir}/")
    print(f"  - Models:           {exp_dir / 'models'}")
    print(f"  - Logs:             {exp_dir / 'logs'}")
    print(f"  - Maps:             {exp_dir / 'maps'}")
    print(f"  - Detailed logs:    {exp_dir / 'detailed_logs'}")
    print(f"  - Evaluation:       {exp_dir / 'evaluation'}")
    print(f"  - Parameterization: {exp_dir / 'parameterization_results'}")
    print(f"  - Checkpoints:      {exp_dir / 'checkpoints'}")
    print("=" * 80)
    
    # Prominent results path for easy copy-paste
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + f"  RESULTS SAVED TO:".center(78) + "*")
    print("*" + f"  {exp_dir}".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80 + "\n")


if __name__ == '__main__':
    main()

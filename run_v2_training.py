"""
Train H1_MDP_Sim8_CentredEng_BroadNov_RespType_v2 across 3 seeds
and extract performance metrics + action diversity.

Usage:
    python run_v2_training.py
"""

import subprocess
import sys
import json
import glob
import os
import re
import numpy as np
from pathlib import Path


# ── Training config ──────────────────────────────────────────────────
SEEDS = [1, 2, 3]
EPISODES = 500
NAME = "H1_MDP_Sim8_CentredEng_BroadNov_RespType_v2"

TRAIN_CMD_TEMPLATE = [
    sys.executable, "train.py",
    "--variant", "h1",
    "--episodes", str(EPISODES),
    "--reward_mode", "baseline",
    "--centred-engagement",
    "--broadened-novelty",
    "--response-type-feature",
    "--response-type-reward",
    "--simulator", "sim8",
    "--name", NAME,
    "--seed", "{seed}",
]


def train_all_seeds():
    """Train sequentially for each seed."""
    exp_dirs = []
    for seed in SEEDS:
        cmd = [c.format(seed=seed) for c in TRAIN_CMD_TEMPLATE]
        print(f"\n{'='*80}")
        print(f"  TRAINING SEED {seed}/{len(SEEDS)}")
        print(f"{'='*80}\n")
        result = subprocess.run(cmd, cwd=os.getcwd())
        if result.returncode != 0:
            print(f"[ERROR] Seed {seed} failed with return code {result.returncode}")
            continue
        # Find the experiment directory just created
        exp_dir = find_latest_exp_dir(seed)
        if exp_dir:
            exp_dirs.append(exp_dir)
            print(f"[OK] Seed {seed} -> {exp_dir}")
    return exp_dirs


def find_latest_exp_dir(seed):
    """Find the most recently created experiment directory for this seed."""
    pattern = f"training_logs/experiments/**/*{NAME}_S{seed}_{EPISODES}ep*"
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def extract_metrics(exp_dirs):
    """Extract performance metrics and action diversity from completed runs."""
    all_metrics = []

    for exp_dir in exp_dirs:
        seed_data = {"dir": exp_dir}

        # ── Load RL metrics JSON ──
        rl_files = sorted(glob.glob(os.path.join(exp_dir, "**", "rl_metrics_*.json"), recursive=True))
        if rl_files:
            with open(rl_files[-1]) as f:
                rl = json.load(f)
            seed_data["rl_metrics"] = rl

        # ── Load learning curves JSON ──
        lc_files = sorted(glob.glob(os.path.join(exp_dir, "**", "learning_curves_*.json"), recursive=True))
        if lc_files:
            with open(lc_files[-1]) as f:
                lc = json.load(f)
            seed_data["learning_curves"] = lc

        # ── Load convergence report JSON ──
        conv_files = sorted(glob.glob(os.path.join(exp_dir, "**", "convergence_*.json"), recursive=True))
        if conv_files:
            with open(conv_files[-1]) as f:
                conv = json.load(f)
            seed_data["convergence"] = conv

        # ── Load metadata JSON ──
        meta_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(meta_file):
            with open(meta_file) as f:
                seed_data["metadata"] = json.load(f)

        all_metrics.append(seed_data)

    return all_metrics


def compute_action_diversity(metrics_list):
    """Compute action diversity percentages from RL metrics across seeds."""
    per_seed_distributions = []
    for m in metrics_list:
        rl = m.get("rl_metrics", {})
        # Try to find action distribution in metrics
        action_dist = rl.get("action_distribution", {})
        if not action_dist:
            # Fallback: check training_efficiency or other keys
            for key in rl:
                if isinstance(rl[key], dict) and "action" in key.lower():
                    action_dist = rl[key]
                    break
        if action_dist:
            per_seed_distributions.append(action_dist)
    return per_seed_distributions


def print_summary(metrics_list):
    """Print aggregated performance summary across seeds."""
    print(f"\n{'='*80}")
    print(f"  MULTI-SEED RESULTS: {NAME}")
    print(f"  Seeds: {SEEDS} | Episodes: {EPISODES}")
    print(f"{'='*80}\n")

    # ── Performance metrics ──
    rewards = []
    coverages = []
    facts = []
    exhibits = []

    for m in metrics_list:
        lc = m.get("learning_curves", {})
        rl = m.get("rl_metrics", {})
        ep_returns = lc.get("episode_returns", [])
        if ep_returns:
            # Last 100 episodes average
            last_100 = ep_returns[-100:] if len(ep_returns) >= 100 else ep_returns
            rewards.append(np.mean(last_100))

        # Extract coverage and facts from RL metrics
        perf = rl.get("performance", {})
        if perf:
            if "mean_coverage" in perf:
                coverages.append(perf["mean_coverage"])
            if "mean_facts" in perf:
                facts.append(perf["mean_facts"])
            if "mean_exhibits_covered" in perf:
                exhibits.append(perf["mean_exhibits_covered"])

    print("── Performance Metrics (last 100 episodes) ──\n")
    if rewards:
        print(f"  Avg Return:         {np.mean(rewards):>8.3f} ± {np.std(rewards):.3f}")
    if coverages:
        print(f"  Coverage Ratio:     {np.mean(coverages):>8.3f} ± {np.std(coverages):.3f}")
    if facts:
        print(f"  Facts Shared:       {np.mean(facts):>8.1f} ± {np.std(facts):.1f}")
    if exhibits:
        print(f"  Exhibits Covered:   {np.mean(exhibits):>8.1f} ± {np.std(exhibits):.1f}")

    # ── Reward components ──
    print("\n── Reward Component Breakdown ──\n")
    component_keys = [
        ("engagement", "Engagement"),
        ("novelty", "Novelty"),
        ("exhaustion", "Exhaustion Penalty"),
        ("transition", "Transition"),
        ("action_repeat", "Action Repeat Penalty"),
        ("response_type", "Response Type"),
    ]
    for key, label in component_keys:
        vals = []
        for m in metrics_list:
            rl = m.get("rl_metrics", {})
            comp = rl.get("reward_components", {})
            if key in comp:
                vals.append(comp[key])
        if vals:
            print(f"  {label:<25s} {np.mean(vals):>8.3f} ± {np.std(vals):.3f}")

    # ── Convergence ──
    print("\n── Convergence ──\n")
    conv_episodes = []
    for m in metrics_list:
        conv = m.get("convergence", {})
        ep = conv.get("episode")
        if ep:
            conv_episodes.append(ep)
    if conv_episodes:
        print(f"  Convergence Episode: {np.mean(conv_episodes):.0f} ± {np.std(conv_episodes):.0f}")
    else:
        print(f"  Convergence Episode: Not achieved / not recorded")

    # ── Action diversity ──
    print("\n── Action Diversity (%) ──\n")
    action_dists = compute_action_diversity(metrics_list)
    if action_dists:
        # Aggregate all action names
        all_actions = set()
        for d in action_dists:
            all_actions.update(d.keys())

        # Sort by mean usage descending
        action_stats = []
        for action in all_actions:
            vals = [d.get(action, 0) for d in action_dists]
            action_stats.append((action, np.mean(vals), np.std(vals)))
        action_stats.sort(key=lambda x: x[1], reverse=True)

        # Normalize to percentages if raw counts
        total = sum(s[1] for s in action_stats)
        is_counts = total > len(SEEDS) * 10  # Heuristic: if total > ~30, these are counts not percentages

        for action, mean_val, std_val in action_stats:
            if is_counts and total > 0:
                pct = mean_val / total * 100
                print(f"  {action:<35s} {pct:>6.1f}%")
            else:
                print(f"  {action:<35s} {mean_val:>6.1f}% ± {std_val:.1f}%")
    else:
        # Fallback: try to reconstruct from learning curves or checkpoints
        print("  [Action distribution not found in RL metrics.]")
        print("  Attempting to reconstruct from checkpoint metrics...")
        _reconstruct_action_diversity(metrics_list)

    print(f"\n{'='*80}")
    print(f"  Experiment directories:")
    for m in metrics_list:
        print(f"    {m['dir']}")
    print(f"{'='*80}\n")


def _reconstruct_action_diversity(metrics_list):
    """Fallback: reconstruct action counts from checkpoint metrics files."""
    for i, m in enumerate(metrics_list):
        exp_dir = m["dir"]
        checkpoint_metrics = sorted(
            glob.glob(os.path.join(exp_dir, "**", "checkpoint_*_metrics.json"), recursive=True),
            key=lambda f: int(re.search(r'ep(\d+)', f).group(1))
        )
        if checkpoint_metrics:
            # Load the last checkpoint metrics
            with open(checkpoint_metrics[-1]) as f:
                ckpt = json.load(f)
            action_counts = ckpt.get("action_counts", ckpt.get("flat_action_counts", {}))
            if action_counts:
                total = sum(action_counts.values())
                print(f"\n  Seed {SEEDS[i]}:")
                sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
                for action, count in sorted_actions:
                    pct = count / total * 100 if total > 0 else 0
                    print(f"    {action:<35s} {pct:>6.1f}%  ({count})")


if __name__ == "__main__":
    print(f"Training {NAME} with seeds {SEEDS}, {EPISODES} episodes each\n")

    exp_dirs = train_all_seeds()

    if not exp_dirs:
        print("[ERROR] No successful training runs. Exiting.")
        sys.exit(1)

    print(f"\n[OK] {len(exp_dirs)}/{len(SEEDS)} seeds completed successfully.")

    metrics_list = extract_metrics(exp_dirs)
    print_summary(metrics_list)

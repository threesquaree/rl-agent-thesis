"""
Post-training report generator for Flat MDP experiments.

Produces a human-readable training_report.txt in the experiment directory covering:
  - Training configuration
  - Learning curve (early / mid / late thirds)
  - Flat action distribution with temporal breakdown
  - Sim8 response-type diversity with temporal breakdown
  - Sample simulator utterances per response type
  - Reward component breakdown
"""

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: Dict[str, int]) -> Tuple[float, float]:
    """Return (entropy_bits, max_entropy_bits) for a frequency dict."""
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    n = len([v for v in counts.values() if v > 0])
    probs = [c / total for c in counts.values() if c > 0]
    h = -sum(p * math.log2(p) for p in probs)
    return h, math.log2(n) if n > 1 else 0.0


def _temporal_breakdown(episode_usage: List[Dict], action_keys: List[str]) -> Dict[str, Dict]:
    """
    Split episode_usage into early/mid/late thirds and compute % per action.
    Returns {action: {early: %, mid: %, late: %}}.
    """
    n = len(episode_usage)
    if n == 0:
        return {}
    third = max(1, n // 3)
    slices = {
        "early": episode_usage[:third],
        "mid":   episode_usage[third: 2 * third],
        "late":  episode_usage[2 * third:],
    }
    result = {k: {"early": 0.0, "mid": 0.0, "late": 0.0} for k in action_keys}
    for phase, eps in slices.items():
        counts: Dict[str, int] = defaultdict(int)
        for ep in eps:
            for k, v in ep.items():
                counts[k] += v
        total = sum(counts.values())
        for k in action_keys:
            result[k][phase] = (counts.get(k, 0) / total * 100) if total > 0 else 0.0
    return result


def _thirds_mean_std(values: List[float]) -> Tuple[str, str, str]:
    """Return (early_str, mid_str, late_str) mean±std for three thirds."""
    n = len(values)
    if n == 0:
        return "—", "—", "—"
    t = max(1, n // 3)
    slices = [values[:t], values[t: 2*t], values[2*t:]]
    out = []
    for s in slices:
        if s:
            m, sd = sum(s)/len(s), (sum((x - sum(s)/len(s))**2 for x in s)/len(s))**0.5
            out.append(f"{m:+.2f}±{sd:.2f}")
        else:
            out.append("—")
    return tuple(out)


# ──────────────────────────────────────────────────────────────────────────────
# Utterance sampler  (reads detailed_logs lazily)
# ──────────────────────────────────────────────────────────────────────────────

def _sample_utterances(
    detailed_logs_dir: Path,
    n_per_type: int = 3,
    max_episodes_to_scan: int = 50,
) -> Dict[str, List[str]]:
    """
    Walk a random subset of episode logs and collect visitor utterances keyed
    by response_type.  Stops once n_per_type samples are collected per type.
    """
    samples: Dict[str, List[str]] = defaultdict(list)
    if not detailed_logs_dir.exists():
        return samples

    episode_dirs = sorted(detailed_logs_dir.iterdir())
    random.shuffle(episode_dirs)
    episode_dirs = episode_dirs[:max_episodes_to_scan]

    for ep_dir in episode_dirs:
        log_file = ep_dir / "episode_log.json"
        if not log_file.exists():
            continue
        try:
            with open(log_file, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for turn in data.get("turns", []):
            dialogue = turn.get("dialogue", {})
            rtype = dialogue.get("response_type", "unknown")
            utt = dialogue.get("user_utterance", "").strip()
            if utt and len(samples[rtype]) < n_per_type:
                samples[rtype].append(utt)

        # Stop early if we have enough samples for all types
        if all(len(v) >= n_per_type for v in samples.values()) and len(samples) >= 4:
            break

    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Main report class
# ──────────────────────────────────────────────────────────────────────────────

class FlatTrainingReportGenerator:

    def __init__(self, experiment_dir: Path):
        self.exp_dir = Path(experiment_dir)
        self.metrics: Dict = {}
        self.metadata: Dict = {}

    # ── data loading ──────────────────────────────────────────────────────────

    def _load_metrics(self) -> bool:
        """Load metrics_tracker JSON (preferred) or fall back to checkpoint metrics."""
        logs_dir = self.exp_dir / "logs"
        # Prefer the full metrics_tracker file which has episode_returns + response_type_counts
        if logs_dir.exists():
            candidates = sorted(logs_dir.glob("metrics_tracker_*.json"),
                                key=lambda p: p.stat().st_mtime)
            if candidates:
                with open(candidates[-1], encoding="utf-8") as f:
                    self.metrics = json.load(f)
                return True
        # Fallback: latest checkpoint metrics file
        ckpt_dir = self.exp_dir / "checkpoints"
        candidates = []
        for d in [ckpt_dir, logs_dir]:
            if d and d.exists():
                candidates += list(d.glob("checkpoint_ep*_metrics.json"))
        if not candidates:
            return False
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest, encoding="utf-8") as f:
            self.metrics = json.load(f)
        return True

    def _load_metadata(self):
        meta_file = self.exp_dir / "metadata.json"
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                self.metadata = json.load(f)

    # ── report sections ───────────────────────────────────────────────────────

    def _section_header(self, title: str, width: int = 80) -> str:
        return "\n" + "=" * width + "\n" + title + "\n" + "=" * width

    def _config_section(self) -> str:
        rp = self.metadata.get("reward_parameters", {})
        lines = [self._section_header("TRAINING CONFIGURATION")]
        lines += [
            f"Experiment   : {self.exp_dir.name}",
            f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Episodes     : {self.metadata.get('episodes', '?')}",
            f"Max turns    : {self.metadata.get('max_turns_per_episode', '?')}",
            f"Simulator    : {self.metadata.get('simulator', '?')}",
            f"Reward mode  : {rp.get('reward_mode', '?')}",
            "",
            "Active flags :",
            f"  centred_engagement    = {rp.get('centred_engagement', False)}",
            f"  broadened_novelty     = {rp.get('broadened_novelty', False)}",
            f"  response_type_reward  = {rp.get('response_type_reward', False)}",
            f"  w_response_type       = {rp.get('w_response_type', '?')}",
            f"  alpha_new/rep/clar/ask= {rp.get('alpha_new','?')} / {rp.get('alpha_rep','?')} / {rp.get('alpha_clar','?')} / {rp.get('alpha_ask','?')}",
        ]
        return "\n".join(lines)

    def _performance_section(self) -> str:
        returns = self.metrics.get("episode_returns", [])
        lengths = self.metrics.get("episode_lengths", [])
        coverage = self.metrics.get("episode_coverage", [])
        lines = [self._section_header("TRAINING PERFORMANCE")]
        if not returns:
            return "\n".join(lines + ["  No episode data available."])

        total = len(returns)
        mean_r = sum(returns) / total
        std_r = (sum((r - mean_r)**2 for r in returns) / total) ** 0.5
        lines.append(f"  Total episodes   : {total}")
        lines.append(f"  Mean reward      : {mean_r:+.3f} ± {std_r:.3f}")
        lines.append(f"  Min / Max reward : {min(returns):+.3f} / {max(returns):+.3f}")
        if lengths:
            mean_l = sum(lengths) / len(lengths)
            lines.append(f"  Mean turns/ep    : {mean_l:.1f}")
        if coverage:
            mean_c = sum(coverage) / len(coverage)
            lines.append(f"  Mean coverage    : {mean_c*100:.1f}%")

        early, mid, late = _thirds_mean_std(returns)
        lines += [
            "",
            "  Learning trend (thirds of training):",
            f"    Early  : {early}",
            f"    Mid    : {mid}",
            f"    Late   : {late}",
        ]
        return "\n".join(lines)

    def _action_distribution_section(self) -> str:
        counts: Dict[str, int] = self.metrics.get("flat_action_counts", {})
        ep_usage: List[Dict] = self.metrics.get("episode_option_usage", [])
        lines = [self._section_header("FLAT ACTION DISTRIBUTION")]

        if not counts:
            return "\n".join(lines + ["  No action data recorded."])

        total = sum(counts.values())
        sorted_actions = sorted(counts.items(), key=lambda x: -x[1])
        breakdown = _temporal_breakdown(ep_usage, list(counts.keys()))
        entropy, max_entropy = _shannon_entropy(counts)

        # Table header
        col = 36
        lines.append(f"  {'Action':<{col}} {'Count':>7}  {'%':>6}   {'Early%':>7}  {'Mid%':>7}  {'Late%':>7}")
        lines.append("  " + "─" * 76)
        for action, cnt in sorted_actions:
            pct = cnt / total * 100
            e = breakdown.get(action, {}).get("early", 0.0)
            m = breakdown.get(action, {}).get("mid",   0.0)
            la = breakdown.get(action, {}).get("late",  0.0)
            lines.append(
                f"  {action:<{col}} {cnt:>7}  {pct:>5.1f}%   {e:>6.1f}%  {m:>6.1f}%  {la:>6.1f}%"
            )
        lines.append("  " + "─" * 76)
        lines.append(f"  {'TOTAL':<{col}} {total:>7}")
        lines += [
            "",
            f"  Shannon entropy  : {entropy:.3f} / {max_entropy:.3f} bits (max)",
            f"  Diversity ratio  : {(entropy/max_entropy*100):.1f}% of uniform" if max_entropy > 0 else "",
        ]

        # Collapse warning
        top_pct = sorted_actions[0][1] / total * 100 if sorted_actions else 0
        if top_pct > 60:
            lines.append(f"\n  ⚠  ACTION COLLAPSE DETECTED: '{sorted_actions[0][0]}' = {top_pct:.1f}% of all turns")
        elif top_pct > 40:
            lines.append(f"\n  ⚠  Dominant action: '{sorted_actions[0][0]}' = {top_pct:.1f}% (borderline)")

        return "\n".join(lines)

    def _response_type_section(self) -> str:
        counts: Dict[str, int] = self.metrics.get("response_type_counts", {})
        ep_usage: List[Dict] = self.metrics.get("episode_response_type_usage", [])
        lines = [self._section_header("SIM8 RESPONSE TYPE DIVERSITY")]

        if not counts:
            return "\n".join(lines + ["  No response type data recorded.",
                                       "  (Ensure MetricsTracker was updated with response_type tracking.)"])

        total = sum(counts.values())
        sorted_types = sorted(counts.items(), key=lambda x: -x[1])
        breakdown = _temporal_breakdown(ep_usage, list(counts.keys()))
        entropy, max_entropy = _shannon_entropy(counts)

        col = 26
        lines.append(f"  {'Response Type':<{col}} {'Count':>7}  {'%':>6}   {'Early%':>7}  {'Mid%':>7}  {'Late%':>7}")
        lines.append("  " + "─" * 70)
        for rtype, cnt in sorted_types:
            pct = cnt / total * 100
            e  = breakdown.get(rtype, {}).get("early", 0.0)
            m  = breakdown.get(rtype, {}).get("mid",   0.0)
            la = breakdown.get(rtype, {}).get("late",  0.0)
            lines.append(
                f"  {rtype:<{col}} {cnt:>7}  {pct:>5.1f}%   {e:>6.1f}%  {m:>6.1f}%  {la:>6.1f}%"
            )
        lines.append("  " + "─" * 70)
        lines.append(f"  {'TOTAL':<{col}} {total:>7}")
        lines += [
            "",
            f"  Shannon entropy  : {entropy:.3f} / {max_entropy:.3f} bits (max)",
            f"  Diversity ratio  : {(entropy/max_entropy*100):.1f}% of uniform" if max_entropy > 0 else "",
        ]

        # Engagement quality note
        engaged_types = {"acknowledgment", "follow_up_question", "question"}
        disengaged_types = {"confusion", "silence"}
        engaged_pct = sum(counts.get(t, 0) for t in engaged_types) / total * 100
        disengaged_pct = sum(counts.get(t, 0) for t in disengaged_types) / total * 100
        lines += [
            "",
            f"  Engaged responses   (ack + follow_up + question) : {engaged_pct:.1f}%",
            f"  Disengaged responses (confusion + silence)        : {disengaged_pct:.1f}%",
        ]
        return "\n".join(lines)

    def _utterance_section(self) -> str:
        detailed_dir = self.exp_dir / "detailed_logs"
        lines = [self._section_header("SAMPLE SIMULATOR UTTERANCES (Sim8)")]
        samples = _sample_utterances(detailed_dir, n_per_type=3)
        if not samples:
            lines.append("  No detailed logs found — utterance samples unavailable.")
            return "\n".join(lines)

        type_order = ["follow_up_question", "acknowledgment", "question", "statement", "confusion", "silence"]
        all_types = type_order + [t for t in samples if t not in type_order]
        for rtype in all_types:
            utterances = samples.get(rtype)
            if not utterances:
                continue
            lines.append(f"\n  [{rtype}]")
            for utt in utterances:
                # Truncate very long utterances
                display = utt if len(utt) <= 120 else utt[:117] + "..."
                lines.append(f"    • {display}")
        return "\n".join(lines)

    def _reward_section(self) -> str:
        # Reward components are stored per-episode in episode_option_usage-style lists
        # We read them directly from the metrics dict
        reward_keys = [
            ("reward_engagement",   "Engagement"),
            ("reward_novelty",      "Novelty"),
            ("reward_responsiveness", "Responsiveness"),
            ("reward_transition",   "Transition"),
            ("reward_conclude",     "Conclude"),
        ]
        lines = [self._section_header("REWARD COMPONENT BREAKDOWN")]
        ep_returns = self.metrics.get("episode_returns", [])
        if not ep_returns:
            lines.append("  No reward data available.")
            return "\n".join(lines)

        # These are stored in episode_option_usage at the episode level from the
        # FlatTrainingLoop episode_summary dict; fall back to scanning for them.
        # The metrics JSON may or may not have per-component episode lists.
        # We report what we can from value_losses / entropies as RL diagnostics.
        value_losses = self.metrics.get("value_losses", [])
        entropies = self.metrics.get("entropies", [])
        policy_losses = self.metrics.get("policy_losses", [])

        def _mean(lst):
            return sum(lst) / len(lst) if lst else 0.0

        lines += [
            f"  Mean value loss    : {_mean(value_losses):.4f}",
            f"  Mean policy loss   : {_mean(policy_losses):.4f}",
            f"  Mean entropy       : {_mean(entropies):.4f}",
            "",
            "  Entropy trend (early → mid → late):",
            "    " + "  →  ".join(_thirds_mean_std(entropies)),
        ]
        return "\n".join(lines)

    # ── public API ────────────────────────────────────────────────────────────

    def generate(self) -> str:
        self._load_metadata()
        if not self._load_metrics():
            return "ERROR: No metrics file found in experiment directory."

        sections = [
            self._config_section(),
            self._performance_section(),
            self._action_distribution_section(),
            self._response_type_section(),
            self._utterance_section(),
            self._reward_section(),
        ]
        footer = (
            "\n" + "=" * 80 + "\n"
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Experiment dir  : {self.exp_dir}\n"
            + "=" * 80
        )
        return "\n".join(sections) + footer

    def save(self, report_text: Optional[str] = None) -> Path:
        if report_text is None:
            report_text = self.generate()
        out_path = self.exp_dir / "training_report.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return out_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point (also callable from train.py)
# ──────────────────────────────────────────────────────────────────────────────

def generate_flat_training_report(experiment_dir: str) -> Path:
    gen = FlatTrainingReportGenerator(Path(experiment_dir))
    text = gen.generate()
    path = gen.save(text)
    print(text)
    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tools/flat_training_report.py <experiment_dir>")
        sys.exit(1)
    generate_flat_training_report(sys.argv[1])

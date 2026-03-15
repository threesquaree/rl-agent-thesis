#!/usr/bin/env bash
# ============================================================================
# Centred Engagement Reward Comparison: SMDP Baseline vs Centred Engagement
#
# Hypothesis: Centred engagement reward (r^ceng_t = w_e * (dwell_t - EMA))
# reduces variance from visitor-level dwell baselines, producing more stable
# gradients and faster convergence compared to raw dwell reward.
#
# Experiment design (thesis §5.1):
#   Condition A: SMDP + standard engagement   (r^eng = dwell_t × w_e)
#   Condition B: SMDP + centred engagement    (r^ceng = w_e × (dwell_t - d̄_t))
#
# All other hyperparameters held constant (seed, episodes, LR, gamma).
# ============================================================================

set -e  # Exit on any error

# ===== CONFIGURATION =====
EPISODES=${EPISODES:-300}
SEED=${SEED:-1}
TURNS=${TURNS:-20}
LR=${LR:-1e-4}
GAMMA=${GAMMA:-0.99}
SIMULATOR=${SIMULATOR:-state_machine}
EMA_ALPHA=${EMA_ALPHA:-0.1}

echo "============================================================"
echo "  Centred Engagement Reward Comparison"
echo "  Episodes: $EPISODES | Seed: $SEED | Turns: $TURNS"
echo "  Simulator: $SIMULATOR | EMA alpha: $EMA_ALPHA"
echo "============================================================"
echo ""

# ===== RUN A: SMDP Baseline (standard engagement) =====
echo ">>> [1/2] Training SMDP with STANDARD engagement reward..."
echo ""
python train.py \
    --mode hrl \
    --name smdp_baseline_eng \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --turns "$TURNS" \
    --lr "$LR" \
    --gamma "$GAMMA" \
    --simulator "$SIMULATOR" \
    --reward_mode baseline \
    --w-engagement 1.0 \
    --novelty-per-fact 1.0 \
    --experiment-type major

echo ""
echo ">>> [1/2] DONE - Standard engagement run complete."
echo ""

# ===== RUN B: SMDP Centred Engagement =====
echo ">>> [2/2] Training SMDP with CENTRED engagement reward..."
echo ""
python train.py \
    --mode hrl \
    --name smdp_centred_eng \
    --episodes "$EPISODES" \
    --seed "$SEED" \
    --turns "$TURNS" \
    --lr "$LR" \
    --gamma "$GAMMA" \
    --simulator "$SIMULATOR" \
    --reward_mode baseline \
    --w-engagement 1.0 \
    --novelty-per-fact 1.0 \
    --centred-engagement \
    --dwell-ema-alpha "$EMA_ALPHA" \
    --experiment-type major

echo ""
echo ">>> [2/2] DONE - Centred engagement run complete."
echo ""

echo "============================================================"
echo "  Both runs complete. Results in:"
echo "  training_logs/experiments/$(date +%Y%m%d)/"
echo "    smdp_baseline_eng_S${SEED}_${EPISODES}ep/"
echo "    smdp_centred_eng_S${SEED}_${EPISODES}ep/"
echo ""
echo "  Compare key metrics:"
echo "    - Episode return curves (learning speed)"
echo "    - Engagement reward component magnitude + variance"
echo "    - Coverage ratio (did shaping improve exploration?)"
echo "    - Action entropy over training (did policy collapse differ?)"
echo "============================================================"

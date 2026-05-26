#!/usr/bin/env bash
# Run full DTE pipeline test (one seed, all modalities, all models).
# Execute from repo root: bash scripts/run_dte_pipeline_test.sh
# Output is tee'd to logs/pipeline_test.log

set -uo pipefail

# Activate conda work environment (contains xgboost and tabpfn)
eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate work

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/logs/pipeline_test.log"
TRAINING="$REPO_ROOT/src/training"

TASKS=("DTE-FFX" "DTE-GNP")
MODALITIES=("Clinical" "DNA" "RNA" "Histopathology")
MODELS="lr,xgb,tabpfn"
TARGETS="orr,1yOS"
SEED=7270

# Suppress sklearn SelectKBest "k > n_features" warnings from joblib workers
export PYTHONWARNINGS="ignore::UserWarning:sklearn.feature_selection._univariate_selection"

# Scripts use relative paths; they must run from src/training/
cd "$TRAINING"

run() {
    local label="$1"; shift
    echo "" | tee -a "$LOG"
    echo ">>> $label" | tee -a "$LOG"
    if ! python "$@" 2>&1 | tee -a "$LOG"; then
        echo "!!! FAILED: $label" | tee -a "$LOG"
    fi
}

echo "========================================" | tee "$LOG"
echo "DTE Pipeline Test  $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── Step 1: Unimodal ──────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== STEP 1: UNIMODAL ===" | tee -a "$LOG"
for task in "${TASKS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
        run "unimodal $task $mod" unimodal.py \
            --modality "$mod" \
            --targets $TARGETS \
            --models $MODELS \
            --seeds $SEED \
            --task "$task" \
            --drop-treatment
    done
done

# ── Step 2: Early Fusion ──────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== STEP 2: EARLY FUSION ===" | tee -a "$LOG"
for task in "${TASKS[@]}"; do
    run "earlyfusion $task" earlyfusion.py \
        --targets $TARGETS \
        --models $MODELS \
        --seeds $SEED \
        --task "$task" \
        --drop-treatment \
        --modality-map "../../data/processed/COMPASS/modality_map.json" \
        --registry-csv "../../data/processed/COMPASS/$task/split_registry.csv" \
        --artifacts-paths-json "../../configs/artifacts_templates_${task}.json"
done

# ── Step 3: Late Fusion ───────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "=== STEP 3: LATE FUSION ===" | tee -a "$LOG"
for task in "${TASKS[@]}"; do
    for base_set in lr xgb tabpfn; do
        for stacker in lr xgb tabpfn avg; do
            run "latefusion $task base=$base_set stacker=$stacker" latefusion.py \
                --targets $TARGETS \
                --base-set $base_set \
                --stacker $stacker \
                --seeds $SEED \
                --task "$task" \
                --paths-json "../../configs/latefusion_paths_sets_${task}.json"
        done
    done
done

echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "Pipeline test done  $(date)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

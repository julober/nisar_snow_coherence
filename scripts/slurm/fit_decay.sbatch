#!/bin/bash
# Fit the temporal decay model, one spatial tile per array task. See scripts/fit_decay.py.
#
# 1) Once, on the login node (same data/fit arguments as below; prints the number of tiles N):
#      python scripts/fit_decay.py --init --zarr $ZARRS --var $VAR --output $OUT [--start-date ... --end-date ...]
# 2) sbatch --array=0-$((N-1))%50 slurm/fit_decay.sbatch      (%50 = at most 50 tasks at once)
#    Redo failed tiles only:  sbatch --array=3,17 slurm/fit_decay.sbatch
# 3) After it finishes:  python scripts/fit_decay.py --finalize --output $OUT
#
# Override the settings below at submit time, e.g.  sbatch --export=ALL,VAR=coherence_20m ...
#SBATCH --job-name=fit_decay
#SBATCH --output=/bsuhome/julialober/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1        # the fit is a single-threaded loop; parallelism comes from the array
#SBATCH --mem=8G                 # PLACEHOLDER: a 256x256 tile of all pairs is small; raise for many pairs
#SBATCH --time=03:00:00          # PLACEHOLDER: depends on pairs per pixel; check one task's runtime first
#SBATCH -p bsudfq # queue (partition)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END


set -euo pipefail
mkdir -p logs

# ---- settings (must match the --init call) ----
ZARRS="${ZARRS:-data/nisar/zarrs/all_pairs/track019/gunw_80m.zarr}"   # space-separated, one per track
VAR="${VAR:-coherence}"                                 # variable name inside the stores
OUT="${OUT:-data/nisar/zarrs/all_pairs/track019/decay_80m.zarr}"
#EXTRA="${EXTRA:---start-date 2025-12-01 --end-date 2026-02-28 }"                                          # e.g. "--start-date 2026-01-01 --end-date 2026-04-30"
EXTRA="${EXTRA:-}" 

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coherence_v2

# $ZARRS and $EXTRA are meant to split into several words
python scripts/fit_decay.py --tile-index "$SLURM_ARRAY_TASK_ID" --zarr $ZARRS --var "$VAR" --output "$OUT" $EXTRA

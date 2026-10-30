#!/bin/bash
# Gamma_thermal, gamma_spatial and gamma_temporal for one track (scripts/gamma_temporal.py): one job per track, with
# the pairs split into batches that dask runs on the CPUs of the job. Submit from the repo root, as the paths are
# relative to it. The settings are environment variables (sbatch passes the environment of the submitting shell on):
#
#   TRACK=19 FRAME=21 GUNW_ROOT=/path/to/stacks \
#   ASO=/bsuhome/julialober/scratch/coherence_data/aso/er_20260318/ASO_EastRiver_2026Mar18_snowdepth_3m.tif \
#       sbatch scripts/slurm/gamma_temporal.sbatch
#
#   TRACK      track number (required)
#   FRAME      frame number (required)
#   GUNW_ROOT  folder holding <TTT>_<A|D>_<FFF>/<ref>-<sec>_POE/GUNW_product.h5 (required)
#   DATA_DIR   folder with nisar/zarrs/... and nisar/east_river/... (default below)
#   ASO        raster whose footprint masks the cubes (default: no mask)
#   ORBIT      poe or moe, the orbit type of the GUNW folders (default poe)
#   PAIR_CHUNK pairs per batch (default 20); lower it if the job runs out of memory
# Any arguments after the file name are added to the python command, e.g. --tif-vars gamma_thermal gamma_temporal.
# If TRACK, FRAME or GUNW_ROOT is missing the job stops at once and says so in its log.
#
# Peak memory is about (CPUs) x (memory of one batch of PAIR_CHUNK pairs, 20 m grid included), so if the job runs
# out of memory lower PAIR_CHUNK (or raise --mem). The number of workers is taken from --cpus-per-task.
# Slurm opens the --output file before this script starts and does not create its folder, so /bsuhome/julialober/logs
# must already exist (the other jobs use it too).
#SBATCH --job-name=gamma_temporal
#SBATCH --output=/bsuhome/julialober/logs/%x_%A.out
#SBATCH --cpus-per-task=8        # PLACEHOLDER: batches run at the same time, one per CPU
#SBATCH --mem=64G                # PLACEHOLDER: about CPUs x one batch; check MaxRSS of a first run with sacct
#SBATCH --time=12:00:00          # PLACEHOLDER: depends on the number of pairs; check a first run
#SBATCH -p bsudfq # queue (partition)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END


set -euo pipefail

# ----------------------------- configuration -----------------------------
TRACK=${TRACK:?set TRACK, e.g. TRACK=19 sbatch scripts/slurm/gamma_temporal.sbatch}
FRAME=${FRAME:?set FRAME, e.g. FRAME=21}
GUNW_ROOT=${GUNW_ROOT:?set GUNW_ROOT to the folder holding <TTT>_<A|D>_<FFF>/<ref>-<sec>_POE}
DATA_DIR=${DATA_DIR:-/bsuhome/julialober/scratch/coherence_data}
ASO=${ASO:-}
ORBIT=${ORBIT:-poe}
PAIR_CHUNK=${PAIR_CHUNK:-20}
# --------------------------------------------------------------------------

ARGS=(--track "$TRACK" --frame "$FRAME" --gunw-root "$GUNW_ROOT" --data-dir "$DATA_DIR"
      --orbit "$ORBIT" --pair-chunk "$PAIR_CHUNK")
if [[ -n "$ASO" ]]; then ARGS+=(--aso "$ASO"); fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coherence_v2

echo "track $TRACK, frame $FRAME, GUNWs in $GUNW_ROOT, data in $DATA_DIR, ASO mask: ${ASO:-none}"
python scripts/gamma_temporal.py "${ARGS[@]}" "$@"

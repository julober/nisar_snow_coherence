#!/bin/bash
# Worker for the temporal decay fit (scripts/fit_decay.py): one spatial tile per array task.
# Don't call this directly; submit everything with the wrapper, which runs --init, submits this file as an array
# job and then once more to write the GeoTIFFs when all tiles are done:
#
#   bash scripts/slurm/submit_fit_decay.sh --zarr <stores...> --var coherence --output <out.zarr> [--start-date ...]
#
# All script arguments (everything after the file name) are passed on to fit_decay.py. Cluster settings (partition,
# time, memory, mail, logs) are the #SBATCH lines below and apply to both the tile tasks and the GeoTIFF job.
# Must be submitted from the repo root, as the paths are relative to it (the wrapper does this).
#SBATCH --job-name=fit_decay
#SBATCH --output=/bsuhome/julialober/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=1        # the fit is a single-threaded loop; parallelism comes from the array
#SBATCH --mem=8G                 # PLACEHOLDER: a 256x256 tile of all pairs is small; raise for many pairs
#SBATCH --time=03:00:00          # PLACEHOLDER: depends on pairs per pixel; check one task's runtime first
#SBATCH -p bsudfq # queue (partition)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END


set -euo pipefail
# No mkdir for the log folder here: Slurm opens the --output file before this script starts, so the wrapper
# (submit_fit_decay.sh) creates that folder at submit time.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coherence_v2

# FIT_STEP=finalize is set by the wrapper for the last job; otherwise this is a tile task of the array.
if [[ "${FIT_STEP:-}" == "finalize" ]]; then
    python scripts/fit_decay.py --finalize "$@"
else
    python scripts/fit_decay.py --tile-index "$SLURM_ARRAY_TASK_ID" "$@"
fi

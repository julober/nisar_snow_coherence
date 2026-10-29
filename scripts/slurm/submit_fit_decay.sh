#!/bin/bash
# Submit the whole temporal decay fit in one command: create the output Zarr, run one Slurm array task per tile,
# then write the GeoTIFFs once every tile has finished.
#
# Takes exactly the arguments of scripts/fit_decay.py (see its --help), so the pair selection is written only once:
#
#   bash scripts/slurm/submit_fit_decay.sh \
#       --zarr data/nisar/zarrs/all_pairs/track019/gunw_80m.zarr data/nisar/zarrs/all_pairs/track128/gunw_80m.zarr \
#       --var coherence --output data/nisar/zarrs/all_pairs/decay_80m.zarr \
#       --start-date 2025-12-01 --end-date 2026-02-28 --max-days 48
#
# Run it on the login node. Options for this wrapper are environment variables (so they cannot clash with the
# fit_decay.py flags):
#   MAX_CONCURRENT=50   array tasks allowed to run at once
#   SBATCH_OPTS="..."   extra sbatch flags for both jobs, e.g. "--mem=16G --time=06:00:00"
#   NO_FINALIZE=1       do not queue the GeoTIFF job
#   RETRY=3,17          only rerun these tiles (the output Zarr must already exist); the GeoTIFF job waits for them
#   FORCE=1             start over even though the output Zarr exists (this DELETES the previous fit)
# Partition, time, memory and mail come from the #SBATCH lines in fit_decay.sbatch.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root: the worker's relative paths expect it

WORKER=scripts/slurm/fit_decay.sbatch
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
SBATCH_OPTS="${SBATCH_OPTS:-}"

if [[ $# -eq 0 ]]; then
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
    exit 1
fi

# Find --output in the arguments, to check it and to tell the user where results go.
OUTPUT=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--output" ]]; then OUTPUT="${args[i+1]:-}"; fi
    if [[ "${args[i]}" == --output=* ]]; then OUTPUT="${args[i]#--output=}"; fi
done
[[ -n "$OUTPUT" ]] || { echo "error: --output is required" >&2; exit 1; }

# Slurm does not create the folder of the log file, and the job fails silently without it. Take the folder from the
# worker's "#SBATCH --output=..." line so that the two cannot disagree.
LOG_DIR="$(dirname "$(sed -n 's/^#SBATCH --output=//p' "$WORKER")")"
mkdir -p "$LOG_DIR"
echo "logs: $LOG_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coherence_v2

if [[ -n "${RETRY:-}" ]]; then
    [[ -e "$OUTPUT" ]] || { echo "error: RETRY set but $OUTPUT does not exist; run without RETRY first" >&2; exit 1; }
    ARRAY="$RETRY"
else
    # --init would overwrite a finished fit, so refuse unless asked.
    if [[ -e "$OUTPUT" && -z "${FORCE:-}" ]]; then
        echo "error: $OUTPUT already exists. Use RETRY=<tiles> to rerun tiles, or FORCE=1 to start over." >&2
        exit 1
    fi
    python scripts/fit_decay.py --init "$@" >/dev/null
    N_TILES=$(python scripts/fit_decay.py --n-tiles "$@" | tail -n 1)
    ARRAY="0-$((N_TILES - 1))%${MAX_CONCURRENT}"
    echo "$N_TILES tiles, results in $OUTPUT"
fi

# SBATCH_OPTS is meant to split into several words.
# shellcheck disable=SC2086
JOB=$(sbatch --parsable --array="$ARRAY" $SBATCH_OPTS "$WORKER" "$@")
echo "submitted tile array $JOB (--array=$ARRAY)"

if [[ -n "${NO_FINALIZE:-}" ]]; then
    echo "NO_FINALIZE set: run 'python scripts/fit_decay.py --finalize --output $OUTPUT' when the array has finished"
    exit 0
fi
# afterok: the GeoTIFFs are only written if every tile succeeded. If one fails, rerun it with RETRY=<tile>.
# shellcheck disable=SC2086
FINAL=$(sbatch --parsable --job-name=fit_decay_tifs --time=00:30:00 --dependency=afterok:"$JOB" \
        --export=ALL,FIT_STEP=finalize $SBATCH_OPTS "$WORKER" "$@")
echo "submitted GeoTIFF job $FINAL (starts after $JOB succeeds)"

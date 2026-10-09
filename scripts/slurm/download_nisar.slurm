#!/bin/bash
#SBATCH -J download_nisar # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log # output and error file name
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 12:00:00 # run time (hh:mm:ss)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END
#SBATCH --array=1-1 # one task per row of tracks_frames.txt; update when adding rows

# ----------------------------- configuration -----------------------------
REPO=/bsuhome/julialober/nisar_snow_coherence            # TODO: repo location on the cluster
PYTHON=/bsuhome/julialober/miniforge3/envs/nisar_snow_coherence/bin/python  # TODO: env name
ENV_FILE=/bsuhome/julialober/.env                        # TODO: file exporting EARTHDATA_TOKEN
TRACK_FILE=$REPO/scripts/slurm/tracks_frames.txt
OUT_ROOT=/bsuhome/julialober/scratch/coherence_data/nisar

START=2025-12-01
END=2026-04-30
CRID=P05023
POL=HH

GUNW_LAYERS=(unwrapped_phase coherence_unwrapped   # 80 m interferogram
             wrapped_phase coherence_wrapped       # 20 m interferogram
             incidence_angle perpendicular_baseline
             reference_slant_range secondary_slant_range)  # for spatial coherence
GSLC_LAYERS=(backscatter NEBZ)                     # for thermal coherence
# --------------------------------------------------------------------------

echo "Starting."
mkdir -p /bsuhome/julialober/logs

# earthaccess.login() reads EARTHDATA_TOKEN from the environment
if [ -z "$EARTHDATA_TOKEN" ] && [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Pick the Nth non-comment, non-blank line (N = Slurm array task id)
LINE=$(grep -v -E '^\s*(#|$)' "$TRACK_FILE" | sed -n "${SLURM_ARRAY_TASK_ID}p")
read -r TRACK FRAME AOI <<< "$LINE"
if [ -z "$TRACK" ] || [ -z "$FRAME" ] || [ -z "$AOI" ]; then
    echo "No valid 'track frame aoi' on task line ${SLURM_ARRAY_TASK_ID} of $TRACK_FILE" >&2
    exit 1
fi

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting download for track $TRACK, frame $FRAME (AOI: $AOI)"

$PYTHON "$REPO/scripts/slurm/download_nisar_track.py" "$TRACK" "$FRAME" \
    --aoi "$AOI" --start "$START" --end "$END" --crid "$CRID" --pol "$POL" \
    --out-root "$OUT_ROOT" \
    --gunw-layers "${GUNW_LAYERS[@]}" \
    --gslc-layers "${GSLC_LAYERS[@]}"

echo "Done"

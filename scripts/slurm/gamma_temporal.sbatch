#!/bin/bash
# Gamma_thermal, gamma_spatial and gamma_temporal for one track (scripts/gamma_temporal.py): one job per track, with
# the pairs split into batches that dask runs on the CPUs of the job. All script arguments (everything after the file
# name) are passed on to gamma_temporal.py; see its --help. Submit from the repo root, as the paths are relative to it:
#
#   sbatch scripts/slurm/gamma_temporal.sbatch --track 19 --frame 21 \
#       --aso /bsuhome/julialober/scratch/coherence_data/aso/er_20260318/ASO_EastRiver_2026Mar18_snowdepth_3m.tif \
#       --data-dir /bsuhome/julialober/scratch/coherence_data
#
# Peak memory is about (CPUs) x (memory of one batch of --pair-chunk pairs, 20 m grid included), so if the job runs
# out of memory lower --pair-chunk (or raise --mem). The number of workers is taken from --cpus-per-task.
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coherence_v2

python scripts/gamma_temporal.py "$@"

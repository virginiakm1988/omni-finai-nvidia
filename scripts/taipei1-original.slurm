#!/bin/bash
#SBATCH --job-name=omni-finai-original
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=omni-finai-%j.log
set -euo pipefail
# Select the current partition/account when submitting; do not use a login GPU.
# AUTOMODEL_ENV is the absolute path of the prepared, isolated Python environment.
: "${AUTOMODEL_ENV:?Set AUTOMODEL_ENV to the prepared tutorial environment}"
source "$AUTOMODEL_ENV/bin/activate"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
python -m project_demo prepare --output-dir runs/preparation
python -m project_demo train --mode live --profile original --gpus 4 --output-dir runs/automodel-original

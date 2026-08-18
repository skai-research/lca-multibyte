#!/bin/bash
#SBATCH --account <your-slurm-account>
#SBATCH --job-name mtpf-train
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --time=30:30:00
#SBATCH --partition <your-slurm-partition>

nvidia-smi

: "${WANDB_API_KEY:?set WANDB_API_KEY (or pass --with_tracking False below)}"
# Optional: W&B org to log into; defaults to your personal account.
# export WANDB_ENTITY=<your-wandb-entity>
: "${HF_TOKEN:?set HF_TOKEN for gated Hugging Face datasets/models}"

export PYTHONPATH=$(pwd)

# export HF_HOME=/path/to/cache
# export WANDB_CACHE_DIR=/path/to/cache

ulimit -c 0  


EXP_DIR=/path/to/experiments

# Training config. See configs/train/ for the full list.
CONFIG_FILE=configs/train/modern_fxt_priors_0.3_en_lca_prev_group_self_256_scale_bp_dualhead.yaml

GPUS=2

# Guard against the SBATCH allocation and GPUS drifting apart.
SBATCH_GPUS=$(grep -oP '#SBATCH --gpus=\K[0-9]+' "$0")
if [ -n "$SBATCH_GPUS" ] && [ "$GPUS" != "$SBATCH_GPUS" ]; then
    echo "ERROR: GPUS=$GPUS does not match #SBATCH --gpus=$SBATCH_GPUS" >&2
    exit 1
fi

accelerate_config_file=configs/accelerate/gpu_$GPUS.yaml
export TORCH_DISTRIBUTED_DEBUG=OFF
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000
export NCCL_TIMEOUT=3600

echo 'Run training...'
echo 'Finding free port'
PORT=$(uv run python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# To resume, append: --resume_from_checkpoint <EXP_DIR>/<run-name>/step_<N>
uv run accelerate launch \
    --main_process_port=$PORT \
    --config_file=$accelerate_config_file \
    --num_processes="$GPUS" \
    src/train/train.py \
    --config_file "$CONFIG_FILE" \
    --exp_dir "$EXP_DIR" \
    --with_tracking True

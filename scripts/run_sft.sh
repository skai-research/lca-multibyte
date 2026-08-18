#!/bin/bash
#SBATCH --account <your-slurm-account>
#SBATCH --job-name mtpf-sft
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=17:30:00
#SBATCH --partition <your-slurm-partition>

nvidia-smi

: "${WANDB_API_KEY:?set WANDB_API_KEY (or pass --with_tracking False below)}"
# Optional: W&B org to log into; defaults to your personal account.
# export WANDB_ENTITY=<your-wandb-entity>

export PYTHONPATH=$(pwd)

# export HF_HOME=/path/to/cache
# export WANDB_CACHE_DIR=/path/to/cache

# Fine-tuning config. See configs/finetune/ for the full list
# (e.g. tulu_sft.yml, summarization_sft.yml, translation_sft.yml).
CONFIG_FILE=configs/finetune/tulu_sft.yml

GPUS=1
accelerate_config_file=configs/accelerate/gpu_$GPUS.yaml

# Pretrained checkpoints to fine-tune. Each entry is a run directory written by
# scripts/run_train.sh, optionally with a /step_<N> suffix to pick a checkpoint.
MODELS=(
    /path/to/pretrained/run_dir
)

# Where fine-tuned checkpoints are written.
WORK_DIR=/path/to/experiments/downstream

SEEDS=(42)
LRS=(2e-4)
LANGS=(en)
DATASET="tulu"   # tulu | cnn_dailymail | samsum | squad | xlsum | opus-100
BSZ=16
gradient_accumulation_steps=4

for model in "${MODELS[@]}"; do
    echo "Starting with model ${model}"
    for SEED in "${SEEDS[@]}"; do
        echo "Starting with seed ${SEED}"
        for LR in "${LRS[@]}"; do
            for language in "${LANGS[@]}"; do
                echo 'Finding free port'
                PORT=$(uv run python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
                uv run accelerate launch \
                    --main_process_port=$PORT \
                    --config_file=$accelerate_config_file \
                    --num_processes="$GPUS" \
                    src/finetune/sft.py \
                    --config_file "$CONFIG_FILE" \
                    --pretrained_path "${model}" \
                    --work_dir "$WORK_DIR" \
                    --language $language \
                    --lr $LR \
                    --seed $SEED \
                    --dataset_name $DATASET \
                    --freeze_bp False \
                    --use_best_model False \
                    --gradient_accumulation_steps $gradient_accumulation_steps \
                    --batch_size $BSZ \
                    --scale_loss2 2.0 \
                    --scale_bp 10 \
                    --with_tracking True
            done
        done
    done
done

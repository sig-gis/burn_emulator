#!/bin/bash -l
#SBATCH --job-name=train_cnns
#SBATCH --output=/mnt/data/mtruong/burn_emulator/logs/train_cnns_%A_%a.out
#SBATCH --error=/mnt/data/mtruong/burn_emulator/logs/train_cnns_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=31000
#SBATCH --nodelist=dragon05
#SBATCH --array=0-7%4



echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"

source $HOME/.bashrc
cd /mnt/data/mtruong/burn_emulator/logs
source /mnt/data/mtruong/burn_emulator/.venv/bin/activate

CONFIG_DIR=/mnt/data/mtruong/burn_emulator/data/configs

MODEL_CONFIGS=(
    "circle"
    "unet"
)

DATA_CONFIGS=(
    "basemax_480"
    "basemax_720"
    "basemax_960"
    "basemax_all"
)

MID=$(( SLURM_ARRAY_TASK_ID % 2 ))
DID=$(( SLURM_ARRAY_TASK_ID / 2 ))

MODEL_CFG=${MODEL_CONFIGS[$MID]}
DATA_CFG=${DATA_CONFIGS[$DID]}

echo "Using model config: $MODEL_CFG" "$DATA_CFG"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    (
        while true; do
            free -h > /mnt/data/mtruong/burn_emulator/logs/train_cnns.out
            nvidia-smi >> /mnt/data/mtruong/burn_emulator/logs/train_cnns.out
            echo "=== $(date) ===" >> /mnt/data/mtruong/burn_emulator/logs/train_cnns.out
            sleep 60
        done
    ) &
    MONITOR_PID=$!
fi

burn_emulator -m train -c "$CONFIG_DIR/$DATA_CFG/$MODEL_CFG.yaml" -c "$CONFIG_DIR/$DATA_CFG/train.yaml"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Job finished at $(date)"
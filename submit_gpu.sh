#!/bin/sh

#SBATCH -c 2 # Number of cores requested
#SBATCH -t 0-06:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=250G # Memory per node
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --account=kempner_dam_lab

# Create slurm_out directory if it doesn't exist
mkdir -p slurm_out

# Load modules (adjust based on your cluster)
module purge
module load Mambaforge
module load cuda cudnn
mamba activate hiergenv

# Print hostname and GPU info
hostname
nvidia-smi

# Set random seed (uncomment to use different seeds)
# seeds=(42 43 44 45 46 47 48 49 50 51)
# for seed in "${seeds[@]}"
# do
#   # Run training for each seed
# done

# Training parameters
dataset=qf_disamb
lr=1e-4
seed=42

# Train transformer on question formation task
python train_transformers.py \
  --callback \
  --batch_size 8 \
  --dataset question_${dataset} \
  --disamb_num 0 \
  --max_train_steps 300000 \
  --encoder_n_layers 6 \
  --lr ${lr} \
  --num_warmup_steps 10000 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --eval_every 2000 \
  --seed ${seed} \
  --tied-embedding \
  --run_name ${dataset}_lr${lr}_seed${seed}  \
  --save_every 100000 \
  --save_dir ./checkpoints/ \
  --save_prefix question_${dataset}_lr${lr}

# Alternative configurations (commented out):

# Exclude simple declarative sentences (all questions + complex declaratives)
# --exclude_simple_decls \

# Exclude complex declarative sentences (all questions + simple declaratives)
# --exclude_complex_decls \

# Train on questions only
# --exclude_identity \

# Load from checkpoint
# --model_load_path ./checkpoints/question_qf_disamb_lr1e-4_12345/checkpoint_100000.pth \

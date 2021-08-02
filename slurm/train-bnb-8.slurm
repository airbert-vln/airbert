#!/bin/bash
#SBATCH --job-name=bnb
#SBATCH --output=logs/bnb-%j.out
#SBATCH --error=logs/bnb-%j.err
#SBATCH --gres gpu:4
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 4
#SBATCH -C v100-32g
#SBATCH -c 10
#SBATCH -A vuo@gpu
#SBATCH --hint nomultithread
#SBATCH --time 20:00:00
#SBATCH --exclusive
#SBATCH --qos=qos_gpu-t3

export TRANSFORMERS_CACHE=$HOME/.cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1

set -x
pretrained=${pretrained:-$work/model_zoos/vlnbert.123mine.bin}
name=${name:-BNB-$SLURM_JOBID}
prefix=${prefix:-""}
num_epochs=${num_epochs:-40}
warmup_proportion=${warmup_proportion:-0.04}
num_workers=${num_workers:-10}
num_negatives=${num_negatives:-3}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
batch_size=${batch_size:-64}

filename=data/runs/run-$name/.torch_distributed_sync
[ -f $filename ] && rm filename


srun \
  --output logs/bnb-%j.out \
  --error logs/bnb-%j.err \
  --export ALL \
  python train_bnb.py \
    --world_size=$SLURM_NTASKS \
    --from_pretrained $pretrained \
    --seed 1 \
    --save_name $name \
    --num_workers $num_workers \
    --num_epochs $num_epochs \
    --warmup_proportion $warmup_proportion \
    --num_negatives $num_negatives \
    --cooldown_factor 8 \
    --prefix "$prefix" \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    $args

set +x

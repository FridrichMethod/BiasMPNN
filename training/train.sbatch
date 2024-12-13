#!/bin/bash
#SBATCH -J biasmpnn
#SBATCH -o biasmpnn_%j.out
#SBATCH -e biasmpnn_%j.err
#SBATCH -p gpu_4l
#SBATCH -A lhlai_g1
#SBATCH -q lhlaig4c
#SBATCH -N 1
#SBATCH -c 9
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

source /appsnew/source/cuda-12.6.2.sh
source activate /lustre1/lhlai_pkuhpc/lzy/miniconda3/envs/pdmodels

nvidia-smi

python ../train.py &

sleep 180s
nvidia-smi
sleep 60s
nvidia-smi

wait

#!/bin/bash -l
#SBATCH -o /home/o/omrosa/research/experiment6/logs/std_out_%j
#SBATCH -e /home/o/omrosa/research/experiment6/logs/std_err_%j
#SBATCH -p Quick
#SBATCH --time=24:00:00
#SBATCH --nodelist=GPU45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=resnet_canny

source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment6/canny

srun python train_canny.py \
    --real-dir /general/omrosa/FFHQ/canny_reals \
    --fake-dir /general/omrosa/FFHQ/canny_fakes \
    --output-dir results/resnet_canny \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --seed 42

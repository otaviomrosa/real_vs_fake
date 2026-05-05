#!/bin/bash -l
#SBATCH -o /home/o/omrosa/research/experiment6/vit/logs/vit_segm_%j.out
#SBATCH -e /home/o/omrosa/research/experiment6/vit/logs/vit_segm_%j.err
#SBATCH -p Quick
#SBATCH --time=24:00:00
#SBATCH --exclude=GPU41
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=vit_segm

export PYTHONNOUSERSITE=1
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment6/vit

srun python train_vit.py \
    --experiment segmentation \
    --real-dir /data/omrosa/FFHQ/segm_reals/resnet18 \
    --fake-dir /data/omrosa/FFHQ/segm_fakes/resnet18 \
    --output-base /home/o/omrosa/research/experiment6/vit/results \
    --epochs 31 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --patch-size 16 \
    --dim 512 \
    --depth 6 \
    --heads 8 \
    --mlp-dim 1024 \
    --seed 42 \
    --num-workers 0 \
    --resume /home/o/omrosa/research/experiment6/vit/results/vit_segmentation/final_model.pth 

#!/bin/bash
#SBATCH --job-name=FFHQ_Dataset
#SBATCH --output=/general/omrosa/FFHQ/generate_%j.log
#SBATCH --partition=Quick
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 1. Load environment
# Assuming your conda is initialized in your .bashrc
source ~/.bashrc
conda activate celeba

cd /home/o/omrosa/research/experiment6/stylegan2-ada-pytorch

# This creates a ZIP file where every image is resized using the Box filter
python dataset_tool.py --source=/data/omrosa/FFHQ/images1024x1024 \
    --dest=/general/omrosa/FFHQ/reals_256_dataset.zip \
    --width=256 --height=256 --resize-filter=box

OUTDIR="/general/omrosa/FFHQ/reals_256"
mkdir -p $OUTDIR
unzip -q /general/omrosa/FFHQ/reals_256_dataset.zip -d $OUTDIR

echo "Generation complete. Files are in $OUTDIR"

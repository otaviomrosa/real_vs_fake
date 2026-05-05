#!/bin/bash
#SBATCH --job-name=FFHQ_Gen_Fakes
#SBATCH --output=/general/omrosa/FFHQ/generate_%j.log
#SBATCH --partition=Quick
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPU41
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 1. Load environment
# Assuming your conda is initialized in your .bashrc
source ~/.bashrc
conda activate celeba

# 2. Navigate to the StyleGAN2-ADA-PyTorch directory
# Replace this path if you cloned the repo elsewhere
cd /home/o/omrosa/research/experiment6/stylegan2-ada-pytorch

# 3. Create the output directory
OUTDIR="/general/omrosa/FFHQ/fakes_256"
mkdir -p $OUTDIR

# 4. Run Generation
# Using seeds 0-69999 to get 70k unique images
# Using trunc=1.0 to ensure the "Neural Turing Test" evaluates the full distribution
python generate.py --outdir=$OUTDIR --trunc=1 --seeds=0-69999 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq70k-paper256-noaug.pkl

echo "Generation complete. Files are in $OUTDIR"

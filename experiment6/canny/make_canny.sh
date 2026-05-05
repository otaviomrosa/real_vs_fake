#!/bin/bash
#SBATCH --job-name=FFHQ_Canny_Blur
#SBATCH --output=/general/omrosa/FFHQ/canny_%j.log
#SBATCH --partition=Quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Load environment
source ~/.bashrc
conda activate celeba

# Move to research folder
cd /home/o/omrosa/research/experiment6

# Execute the python script
python generate_canny.py

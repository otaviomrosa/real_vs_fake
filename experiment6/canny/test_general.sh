#!/bin/bash -l
#SBATCH -p Quick
#SBATCH --gpus=1
#SBATCH --mem=4GB
#SBATCH --job-name=test_general
#SBATCH -o /home/o/omrosa/research/experiment6/canny/logs/test_general_%j.out
#SBATCH -e /home/o/omrosa/research/experiment6/canny/logs/test_general_%j.err

echo "Node: $(hostname)"
ls /general/omrosa/FFHQ/canny_fakes | head -5 && echo "SUCCESS: /general is accessible" || echo "FAILED: /general not accessible"
EOF
sbatch test_general.sh

"""Generate SLURM job scripts for all Experiment 6 ablation experiments.

Run this on the GAIVI login node after uploading train_ablation.py and sanity_check.py:
    cd /home/o/omrosa/research/experiment6
    python generate_ablation_jobs.py

Then submit with the rolling 6-job window:
    python ablations/submit_all.py
"""

import os
from pathlib import Path

BASE = Path('/home/o/omrosa/research/experiment6')
ABLATION_DIR = BASE / 'ablations'
LOG_DIR   = ABLATION_DIR / 'logs'
SLURM_DIR = ABLATION_DIR / 'slurm_scripts'
RESULT_DIR = ABLATION_DIR / 'results'

REAL_DIRS = {
    'canny':        '/data/omrosa/FFHQ/canny_reals',
    'segmentation': '/data/omrosa/FFHQ/segm_reals/resnet18',
}
FAKE_DIRS = {
    'canny':        '/data/omrosa/FFHQ/canny_fakes',
    'segmentation': '/data/omrosa/FFHQ/segm_fakes/resnet18',
}

ABLATIONS = {
    'crop':       [64, 16, 4, 1],
    'downsample': [64, 16, 4, 1],
    'pca':        [2000, 500, 50, 10],
    'blur':       [5, 10, 20, 40],
}

ARCHS = ['resnet', 'densenet', 'vit']
REPS  = ['canny', 'segmentation']

# ViT PCA jobs may exceed 24h — use general partition (7-day)
def get_partition(arch, ablation):
    if arch == 'vit' and ablation == 'pca':
        return 'general', '72:00:00'
    return 'Quick', '24:00:00'


CNN_TEMPLATE = """\
#!/bin/bash -l
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -e {log_dir}/{job_name}_%j.err
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH --exclude=GPU41,GPU45,GPU46
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name={job_name}

export PYTHONNOUSERSITE=1
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment6

srun python train_ablation.py \\
    --arch {arch} \\
    --representation {rep} \\
    --ablation {ablation} \\
    --param {param} \\
    --real-dir {real_dir} \\
    --fake-dir {fake_dir} \\
    --output-base {result_dir} \\
    --epochs 50
"""

SANITY_TEMPLATE = """\
#!/bin/bash -l
#SBATCH -o {log_dir}/sanity_check_%j.out
#SBATCH -e {log_dir}/sanity_check_%j.err
#SBATCH -p Quick
#SBATCH --time=24:00:00
#SBATCH --exclude=GPU41,GPU45,GPU46
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=sanity_check

export PYTHONNOUSERSITE=1
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment6

srun python sanity_check.py \\
    --real-dir /data/omrosa/FFHQ/reals_256 \\
    --output-dir {result_dir}/sanity_check
"""


def main():
    for d in (LOG_DIR, SLURM_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)

    scripts = []

    # ViT jobs first (longer runtime — start them early in the queue)
    for arch in ['vit', 'densenet', 'resnet']:
        for rep in REPS:
            for ablation, params in ABLATIONS.items():
                for param in params:
                    job_name = f'{arch}_{rep}_{ablation}_{param}'
                    partition, time_limit = get_partition(arch, ablation)
                    script_path = SLURM_DIR / f'{job_name}.sh'
                    content = CNN_TEMPLATE.format(
                        log_dir=LOG_DIR,
                        job_name=job_name,
                        partition=partition,
                        time=time_limit,
                        arch=arch,
                        rep=rep,
                        ablation=ablation,
                        param=param,
                        real_dir=REAL_DIRS[rep],
                        fake_dir=FAKE_DIRS[rep],
                        result_dir=RESULT_DIR,
                    )
                    script_path.write_text(content)
                    scripts.append((job_name, str(script_path)))

    # Sanity check last
    sanity_path = SLURM_DIR / 'sanity_check.sh'
    sanity_path.write_text(SANITY_TEMPLATE.format(
        log_dir=LOG_DIR, result_dir=RESULT_DIR
    ))
    scripts.append(('sanity_check', str(sanity_path)))

    print(f'Generated {len(scripts)} SLURM scripts in {SLURM_DIR}')
    print(f'  Training jobs : {len(scripts) - 1}')
    print(f'  Sanity check  : 1')

    # Write the job list for submit_all.py
    job_list_path = ABLATION_DIR / 'job_list.txt'
    with open(job_list_path, 'w') as f:
        for name, path in scripts:
            f.write(f'{name}\t{path}\n')
    print(f'Job list written: {job_list_path}')
    print()
    print('Next step:')
    print(f'  python {ABLATION_DIR}/submit_all.py')


if __name__ == '__main__':
    main()

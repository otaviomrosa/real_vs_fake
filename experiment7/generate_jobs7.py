"""Generate SLURM job scripts for Experiment 7 ablations on raw RGB FFHQ images.

Run on the GAIVI login node:
    cd /home/o/omrosa/research/experiment7
    python generate_jobs7.py

Then submit:
    python ablations/submit_all.py
"""

import os
from pathlib import Path

BASE     = Path('/home/o/omrosa/research/experiment7')
ABL_DIR  = BASE / 'ablations'
LOG_DIR  = ABL_DIR / 'logs'
SLURM_DIR = ABL_DIR / 'slurm_scripts'
RESULT_DIR = ABL_DIR / 'results'

REAL_DIR = '/data/omrosa/FFHQ/reals_256'
FAKE_DIR = '/data/omrosa/FFHQ/fakes_256'

ABLATIONS = {
    'crop':       [64, 16, 4, 1],
    'downsample': [64, 16, 4, 1],
    'pca':        [2000, 500, 50, 10],
    'blur':       [5, 10, 20, 40],
}

ARCHS = ['vit', 'densenet', 'resnet']  # vit first (longer runtime)


def get_partition(arch, ablation):
    if arch == 'vit' and ablation == 'pca':
        return 'general', '72:00:00'
    return 'Quick', '24:00:00'


JOB_TEMPLATE = """\
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

cd /home/o/omrosa/research/experiment7

srun python train_ablation7.py \\
    --arch {arch} \\
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
#SBATCH --job-name=sanity_check7

export PYTHONNOUSERSITE=1
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment7

srun python sanity_check7.py \\
    --real-dir {real_dir} \\
    --output-dir {result_dir}/sanity_check
"""


def main():
    for d in (LOG_DIR, SLURM_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)

    scripts = []

    for arch in ARCHS:
        for ablation, params in ABLATIONS.items():
            for param in params:
                job_name = f'{arch}_{ablation}_{param}'
                partition, time_limit = get_partition(arch, ablation)
                script_path = SLURM_DIR / f'{job_name}.sh'
                content = JOB_TEMPLATE.format(
                    log_dir=LOG_DIR,
                    job_name=job_name,
                    partition=partition,
                    time=time_limit,
                    arch=arch,
                    ablation=ablation,
                    param=param,
                    real_dir=REAL_DIR,
                    fake_dir=FAKE_DIR,
                    result_dir=RESULT_DIR,
                )
                script_path.write_text(content)
                scripts.append((job_name, str(script_path)))

    sanity_path = SLURM_DIR / 'sanity_check7.sh'
    sanity_path.write_text(SANITY_TEMPLATE.format(
        log_dir=LOG_DIR, real_dir=REAL_DIR, result_dir=RESULT_DIR
    ))
    scripts.append(('sanity_check7', str(sanity_path)))

    print(f'Generated {len(scripts)} SLURM scripts')
    print(f'  Training jobs: {len(scripts) - 1}')
    print(f'  Sanity check:  1')

    job_list = ABL_DIR / 'job_list.txt'
    with open(job_list, 'w') as f:
        for name, path in scripts:
            f.write(f'{name}\t{path}\n')
    print(f'Job list: {job_list}')
    print()
    print(f'Next: python {ABL_DIR}/submit_all.py')


if __name__ == '__main__':
    main()

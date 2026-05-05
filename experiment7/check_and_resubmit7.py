"""Check which Experiment 7 jobs are missing or incomplete, then submit them.

A job is considered done if its training_log.json exists and epochs_completed
equals total_epochs. Jobs already in the SLURM queue are skipped.
Partial results trigger --resume so training picks up from the last checkpoint.

Run on the GAIVI login node:
    cd /home/o/omrosa/research/experiment7
    python check_and_resubmit7.py [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RESULT_BASE = Path('/home/o/omrosa/research/experiment7/ablations/results')
LOG_DIR     = Path('/home/o/omrosa/research/experiment7/ablations/logs')
TRAIN_SCRIPT = '/home/o/omrosa/research/experiment7/train_ablation7.py'
REAL_DIR    = '/data/omrosa/FFHQ/reals_256'
FAKE_DIR    = '/data/omrosa/FFHQ/fakes_256'
TOTAL_EPOCHS = 50
MAX_CONCURRENT = 6

ARCHS = ['resnet', 'densenet', 'vit']
ABLATIONS = {
    'crop':       [64, 16, 4, 1],
    'downsample': [64, 16, 4, 1],
    'pca':        [2000, 500, 50, 10],
    'blur':       [5, 10, 20, 40],
}


def param_str(param):
    return str(int(param)) if param == int(param) else str(param)


def job_name(arch, ablation, param):
    return f'{arch}_{ablation}_{param_str(param)}'


def output_dir(arch, ablation, param):
    return RESULT_BASE / arch / f'{ablation}_{param_str(param)}'


def is_complete(arch, ablation, param):
    log = output_dir(arch, ablation, param) / 'training_log.json'
    if not log.exists():
        return False
    try:
        data = json.loads(log.read_text())
        return data.get('epochs_completed', 0) >= data.get('total_epochs', TOTAL_EPOCHS)
    except Exception:
        return False


def has_partial(arch, ablation, param):
    """True if a checkpoint exists to resume from."""
    return (output_dir(arch, ablation, param) / 'final_model.pth').exists()


def queued_job_names():
    """Return set of job names currently in the SLURM queue (running or pending)."""
    result = subprocess.run(
        ['squeue', '-u', 'omrosa', '-h', '--format=%j'],
        capture_output=True, text=True
    )
    return set(result.stdout.split())


def make_slurm_script(arch, ablation, param):
    name = job_name(arch, ablation, param)
    odir = output_dir(arch, ablation, param)

    if arch == 'vit' and ablation == 'pca':
        partition, time_limit = 'general', '72:00:00'
    else:
        partition, time_limit = 'Quick', '24:00:00'

    resume_flag = f'--resume {odir}/final_model.pth' if has_partial(arch, ablation, param) else ''

    return f"""\
#!/bin/bash -l
#SBATCH -o {LOG_DIR}/{name}_%j.out
#SBATCH -e {LOG_DIR}/{name}_%j.err
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH --exclude=GPU41,GPU45,GPU46
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name={name}

export PYTHONNOUSERSITE=1
source /apps/anaconda3/etc/profile.d/conda.sh
conda activate celeba

cd /home/o/omrosa/research/experiment7

srun python {TRAIN_SCRIPT} \\
    --arch {arch} \\
    --ablation {ablation} \\
    --param {param} \\
    --real-dir {REAL_DIR} \\
    --fake-dir {FAKE_DIR} \\
    --output-base {RESULT_BASE} \\
    --epochs {TOTAL_EPOCHS} \\
    {resume_flag}
"""


def sbatch(script_text, dependency_id=None, dry_run=False):
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script_text)
        tmp = f.name
    try:
        cmd = ['sbatch']
        if dependency_id:
            cmd += [f'--dependency=afterany:{dependency_id}']
        cmd.append(tmp)
        if dry_run:
            print(f'  [dry-run] sbatch {" ".join(cmd[1:])}')
            return 'DRY000000'
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  ERROR: {result.stderr.strip()}')
            sys.exit(1)
        return result.stdout.strip().split()[-1]
    finally:
        os.unlink(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be submitted without actually submitting')
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print('Checking job status...\n')
    queued = queued_job_names()

    done    = []
    running = []
    missing = []

    for arch in ARCHS:
        for ablation, params in ABLATIONS.items():
            for param in params:
                name = job_name(arch, ablation, param)
                if is_complete(arch, ablation, param):
                    done.append(name)
                elif name in queued:
                    running.append(name)
                else:
                    missing.append((arch, ablation, param, name))

    print(f'  Done    : {len(done):2d} / 48')
    print(f'  Queued  : {len(running):2d} / 48')
    print(f'  Missing : {len(missing):2d} / 48')

    if done:
        print(f'\nCompleted:')
        for name in done:
            print(f'  ✓ {name}')

    if running:
        print(f'\nAlready in queue (skipping):')
        for name in running:
            print(f'  ~ {name}')

    if not missing:
        print('\nAll jobs complete or running. Nothing to submit.')
        return

    print(f'\nSubmitting {len(missing)} missing jobs '
          f'(rolling {MAX_CONCURRENT}-slot window)...')
    if args.dry_run:
        print('(DRY RUN)\n')

    submitted_ids = []
    for i, (arch, ablation, param, name) in enumerate(missing):
        dep_id = submitted_ids[i - MAX_CONCURRENT] if i >= MAX_CONCURRENT else None
        partial = has_partial(arch, ablation, param)
        script  = make_slurm_script(arch, ablation, param)
        job_id  = sbatch(script, dependency_id=dep_id, dry_run=args.dry_run)
        submitted_ids.append(job_id)

        dep_str     = f'after {dep_id}' if dep_id else 'immediate'
        resume_note = ' (resuming)' if partial else ''
        print(f'  [{i+1:2d}/{len(missing)}] {name}{resume_note} -> {job_id}  ({dep_str})')

    print(f'\nDone. {len(missing)} jobs submitted.')
    print('Monitor: squeue -u omrosa')


if __name__ == '__main__':
    main()

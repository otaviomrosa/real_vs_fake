"""Submit all ablation jobs with a rolling 6-slot window.

Job i depends on job (i-6), so at most 6 jobs run simultaneously at any time.
SLURM handles scheduling; this script just submits once.

Run from within the experiment directory:
    python ablations/submit_all.py [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

MAX_CONCURRENT = 6
# Resolve job_list.txt relative to this script's own location (ablations/)
JOB_LIST = Path(__file__).parent / 'job_list.txt'


def sbatch(script_path, dependency_id=None, dry_run=False):
    cmd = ['sbatch']
    if dependency_id:
        cmd += [f'--dependency=afterany:{dependency_id}']
    cmd.append(script_path)

    if dry_run:
        dep_str = f' (after {dependency_id})' if dependency_id else ''
        print(f'[dry-run] {" ".join(cmd)}{dep_str}')
        return f'DRY{script_path[-10:]}'  # fake job id

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'ERROR submitting {script_path}:\n{result.stderr}')
        sys.exit(1)
    # Output: "Submitted batch job 123456"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true',
                   help='Print commands without submitting')
    args = p.parse_args()

    if not JOB_LIST.exists():
        print(f'ERROR: {JOB_LIST} not found.')
        print('Run: python generate_ablation_jobs.py first.')
        sys.exit(1)

    jobs = []
    with open(JOB_LIST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, script = line.split('\t')
            jobs.append((name, script))

    print(f'Submitting {len(jobs)} jobs with max {MAX_CONCURRENT} concurrent...')
    if args.dry_run:
        print('(DRY RUN — no actual submission)')
    print()

    submitted_ids = []

    for i, (name, script) in enumerate(jobs):
        dep_id = submitted_ids[i - MAX_CONCURRENT] if i >= MAX_CONCURRENT else None
        job_id = sbatch(script, dependency_id=dep_id, dry_run=args.dry_run)
        submitted_ids.append(job_id)
        slot = i % MAX_CONCURRENT
        dep_str = f'after {dep_id}' if dep_id else 'immediate'
        print(f'[{i+1:3d}/{len(jobs)}] slot={slot} dep={dep_str} -> {job_id}  {name}')

    print()
    print(f'Done. {len(jobs)} jobs submitted.')
    print('Monitor with: squeue -u omrosa')
    if not args.dry_run:
        print('Resume a preempted job by resubmitting its script with:')
        print('  sbatch --dependency=... <script.sh>  # add --resume arg inside the script')


if __name__ == '__main__':
    main()

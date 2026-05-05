"""Rolling job submitter — keeps at most MAX_CONCURRENT jobs in the SLURM queue.

Run inside tmux so it survives SSH disconnection:
    tmux new -s submit
    python ablations/submit_rolling.py
    # Ctrl+B, D  to detach

State is saved to submitted_jobs.txt next to this script so it can resume
if the session is interrupted. Re-running will skip already-submitted jobs.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

MAX_CONCURRENT = 6
POLL_INTERVAL  = 60   # seconds between squeue checks
USER           = 'omrosa'

SCRIPT_DIR = Path(__file__).parent
JOB_LIST   = SCRIPT_DIR / 'job_list.txt'
STATE_FILE = SCRIPT_DIR / 'submitted_jobs.txt'


def count_active_jobs():
    """Return number of jobs in queue for USER (running + pending)."""
    result = subprocess.run(
        ['squeue', '-u', USER, '-h'],  # -h suppresses header
        capture_output=True, text=True
    )
    return len([l for l in result.stdout.splitlines() if l.strip()])


def sbatch(script_path):
    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr.strip()}')
        return None
    job_id = result.stdout.strip().split()[-1]
    return job_id


def load_submitted():
    if not STATE_FILE.exists():
        return set()
    return set(STATE_FILE.read_text().splitlines())


def record_submitted(name):
    with open(STATE_FILE, 'a') as f:
        f.write(name + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    if not JOB_LIST.exists():
        print(f'ERROR: {JOB_LIST} not found. Run generate_jobs first.')
        sys.exit(1)

    jobs = []
    for line in JOB_LIST.read_text().splitlines():
        if line.strip():
            name, script = line.split('\t')
            jobs.append((name, script))

    submitted = load_submitted()
    pending   = [(n, s) for n, s in jobs if n not in submitted]

    print(f'Total jobs: {len(jobs)} | Already submitted: {len(submitted)} | Remaining: {len(pending)}')
    if args.dry_run:
        print('(DRY RUN)')
    print()

    while pending:
        active = count_active_jobs()
        slots  = MAX_CONCURRENT - active

        if slots <= 0:
            print(f'  Queue full ({active}/{MAX_CONCURRENT}). Waiting {POLL_INTERVAL}s...', flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        to_submit = pending[:slots]
        for name, script in to_submit:
            if args.dry_run:
                print(f'  [dry-run] would sbatch {name}')
                record_submitted(name)
            else:
                job_id = sbatch(script)
                if job_id:
                    print(f'  Submitted {name} -> {job_id}  ({len(pending)-1} remaining)', flush=True)
                    record_submitted(name)
                else:
                    print(f'  FAILED to submit {name}, will retry next cycle.', flush=True)
                    break  # don't advance past a failure

        pending = [(n, s) for n, s in jobs if n not in load_submitted()]

        if pending:
            time.sleep(POLL_INTERVAL)

    print('\nAll jobs submitted.')
    print('Monitor with: squeue -u omrosa')


if __name__ == '__main__':
    main()

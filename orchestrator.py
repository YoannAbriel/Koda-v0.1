"""
Orchestrator for the 3-phase pipeline.

Detects if phase 1 (train_continue.py) is already running.
If yes: waits for it to finish (watching marker or process).
If no and not done: launches it.

Then runs phase 2 and phase 3 sequentially.
"""
import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from datetime import datetime
from config import KODA_ROOT

ROOT = Path(str(KODA_ROOT))
MARKERS = ROOT / 'markers'
MARKERS.mkdir(exist_ok=True)


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def is_train_continue_running():
    """Check if train_continue.py is currently running."""
    try:
        out = subprocess.check_output(['pgrep', '-f', 'python3.*train_continue.py'], text=True)
        pids = [int(p) for p in out.strip().split('\n') if p.strip()]
        return len(pids) > 0
    except subprocess.CalledProcessError:
        return False


def get_current_pretrain_step():
    """Read the latest step from train_continue.log."""
    log_path = ROOT / 'train_continue.log'
    if not log_path.exists():
        return 0
    try:
        with open(log_path) as f:
            content = f.read()
        import re
        steps = re.findall(r'Step (\d+)/', content)
        if steps:
            return int(steps[-1])
    except Exception:
        pass
    return 0


def wait_for_phase1():
    """Wait for phase 1 to complete (either marker exists or process dies after 300K)."""
    marker = MARKERS / 'phase1.done'

    while not marker.exists():
        running = is_train_continue_running()
        step = get_current_pretrain_step()

        if not running:
            # Process died — check if it reached the target
            if step >= 300000:
                log(f'Phase 1 process ended at step {step}, creating marker')
                marker.write_text('completed')
                return True
            else:
                log(f'Phase 1 process died at step {step} (< 300000), restarting')
                # Restart it
                subprocess.Popen(
                    ['nohup', 'python3', '-u', 'train_continue.py'],
                    stdout=open(str(ROOT / 'train_continue.log'), 'a'),
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT),
                    preexec_fn=os.setsid,
                )
                time.sleep(30)
        else:
            log(f'Phase 1 running, current step: {step}/300000')

        time.sleep(300)  # check every 5 min

    log('Phase 1 done!')
    return True


def run_phase(name, script, marker_name):
    marker = MARKERS / marker_name
    if marker.exists():
        log(f'SKIP: {name} (marker exists)')
        return True

    log(f'START: {name}')
    log_path = ROOT / f'{script.replace(".py", "")}.log'

    with open(log_path, 'w') as logf:
        result = subprocess.run(
            ['python3', '-u', script],
            cwd=str(ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )

    if result.returncode == 0 and marker.exists():
        log(f'DONE: {name}')
        return True
    else:
        log(f'FAILED: {name} (code {result.returncode})')
        log(f'  See {log_path}')
        return False


def main():
    log('=== Orchestrator starting ===')

    # Phase 1: handle the special case of pre-existing process
    log('Phase 1: Continue pretraining')
    marker1 = MARKERS / 'phase1.done'
    if marker1.exists():
        log('  Already done, skipping')
    else:
        if is_train_continue_running():
            log('  Already running, waiting for completion')
            wait_for_phase1()
        else:
            step = get_current_pretrain_step()
            if step >= 300000:
                log(f'  Already at step {step}, marking done')
                marker1.write_text('completed')
            else:
                log('  Not running, starting fresh')
                subprocess.Popen(
                    ['nohup', 'python3', '-u', 'train_continue.py'],
                    stdout=open(str(ROOT / 'train_continue.log'), 'a'),
                    stderr=subprocess.STDOUT,
                    cwd=str(ROOT),
                    preexec_fn=os.setsid,
                )
                time.sleep(30)
                wait_for_phase1()

    # Phase 2
    if not run_phase('Phase 2: SFT v2 long', 'phase2.py', 'phase2.done'):
        log('STOPPED: Phase 2 failed')
        sys.exit(1)

    # Phase 3
    if not run_phase('Phase 3: Context extension', 'phase3.py', 'phase3.done'):
        log('STOPPED: Phase 3 failed')
        sys.exit(1)

    log('=== ALL PHASES COMPLETE ===')


if __name__ == '__main__':
    main()

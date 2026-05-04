"""
Phase 1: continue pretraining (existing train_continue.py).
Wraps it to write a done marker on success.
"""
import subprocess, sys
from pathlib import Path

DONE_MARKER = Path("/opt/yoann-test/markers/phase1.done")
DONE_MARKER.parent.mkdir(exist_ok=True)

if DONE_MARKER.exists():
    print("Phase 1 already done", flush=True)
    sys.exit(0)

print("Phase 1: Continue pretraining", flush=True)
result = subprocess.run(["python3", "-u", "train_continue.py"], cwd="/opt/yoann-test")

if result.returncode == 0:
    DONE_MARKER.write_text("completed")
    print("Phase 1 done", flush=True)
    sys.exit(0)
else:
    print(f"Phase 1 FAILED with code {result.returncode}", flush=True)
    sys.exit(1)

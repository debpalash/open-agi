#!/usr/bin/env python3
"""
Sync protocol for dual-machine autoresearch.

Before each experiment:  python sync.py pull
After each winning exp:  python sync.py push "description of what changed"

This handles git pull/push with conflict resolution and updates best_config.json.
"""

import json
import os
import subprocess
import sys

CONFIG_FILE = "best_config.json"

def run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"WARN: {cmd}\n{r.stderr.strip()}")
    return r

def pull():
    """Pull latest from remote, rebase local changes."""
    print("Syncing from remote...")
    r = run("git pull --rebase origin main")
    if r.returncode != 0:
        # If rebase fails, abort and try merge
        run("git rebase --abort", check=False)
        r = run("git pull origin main --no-rebase")
    if r.returncode == 0:
        print("✓ Synced with remote")
        # Show latest best config
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            print(f"  Current best: val_bpb={cfg['best_val_bpb']:.6f} ({cfg['best_machine']})")
            print(f"  Total experiments: {len(cfg.get('history', []))}")
    else:
        print("✗ Sync failed — check git status")

def push(description="", val_bpb=None, machine=None, params_m=None, steps=None):
    """Commit and push current state."""
    # Auto-detect machine
    if machine is None:
        try:
            import mlx.core
            machine = "mac"
        except ImportError:
            machine = "cuda"

    # Parse run.log for results if not provided
    if val_bpb is None and os.path.exists("run.log"):
        with open("run.log") as f:
            for line in f:
                if line.startswith("val_bpb:"):
                    val_bpb = float(line.split(":")[1].strip())
                elif line.startswith("num_params_M:"):
                    params_m = float(line.split(":")[1].strip())
                elif line.startswith("num_steps:"):
                    steps = int(line.split(":")[1].strip())

    if val_bpb is None:
        print("No val_bpb found. Run training first or pass --val_bpb")
        return

    # Update best_config.json
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    else:
        cfg = {"best_val_bpb": 999, "config": {}, "history": []}

    # Get current commit
    commit = run("git rev-parse --short HEAD", check=False).stdout.strip()

    # Add to history
    entry = {
        "machine": machine,
        "val_bpb": val_bpb,
        "params_M": params_m,
        "steps": steps,
        "description": description,
        "commit": commit,
    }
    cfg.setdefault("history", []).append(entry)

    # Update best if improved
    if val_bpb < cfg.get("best_val_bpb", 999):
        cfg["best_val_bpb"] = val_bpb
        cfg["best_machine"] = machine
        cfg["best_commit"] = commit
        print(f"★ New best! val_bpb={val_bpb:.6f} (was {cfg.get('best_val_bpb', 'N/A')})")
        # Read current hyperparams from the appropriate train file
        cfg["config"] = _read_current_config(machine)

    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    # Commit and push
    run("git add -A")
    msg = f"[{machine}] val_bpb={val_bpb:.6f} — {description}" if description else f"[{machine}] val_bpb={val_bpb:.6f}"
    run(f'git commit -m "{msg}"')

    r = run("git push origin main")
    if r.returncode != 0:
        print("Push failed — pulling and retrying...")
        run("git pull --rebase origin main")
        run("git push origin main")

    print(f"✓ Pushed: {msg}")

def _read_current_config(machine):
    """Extract hyperparameter values from the train file."""
    fname = "train.py" if machine == "mac" else "train_cuda.py"
    config = {}
    keys = ["MODEL_DIM", "N_HEADS", "PRELUDE_DEPTH", "CODA_DEPTH", "N_LOOPS",
            "FFN_MULT", "USE_MOE", "N_EXPERTS", "TOP_K", "SHARED_EXPERTS",
            "USE_LTI", "LORA_RANK", "LR", "WEIGHT_DECAY", "WARMUP_RATIO",
            "WARMDOWN_RATIO", "BATCH_SIZE", "TOTAL_BATCH"]
    try:
        with open(fname) as f:
            for line in f:
                line = line.strip()
                for key in keys:
                    if line.startswith(f"{key}") and "=" in line:
                        val_str = line.split("=")[1].split("#")[0].strip()
                        try:
                            config[key] = eval(val_str)
                        except Exception:
                            config[key] = val_str
    except FileNotFoundError:
        pass
    return config

def status():
    """Show current state."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        print(f"Best val_bpb: {cfg['best_val_bpb']:.6f} ({cfg['best_machine']})")
        print(f"Experiments: {len(cfg.get('history', []))}")
        print("\nRecent history:")
        for e in cfg.get("history", [])[-10:]:
            flag = "★" if e.get("val_bpb") == cfg["best_val_bpb"] else " "
            print(f"  {flag} [{e.get('machine','?'):4s}] {e.get('val_bpb',0):.6f} — {e.get('description','')}")
    else:
        print("No experiments yet. Run training first.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sync.py [pull|push|status] [description]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "pull":
        pull()
    elif cmd == "push":
        desc = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        push(description=desc)
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command: {cmd}")

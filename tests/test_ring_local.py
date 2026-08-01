#!/usr/bin/env python3
"""Standalone ring init test — spawns two subprocess processes that each call
mx.distributed.init and attempt a barrier.

Run:
    cd omlx && python3 tests/test_ring_local.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    endpoints = ["127.0.0.1:5500", "127.0.0.1:5501"]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write shared hostfile and endpoints
        from omlx.mlx_distributed import create_hostfile_json

        hostfile_path = os.path.join(tmpdir, "hostfile.json")
        with open(hostfile_path, "w") as f:
            json.dump(create_hostfile_json(endpoints), f)

        endpoints_path = os.path.join(tmpdir, "endpoints.json")
        with open(endpoints_path, "w") as f:
            json.dump(endpoints, f)

        # Write the rank script code to a temp file
        script_path = os.path.join(tmpdir, "rank.py")
        with open(script_path, "w") as f:
            f.write(f"""\
import json, os, sys

endpoints_path = sys.argv[1]
rank = int(sys.argv[2])
hostfile_path = sys.argv[3]
ring_port = int(sys.argv[4]) if len(sys.argv) > 4 else 5000

with open(endpoints_path) as f:
    endpoints = json.load(f)
os.environ["MLX_HOSTFILE"] = hostfile_path
os.environ["MLX_RANK"] = str(rank)

# Ensure the package root is on sys.path for imports
sys.path.insert(0, {_ROOT_DIR!r})

import mlx.core as mx
from omlx.mlx_distributed import init_mlx_distributed, barrier

group = init_mlx_distributed(
    endpoints=endpoints,
    local_rank=rank,
    ring_port=ring_port,
)

rank_ok = group is not None
size = group.size() if group else 0
my_rank = group.rank() if group else -1

barrier_ok = False
if group:
    try:
        barrier(group)
        barrier_ok = True
    except Exception as exc:
        pass

print(f"OK_{{rank}}={{rank_ok}} SIZE_{{rank}}={{size}} RANK_VAL_{{rank}}={{my_rank}} BARRIER_{{rank}}={{barrier_ok}}")
""")

        print(f"Hostfile: {open(hostfile_path).read()}")
        print()

        # Start rank 1 (worker) first so it's ready to connect
        # Each rank needs its own listen port matching its endpoint
        p1 = subprocess.Popen(
            [sys.executable, script_path, endpoints_path, "1", hostfile_path, str(5501)],
            cwd=_ROOT_DIR,
            env=os.environ,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )

        # Give it a moment to start listening
        time.sleep(1.5)

        # Start rank 0 (coordinator)
        p0 = subprocess.Popen(
            [sys.executable, script_path, endpoints_path, "0", hostfile_path, str(5500)],
            cwd=_ROOT_DIR,
            env=os.environ,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for both to finish (with timeout)
        stdout_parts = []
        stderr_parts = []
        for p in [p0, p1]:
            try:
                out, err = p.communicate(timeout=30)
                stdout_parts.append(out.strip())
                stderr_parts.append(err.strip())
            except subprocess.TimeoutExpired:
                p.kill()
                out, _ = p.communicate()
                stdout_parts.append(out.strip())
                stderr_parts.append("(killed by timeout)")

        print("=== stdout ===")
        for line in stdout_parts:
            if line:
                print(line)

        print("\n=== stderr ===")
        for line in stderr_parts:
            if line:
                print(line)

        # Parse results
        results = {}
        for part in stdout_parts:
            for token in part.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    results[k] = v

        print("\n=== Results ===")
        ok0 = results.get("OK_0") == "True"
        ok1 = results.get("OK_1") == "True"
        b0 = results.get("BARRIER_0") == "True"
        b1 = results.get("BARRIER_1") == "True"

        if ok0 and ok1 and b0 and b1:
            print("PASS: Ring initialized and barrier completed on both ranks")
            return 0
        else:
            print(f"FAIL: rank0_ok={ok0} rank1_ok={ok1} barrier0={b0} barrier1={b1}")
            return 1


if __name__ == "__main__":
    sys.exit(main())

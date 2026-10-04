"""Work through the benchmark queue, one subprocess per job, appending results to results.jsonl.

Restart-safe: jobs already in results.jsonl are skipped, so relaunching resumes where it stopped.
Usage: python experiments/runner.py [--workers 2] [--only-dataset adult]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")
FAILURES = os.path.join(HERE, "logs", "failures.jsonl")
LOG = os.path.join(HERE, "logs", "run.log")
JOB_TIMEOUT_S = 3 * 3600

DATASETS = ["adult", "diabetes130", "brfss"]
EPSILONS = [1.0, 2.0, 4.0]


def queue():
    """Ordered so that every prefix of the queue is a usable (if smaller) experiment."""
    jobs = []
    for ds in DATASETS:
        jobs.append((ds, "identity", 0.0, 0))
        for seed in range(5):
            for eps in EPSILONS:
                jobs.append((ds, "mst", eps, seed))
                jobs.append((ds, "dpvae", eps, seed))
        for seed in range(3):
            for eps in EPSILONS:
                jobs.append((ds, "patectgan", eps, seed))
                jobs.append((ds, "dpctgan", eps, seed))
        jobs.append((ds, "ctgan", 0.0, 0))
        jobs.append((ds, "tvae", 0.0, 0))
    for ds in DATASETS:
        for seed in range(5, 10):
            for eps in EPSILONS:
                jobs.append((ds, "mst", eps, seed))
                jobs.append((ds, "dpvae", eps, seed))
        for seed in range(3, 5):
            for eps in EPSILONS:
                jobs.append((ds, "patectgan", eps, seed))
                jobs.append((ds, "dpctgan", eps, seed))
        for seed in range(1, 3):
            jobs.append((ds, "ctgan", 0.0, seed))
            jobs.append((ds, "tvae", 0.0, seed))
    for ds in DATASETS:
        for seed in range(3):
            for eps in EPSILONS:
                jobs.append((ds, "dpvae_argmax", eps, seed))
    for ds in DATASETS:
        for seed in range(3, 10):
            for eps in EPSILONS:
                jobs.append((ds, "dpvae_argmax", eps, seed))
    # AIM's runtime grows steeply with epsilon (~27 min at eps=1, >2 h at eps=2 on Adult),
    # so one seed per cell first and extra seeds only as time allows.
    for seed in range(3):
        for ds in DATASETS:
            jobs.append((ds, "aim", 1.0, seed))
    for ds in DATASETS:
        for eps in EPSILONS[1:]:
            jobs.append((ds, "aim", eps, 0))
    return jobs


def key(ds, method, eps, seed):
    return f"{ds}|{method}|{float(eps)}|{int(seed)}"


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add(key(r["dataset"], r["method"], r["eps"], r["seed"]))
                except Exception:
                    pass
    return keys


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def run(job, threads):
    ds, method, eps, seed = job
    if os.path.exists(os.path.join(HERE, "logs", "DRAIN")):
        return
    t0 = time.time()
    env = dict(os.environ, TORCH_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
    cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--dataset", ds, "--method", method,
           "--eps", str(eps), "--seed", str(seed)]
    log(f"START {key(*job)}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S, env=env)
        out = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if out:
            with open(RESULTS, "a") as f:
                f.write(out[-1][len("RESULT "):] + "\n")
            log(f"DONE  {key(*job)} {time.time() - t0:.0f}s")
            return
        err = f"returncode={p.returncode} " + (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        err = "timeout"
    with open(FAILURES, "a") as f:
        f.write(json.dumps({"key": key(*job), "err": err, "elapsed": time.time() - t0}) + "\n")
    log(f"FAIL  {key(*job)} {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only-dataset", default="")
    ap.add_argument("--only-method", default="")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    done = done_keys(RESULTS)
    failed = set()
    if os.path.exists(FAILURES):
        with open(FAILURES) as f:
            failed = {json.loads(l)["key"] for l in f if l.strip()}
    todo = [j for j in queue() if key(*j) not in done and key(*j) not in failed
            and (not a.only_dataset or j[0] == a.only_dataset)
            and (not a.only_method or j[1] == a.only_method)]
    log(f"runner start pid={os.getpid()} todo={len(todo)} done={len(done)} failed={len(failed)}")
    threads = max(1, 4 // a.workers)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda j: run(j, threads), todo))
    log("runner finished")


if __name__ == "__main__":
    main()

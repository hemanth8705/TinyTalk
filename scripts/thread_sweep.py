"""Thread sweep: run llama-bench with different CPU thread counts and compare speed.

Usage (from the repo root):
    python scripts\thread_sweep.py
    python scripts\thread_sweep.py --threads 4 6 8 --reps 5

Prefill = tokens/sec while reading the prompt. Decode = tokens/sec while writing the answer.
Results are printed as a table and appended to results/thread_sweep.csv.
"""
import argparse
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "tools" / "llama.cpp" / "llama-bench.exe"
DEFAULT_MODEL = ROOT / "models" / "Qwen3-1.7B-Q4_K_M.gguf"
CSV_PATH = ROOT / "results" / "thread_sweep.csv"


def run_bench(model, threads, n_prompt, n_gen, reps):
    """Run llama-bench once (CPU only) and return {'prefill': (avg, std), 'decode': (avg, std)}."""
    cmd = [
        str(BENCH), "-m", str(model),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-r", str(reps), "-ngl", "0", "-t", str(threads),
        "-o", "json",
    ]
    print(f"Executing: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-bench failed (exit {proc.returncode}):\n{proc.stderr[-800:]}")

    # llama.cpp logs go to stderr; the JSON result is on stdout.
    text = proc.stdout
    rows = json.loads(text[text.index("["):])

    result = {}
    for row in rows:
        if row.get("n_gen", 0) > 0:
            result["decode"] = (row["avg_ts"], row["stddev_ts"])
        elif row.get("n_prompt", 0) > 0:
            result["prefill"] = (row["avg_ts"], row["stddev_ts"])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--prompt-tokens", type=int, default=128, help="tokens used for the prefill test")
    ap.add_argument("--gen-tokens", type=int, default=32, help="tokens generated for the decode test")
    ap.add_argument("--reps", type=int, default=3, help="repeats per test (averaged)")
    ap.add_argument("--cooldown", type=int, default=5, help="seconds to rest between runs")
    args = ap.parse_args()

    if not BENCH.exists():
        raise SystemExit(f"llama-bench not found at {BENCH}")
    if not args.model.exists():
        raise SystemExit(f"model not found at {args.model}")

    print(f"Model: {args.model.name}")
    print("Tip: plug in the laptop, set power mode to Best performance, close other apps.\n")

    results = []
    for i, t in enumerate(args.threads):
        print(f"Running with {t} threads ...", flush=True)
        r = run_bench(args.model, t, args.prompt_tokens, args.gen_tokens, args.reps)
        results.append((t, r["prefill"], r["decode"]))
        print(f"  prefill {r['prefill'][0]:7.1f} t/s   decode {r['decode'][0]:6.2f} t/s", flush=True)
        if i < len(args.threads) - 1:
            time.sleep(args.cooldown)

    print("\n threads | prefill t/s (+/-) | decode t/s (+/-)")
    print("---------+-------------------+------------------")
    for t, (pp, pp_sd), (tg, tg_sd) in results:
        print(f" {t:7d} | {pp:8.1f} ({pp_sd:5.1f}) | {tg:7.2f} ({tg_sd:5.2f})")

    best_decode = max(results, key=lambda x: x[2][0])
    best_prefill = max(results, key=lambda x: x[1][0])
    print(f"\nBest decode:  {best_decode[0]} threads ({best_decode[2][0]:.2f} t/s)")
    print(f"Best prefill: {best_prefill[0]} threads ({best_prefill[1][0]:.1f} t/s)")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "model", "threads", "prefill_tps", "prefill_std",
                        "decode_tps", "decode_std", "prompt_tokens", "gen_tokens", "reps"])
        now = datetime.now().isoformat(timespec="seconds")
        for t, (pp, pp_sd), (tg, tg_sd) in results:
            w.writerow([now, args.model.name, t, round(pp, 2), round(pp_sd, 2),
                        round(tg, 2), round(tg_sd, 2), args.prompt_tokens, args.gen_tokens, args.reps])
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()

"""Benchmark harness: starts llama-server, sends prompts, measures speed and memory.

The same file runs on the laptop and (later) on the phone. The client talks to the server
over localhost, so network delay never enters the numbers.

Usage (from the repo root):
    python scripts\\harness.py
    python scripts\\harness.py --model models\\Qwen3-1.7B-Q4_K_M.gguf --ctx 2048 --threads 8 --reps 3

Measured per generation: load time, TTFT, prefill/decode tokens per sec, token-gap
percentiles (stutter), and process RAM (current + peak). On the phone (Termux with the
Termux:API app) it also records battery %, battery temperature, charging state and CPU
temperature; on the laptop those columns stay empty. One CSV row is appended per
generation to results/benchmark.csv, so a crash never loses finished runs. Raw outputs go
to results/raw/ (git-ignored).

Every generation produces exactly --max-tokens tokens by default (end-of-text is ignored),
so all prompts do the same amount of decode work. Use --no-fixed-output for natural lengths.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MB = 1024 * 1024
CSV_PATH = ROOT / "results" / "benchmark.csv"  # new schema; the old benchmarks.csv is left untouched
RAW_DIR = ROOT / "results" / "raw"
CSV_FIELDS = [
    "timestamp", "run_id", "device", "model", "quant", "n_ctx", "threads", "rep",
    "prompt_id", "fixed_len", "n_prompt", "n_out", "load_s", "ttft_s", "prefill_tps", "decode_tps",
    "decode_tps_client", "itl_p50_ms", "itl_p95_ms", "itl_max_ms", "rss_mb", "peak_rss_mb",
    "batt_start_pct", "batt_end_pct", "batt_temp_c", "plugged", "cpu_temp_c",
    "llama_cpp_build", "gguf_sha256",
]


# ---------- process memory (no third-party packages) ----------

def process_memory_mb(pid):
    """Return (current_rss_mb, peak_rss_mb) for a process, or (None, None)."""
    if sys.platform == "win32":
        return _win_memory(pid)
    try:  # Linux / Android (Termux)
        cur = peak = None
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
        return cur, peak
    except OSError:
        return None, None


def _win_memory(pid):
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]

    handle = k32.OpenProcess(0x0410, False, pid)  # QUERY_INFORMATION | VM_READ
    if not handle:
        return None, None
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return None, None
        return pmc.WorkingSetSize / MB, pmc.PeakWorkingSetSize / MB
    finally:
        k32.CloseHandle(handle)


# ---------- phone sensors (Termux only; everything returns None elsewhere) ----------

def battery_status():
    """Return {'pct', 'temp_c', 'plugged'} from `termux-battery-status`, or None if unavailable."""
    exe = shutil.which("termux-battery-status")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=10)
        d = json.loads(out.stdout)
        return {"pct": d.get("percentage"), "temp_c": d.get("temperature"),
                "plugged": d.get("plugged", "")}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def cpu_temp_c():
    """Hottest readable thermal zone in deg C. Many phones block this without root; then None."""
    best = None
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            v = float(zone.read_text().strip())
        except (OSError, ValueError):
            continue
        v = v / 1000 if v > 1000 else v
        if 0 < v < 150 and (best is None or v > best):
            best = v
    return best


# ---------- helpers ----------

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * MB), b""):
            h.update(block)
    return h.hexdigest()


def guess_quant(path):
    m = re.search(r"(IQ\d\w*|Q\d[\w]*?|BF16|F16|F32)(?=\.gguf$)", path.name, re.I)
    return m.group(1).upper() if m else "unknown"


def server_build(server_exe):
    try:
        out = subprocess.run([server_exe, "--version"], capture_output=True, text=True, timeout=30)
        m = re.search(r"build (\d+)", out.stdout + out.stderr)
        return m.group(1) if m else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(pct / 100 * (len(s) - 1))))]


def default_server():
    if sys.platform == "win32":
        return ROOT / "tools" / "llama.cpp" / "llama-server.exe"
    return Path(shutil.which("llama-server") or "llama-server")


# ---------- server lifecycle ----------

def start_server(args, run_id):
    """Start llama-server, wait until /health is OK. Returns (process, load_seconds)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log = open(RAW_DIR / f"{run_id}_server.log", "w")
    cmd = [
        str(args.server), "-m", str(args.model), "-c", str(args.ctx), "-t", str(args.threads),
        "-ngl", "0", "-np", "1", "--host", "127.0.0.1", "--port", str(args.port),
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)
    url = f"http://127.0.0.1:{args.port}/health"
    while True:
        if proc.poll() is not None:
            raise SystemExit(f"llama-server exited early (code {proc.returncode}); see {log.name}")
        if time.perf_counter() - t0 > args.load_timeout:
            proc.kill()
            raise SystemExit(f"llama-server not ready after {args.load_timeout}s; see {log.name}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return proc, time.perf_counter() - t0
        except OSError:
            pass
        time.sleep(0.2)


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------- one generation ----------

def generate(args, text):
    """Stream one chat completion; return timing facts and the output text."""
    payload = {
        "messages": [{"role": "user", "content": text}],
        "stream": True,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,                         # never reuse a cached prompt: honest prefill
        "chat_template_kwargs": {"enable_thinking": False},  # Qwen3: no thinking tokens
        "stream_options": {"include_usage": True},
    }
    if args.fixed_output:
        payload["ignore_eos"] = True  # always generate exactly max_tokens tokens
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )

    t_send = time.perf_counter()
    stamps, pieces, timings, usage = [], [], None, None
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            now = time.perf_counter()
            timings = chunk.get("timings", timings)
            usage = chunk.get("usage", usage)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    stamps.append(now)
                    pieces.append(piece)

    if not stamps:
        raise RuntimeError("no tokens received")
    gaps_ms = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])]
    n_out = (timings or {}).get("predicted_n") or (usage or {}).get("completion_tokens") or len(stamps)
    client_span = stamps[-1] - stamps[0]
    return {
        "text": "".join(pieces),
        "ttft_s": stamps[0] - t_send,
        "n_prompt": (timings or {}).get("prompt_n") or (usage or {}).get("prompt_tokens"),
        "n_out": n_out,
        "prefill_tps": (timings or {}).get("prompt_per_second"),
        "decode_tps": (timings or {}).get("predicted_per_second"),
        "decode_tps_client": (len(stamps) - 1) / client_span if client_span > 0 else None,
        "itl_p50_ms": percentile(gaps_ms, 50),
        "itl_p95_ms": percentile(gaps_ms, 95),
        "itl_max_ms": max(gaps_ms) if gaps_ms else None,
        "stamps": [s - t_send for s in stamps],
    }


def fmt(v, nd=2):
    return "" if v is None else round(v, nd)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "Qwen3-1.7B-Q4_K_M.gguf")
    ap.add_argument("--server", type=Path, default=default_server())
    ap.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "speed_prompts.json")
    ap.add_argument("--device", default="laptop", help="label stored in the CSV")
    ap.add_argument("--quant", default=None, help="label; guessed from the filename if omitted")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--fixed-output", action=argparse.BooleanOptionalAction, default=True,
                    help="force exactly --max-tokens output tokens per prompt (default on)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between generations")
    ap.add_argument("--load-timeout", type=int, default=180)
    ap.add_argument("--request-timeout", type=int, default=300)
    args = ap.parse_args()

    for p, what in ((args.model, "model"), (args.prompts, "prompts file")):
        if not p.exists():
            raise SystemExit(f"{what} not found: {p}")
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))

    quant = args.quant or guess_quant(args.model)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Run {run_id}: {args.model.name} | {quant} | ctx {args.ctx} | {args.threads} threads | {args.device}")
    print("Hashing model file (for reproducibility) ...", flush=True)
    sha, build = sha256_of(args.model), server_build(args.server)

    proc, load_s = start_server(args, run_id)
    print(f"Server ready in {load_s:.1f}s (build {build}). Warming up ...", flush=True)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_csv = not CSV_PATH.exists()
    raw_f = open(RAW_DIR / f"{run_id}_outputs.jsonl", "w", encoding="utf-8")
    rows = []
    try:
        generate(args, "Say hello in one short sentence.")  # warm-up, discarded
        with CSV_PATH.open("a", newline="") as csv_f:
            w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
            if new_csv:
                w.writeheader()
            for rep in range(1, args.reps + 1):
                for p in prompts:
                    batt0 = battery_status()  # sensor calls sit outside the timed window
                    g = generate(args, p["text"])
                    batt1 = battery_status()
                    rss, peak = process_memory_mb(proc.pid)
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"), "run_id": run_id,
                        "device": args.device, "model": args.model.name, "quant": quant,
                        "n_ctx": args.ctx, "threads": args.threads, "rep": rep, "prompt_id": p["id"],
                        "fixed_len": args.fixed_output,
                        "n_prompt": g["n_prompt"], "n_out": g["n_out"], "load_s": fmt(load_s),
                        "ttft_s": fmt(g["ttft_s"], 3), "prefill_tps": fmt(g["prefill_tps"], 1),
                        "decode_tps": fmt(g["decode_tps"]), "decode_tps_client": fmt(g["decode_tps_client"]),
                        "itl_p50_ms": fmt(g["itl_p50_ms"], 1), "itl_p95_ms": fmt(g["itl_p95_ms"], 1),
                        "itl_max_ms": fmt(g["itl_max_ms"], 1), "rss_mb": fmt(rss, 0),
                        "peak_rss_mb": fmt(peak, 0),
                        "batt_start_pct": batt0["pct"] if batt0 else "",
                        "batt_end_pct": batt1["pct"] if batt1 else "",
                        "batt_temp_c": batt1["temp_c"] if batt1 else "",
                        "plugged": batt1["plugged"] if batt1 else "",
                        "cpu_temp_c": fmt(cpu_temp_c(), 1),
                        "llama_cpp_build": build, "gguf_sha256": sha,
                    }
                    w.writerow(row)
                    csv_f.flush()
                    raw_f.write(json.dumps({"run_id": run_id, "rep": rep, "prompt_id": p["id"],
                                            "output": g["text"], "token_times_s": g["stamps"]}) + "\n")
                    raw_f.flush()
                    rows.append(row)
                    print(f"  rep {rep} {p['id']:<14} ttft {g['ttft_s']:.2f}s  "
                          f"prefill {fmt(g['prefill_tps'], 0)} t/s  decode {fmt(g['decode_tps'], 1)} t/s  "
                          f"out {g['n_out']} tok", flush=True)
                    time.sleep(args.pause)
    finally:
        raw_f.close()
        stop_server(proc)

    def med(key):
        vals = [float(r[key]) for r in rows if r[key] != ""]
        return statistics.median(vals) if vals else None

    print("\nSummary (median over all generations):")
    print(f"  load time        {load_s:.1f} s")
    print(f"  TTFT             {fmt(med('ttft_s'), 3)} s")
    print(f"  prefill          {fmt(med('prefill_tps'), 1)} tokens/s")
    print(f"  decode           {fmt(med('decode_tps'), 2)} tokens/s")
    print(f"  token gap p95    {fmt(med('itl_p95_ms'), 1)} ms (median of per-prompt p95)")
    print(f"  peak RAM         {fmt(max((float(r['peak_rss_mb']) for r in rows if r['peak_rss_mb'] != ''), default=None), 0)} MB")
    short = [r for r in rows if r["n_out"] != "" and int(r["n_out"]) < args.max_tokens]
    if args.fixed_output and short:
        print(f"  WARNING: {len(short)}/{len(rows)} generations ended before {args.max_tokens} tokens; "
              "the server did not honor ignore_eos, so output lengths are not fixed.")
    b0 = [float(r["batt_start_pct"]) for r in rows if r["batt_start_pct"] != ""]
    if b0:
        temps = [float(r["batt_temp_c"]) for r in rows if r["batt_temp_c"] != ""]
        print(f"  battery          {b0[0]:.0f}% -> {float(rows[-1]['batt_end_pct']):.0f}%"
              + (f", battery temp max {max(temps):.1f} C" if temps else ""))
    print(f"\nRows appended to {CSV_PATH}")


if __name__ == "__main__":
    main()

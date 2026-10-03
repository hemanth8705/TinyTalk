"""Score the outputs of one or more quality runs against the checks in prompts/quality_prompts.json.

Usage (from the repo root):
    python scripts/score_quality.py 20261003-201500
    python scripts/score_quality.py 20261003-201500 20261003-203000   # side by side (e.g. laptop vs phone)

Each argument is a run id: the harness (--quality) wrote results/raw/<run_id>_outputs.jsonl.
Scoring is automatic and approximate (substring / regex / JSON checks): good for comparing
models and devices, not a replacement for reading the answers. Writes <run_id>_quality.csv.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"


def clean(text):
    """Drop <think> blocks and surrounding whitespace."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def normalize(text):
    return text.strip().strip("`'\".!").strip()


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def run_check(check, out):
    t = check["type"]
    if t == "contains_any":
        return any(v.lower() in out.lower() for v in check["values"])
    if t == "regex":
        flags = sum({"i": re.I, "s": re.S, "m": re.M}[c] for c in check.get("flags", "i"))
        return re.search(check["pattern"], out, flags) is not None
    if t == "equals":
        a, b = normalize(out), check["value"]
        return a == b if check.get("case") else a.lower() == b.lower()
    if t == "bullets":
        lines = [l for l in out.splitlines() if l.strip()]
        bullets = [l for l in lines if re.match(r"\s*([-*•]|\d+[.)])\s+", l)]
        return len(bullets) == check["count"] and len(lines) == len(bullets)
    if t == "word_count":
        n = len(re.findall(r"[A-Za-z0-9'À-ſ-]+", out))
        return check["min"] <= n <= check["max"]
    if t == "json_keys":
        data = extract_json(out)
        return isinstance(data, dict) and all(k in data for k in check["keys"])
    if t == "json_equals":
        return extract_json(out) == check["value"]
    raise ValueError(f"unknown check type: {t}")


def score_run(run_id, prompts):
    path = RAW_DIR / f"{run_id}_outputs.jsonl"
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    outputs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        outputs[d["prompt_id"]] = d
    meta = {}
    csv_path = RAW_DIR / f"{run_id}_benchmark.csv"
    if csv_path.exists():
        first = next(csv.DictReader(csv_path.open()), {})
        meta = {k: first.get(k, "") for k in ("device", "model", "quant", "run_type", "server_args")}
    results = []
    for p in prompts:
        d = outputs.get(p["id"])
        if d is None:
            results.append({"id": p["id"], "category": p["category"], "pass": None, "tokens": None, "output": ""})
            continue
        out = clean(d["output"])
        results.append({"id": p["id"], "category": p["category"], "tokens": len(d["token_times_s"]),
                        "pass": all(run_check(c, out) for c in p["checks"]), "output": out})
    return meta, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--prompts", type=Path, default=ROOT / "prompts" / "quality_prompts.json")
    ap.add_argument("--max-tokens", type=int, default=256, help="the cap the run used; flags answers cut off by it")
    args = ap.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))

    scored = {}
    for rid in args.run_ids:
        meta, res = score_run(rid, prompts)
        scored[rid] = res
        out_path = RAW_DIR / f"{rid}_quality.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "prompt_id", "category", "pass", "tokens", "output"])
            for r in res:
                w.writerow([rid, r["id"], r["category"], r["pass"], r["tokens"], r["output"]])
        done = [r for r in res if r["pass"] is not None]
        passed = sum(r["pass"] for r in done)
        cut = sum(1 for r in done if r["tokens"] and r["tokens"] >= args.max_tokens)
        label = f"{meta.get('device', '?')} | {meta.get('model', '?')} | {meta.get('server_args') or 'default args'}"
        print(f"\n{rid}  [{label}]")
        print(f"  overall: {passed}/{len(done)} passed ({100 * passed / max(len(done), 1):.0f}%)"
              + (f"  | {cut} answers hit the {args.max_tokens}-token cap" if cut else "") + f"  -> {out_path.name}")
        by_cat = defaultdict(list)
        for r in done:
            by_cat[r["category"]].append(r["pass"])
        for cat, vals in by_cat.items():
            print(f"    {cat:<12} {sum(vals)}/{len(vals)}")

    # per-prompt table, with disagreements between runs marked
    ids = [p["id"] for p in prompts]
    print("\nPer prompt (P = pass, . = fail, ? = missing):")
    print(f"  {'prompt':<16}" + "".join(f"{rid[-6:]:>9}" for rid in args.run_ids))
    for pid in ids:
        marks = []
        for rid in args.run_ids:
            r = next(x for x in scored[rid] if x["id"] == pid)
            marks.append("?" if r["pass"] is None else ("P" if r["pass"] else "."))
        flag = "   <-- runs disagree" if len(set(marks)) > 1 else ""
        print(f"  {pid:<16}" + "".join(f"{m:>9}" for m in marks) + flag)


if __name__ == "__main__":
    main()

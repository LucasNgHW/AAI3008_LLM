#!/usr/bin/env python3
"""
tools/profile_latency.py
------------------------
Standalone latency profiler for the NLP RAG pipeline.

Measures each stage in isolation over representative queries and prints a
breakdown table.  Run this after any significant code change to verify the
end-to-end budget is still under the 3 s target.

Usage
-----
    python tools/profile_latency.py              # warm profile, 3 reps/query
    python tools/profile_latency.py --cold       # also measure model load time
    python tools/profile_latency.py --quiet      # summary table only
    python tools/profile_latency.py --reps 5     # more reps for stable stats

Output example
--------------
    Stage                  min ms   avg ms   max ms   p95 ms
    ------               --------  -------  -------  -------
    embed_query               4.1      5.2      8.7      7.9
    qdrant_search             3.2      4.1      9.3      8.1
    rerank (miss)            42.1    487.3    612.4    608.2
    rerank (hit)              0.1      0.2      0.4      0.3
    generate (LLM)          831.0    994.1   1412.3   1380.0
    TOTAL                   880.5   1491.1   2043.1   1997.0

Notes
-----
- Uses REAL models and local Qdrant (not mocks).
- LLM stage requires GEMINI_API_KEY to be set.
- Cold-start timing spawns a subprocess for a clean import state.
- If Qdrant is empty, search returns 0 results and rerank is skipped.
"""

import argparse
import os
import sys
import time
import statistics
from typing import Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

QUERIES = [
    "What is tokenisation in NLP?",
    "Explain self-attention in the transformer architecture.",
    "What are word embeddings and how are they trained?",
    "How does BERT use masked language modelling?",
    "What is the difference between encoder and decoder in transformers?",
]

BUDGET_MS = 3000.0


# ── Utilities ─────────────────────────────────────────────────────────────────

def timeit(fn: Callable, *args, **kwargs) -> tuple:
    t0 = time.perf_counter()
    r  = fn(*args, **kwargs)
    return r, time.perf_counter() - t0


def _stats(times_s: list) -> tuple:
    """Return (min, mean, max, p95) in milliseconds."""
    if not times_s:
        return (0.0, 0.0, 0.0, 0.0)
    ms  = [t * 1000 for t in times_s]
    p95 = sorted(ms)[max(0, int(len(ms) * 0.95) - 1)]
    return min(ms), statistics.mean(ms), max(ms), p95


def _row(label: str, times_s: list, flag: bool = False) -> str:
    if not times_s:
        return f"  {label:<22}  {'(no data)':>8}"
    mn, av, mx, p95 = _stats(times_s)
    mark = "  ⚠ BOTTLENECK" if flag else ""
    return f"  {label:<22}  {mn:>8.1f}  {av:>8.1f}  {mx:>8.1f}  {p95:>8.1f}{mark}"


def _table(rows: list) -> None:
    header = f"  {'Stage':<22}  {'min ms':>8}  {'avg ms':>8}  {'max ms':>8}  {'p95 ms':>8}"
    sep    = "  " + "-" * 62
    print(sep)
    print(header)
    print(sep)
    for label, times, over in rows:
        print(_row(label, times, over))
    print(sep)


# ── Cold-start profiler ───────────────────────────────────────────────────────

def profile_cold_start(quiet: bool) -> None:
    import subprocess, json
    script = (
        "import time, json, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "import pipeline.embedder as emb\n"
        "import rag.reranker as rr\n"
        "emb._model = None; rr._reranker = None\n"
        "t0 = time.perf_counter(); emb.get_model();   e = (time.perf_counter()-t0)*1000\n"
        "t0 = time.perf_counter(); rr.get_reranker(); r = (time.perf_counter()-t0)*1000\n"
        'print(json.dumps({"embed_ms": e, "rerank_ms": r}))\n'
    )
    try:
        res  = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=120)
        data = {}
        for line in res.stdout.splitlines():
            try:
                import json as _j
                data.update(_j.loads(line))
            except Exception:
                pass
        if data:
            e = data.get("embed_ms", 0)
            r = data.get("rerank_ms", 0)
            print(f"\n  Cold-start model load times:")
            print(f"    Embedding model : {e:.0f} ms")
            print(f"    Reranker model  : {r:.0f} ms")
            print(f"    Combined        : {e+r:.0f} ms  (paid once per process, not per query)")
    except Exception as exc:
        print(f"  Cold-start profiling failed: {exc}")


# ── Warm stage profiler ───────────────────────────────────────────────────────

def profile_stages(reps: int, quiet: bool) -> dict:
    from pipeline.embedder import embed_query, warmup as ewarmup
    from rag.reranker      import rerank, warmup as rwarmup, clear_cache
    from rag.retriever     import retrieve
    from rag.generator     import generate_answer

    print("  Warming up models...")
    ewarmup()
    rwarmup()
    print("  Models warm.  Profiling...\n")

    times = {k: [] for k in ("embed", "search", "rerank_miss", "rerank_hit", "generate", "total")}

    for q in QUERIES:
        clear_cache()
        for rep in range(reps):
            t_total = time.perf_counter()

            _,      t_embed  = timeit(embed_query, q)
            chunks, t_search = timeit(retrieve, q, top_k=15)

            t_rerank = 0.0
            if chunks:
                if rep == 0:
                    clear_cache()
                    _, t_rerank = timeit(rerank, q, chunks, top_n=5)
                    times["rerank_miss"].append(t_rerank)
                else:
                    _, t_rerank = timeit(rerank, q, chunks, top_n=5)
                    times["rerank_hit"].append(t_rerank)
                top5 = chunks[:5]
            else:
                top5 = []

            _, t_gen = timeit(
                generate_answer, q, top5,
                user_profile={"preferred_difficulty": "intermediate", "top_topics": []},
            )

            elapsed = time.perf_counter() - t_total
            times["embed"].append(t_embed)
            times["search"].append(t_search)
            times["generate"].append(t_gen)
            times["total"].append(elapsed)

            if not quiet:
                print(
                    f"  {q[:36]:<36}  rep {rep+1}  "
                    f"emb={t_embed*1e3:.0f}ms "
                    f"srch={t_search*1e3:.0f}ms "
                    f"rrnk={t_rerank*1e3:.0f}ms "
                    f"gen={t_gen*1e3:.0f}ms "
                    f"tot={elapsed*1e3:.0f}ms"
                )

    return times


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Profile NLP Assistant pipeline latency.")
    ap.add_argument("--cold",  action="store_true", help="Measure cold model load times")
    ap.add_argument("--quiet", action="store_true", help="Summary table only")
    ap.add_argument("--reps",  type=int, default=3,  help="Reps per query (default 3)")
    args = ap.parse_args()

    print(f"\n{'='*65}")
    print(f"  NLP Assistant — Latency Profiler")
    print(f"  Budget: <{BUDGET_MS:.0f} ms  |  {len(QUERIES)} queries  |  {args.reps} reps each")
    print(f"{'='*65}")

    if args.cold:
        profile_cold_start(args.quiet)

    times = profile_stages(reps=args.reps, quiet=args.quiet)

    avg_total  = statistics.mean(times["total"]) * 1000 if times["total"] else 0
    over_total = avg_total > BUDGET_MS

    def over(key: str, threshold_ms: float) -> bool:
        t = times.get(key, [])
        return bool(t) and statistics.mean(t) * 1000 > threshold_ms

    rows = [
        ("embed_query",    times["embed"],        over("embed",        50)),
        ("qdrant_search",  times["search"],       over("search",       50)),
        ("rerank (miss)",  times["rerank_miss"],  over("rerank_miss",  800)),
        ("rerank (hit)",   times["rerank_hit"],   False),
        ("generate (LLM)", times["generate"],     over("generate",    2000)),
        ("TOTAL",          times["total"],        over_total),
    ]

    print(f"\n  Results — {args.reps} reps × {len(QUERIES)} queries")
    _table(rows)

    verdict = "✓  WITHIN BUDGET" if not over_total else "✗  OVER BUDGET"
    print(f"\n  Avg total: {avg_total:.0f} ms  →  {verdict}  (target <{BUDGET_MS:.0f} ms)\n")

    if over_total:
        ranked = sorted(
            [(k, statistics.mean(v) * 1000) for k, v in times.items()
             if v and k not in ("total",)],
            key=lambda x: x[1], reverse=True,
        )
        print("  Biggest contributors:")
        for name, avg in ranked[:3]:
            pct = avg / avg_total * 100
            print(f"    {name:<20} {avg:>7.0f} ms  ({pct:.0f}% of total)")
        print()


if __name__ == "__main__":
    main()

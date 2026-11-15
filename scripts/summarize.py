"""Summarise held-out evaluation: success with Wilson 95% CI, efficiency, paired comparisons.
Usage: python scripts/summarize.py phase5_results.json
"""
import sys, json, math, collections

rows = json.load(open(sys.argv[1]))
SC = ["nominal", "grasp_fail", "displace", "obstacles"]
MODES = ["open_loop", "verify_retry", "no_wm", "echo_detect_only", "echo"]


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_test_p(b, c):
    """Two-sided exact McNemar/sign test on discordant pairs (b: A wins, c: B wins)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


by = collections.defaultdict(dict)
for r in rows:
    by[(r["scenario"], r["mode"])][r["i"]] = r

print("SUCCESS RATE (95% Wilson CI) | mean actions | mean detections")
print(f"{'scenario':11s} " + " ".join(f"{m:>26s}" for m in MODES))
for sc in SC:
    cells = []
    for m in MODES:
        rs = list(by[(sc, m)].values()); k = sum(r["success"] for r in rs); n = len(rs)
        lo, hi = wilson(k, n)
        cells.append(f"{k/n:4.2f} [{lo:.2f},{hi:.2f}] a={sum(r['n_actions'] for r in rs)/n:3.1f} d={sum(r['detections'] for r in rs)/n:3.1f}")
    print(f"{sc:11s} " + " ".join(f"{c:>26s}" for c in cells))

print("\nPAIRED COMPARISONS on identical layouts + injections (A-only wins / B-only wins, sign-test p)")
for a, b in (("echo", "open_loop"), ("echo", "verify_retry"), ("echo", "no_wm"), ("echo", "echo_detect_only")):
    for sc in SC:
        ids = sorted(set(by[(sc, a)]) & set(by[(sc, b)]))
        w_a = sum(by[(sc, a)][i]["success"] and not by[(sc, b)][i]["success"] for i in ids)
        w_b = sum(by[(sc, b)][i]["success"] and not by[(sc, a)][i]["success"] for i in ids)
        print(f"  {a:6s} vs {b:16s} {sc:11s}: {w_a:2d} / {w_b:2d}   p={sign_test_p(w_a, w_b):.3f}")

print("\nSTEP-LEVEL DETECTION (flagged vs truly bad step)")
for m in ("no_wm", "echo"):
    fl = [f for r in rows if r["mode"] == m for f in r["flags"]]
    bad = [f[0] for f in fl if f[1]]; good = [f[0] for f in fl if not f[1]]
    print(f"  {m:6s}: recall {sum(bad)/len(bad):.2f} (n={len(bad)}), false-alarm {sum(good)/len(good):.2f} (n={len(good)})")

inf = [r for r in rows if r["mode"] == "echo" and r["scenario"] == "nominal" and not r["success"]]
print(f"\nnominal echo failures: {len(inf)} of {len(by[('nominal','echo')])}")

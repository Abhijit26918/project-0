"""Push-task evaluation: rule vs world-model command selection, open loop vs feedback retries.
Usage: python scripts/eval_push.py <n_tasks> [workers] [tag]
"""
import sys, pathlib, json, math, collections, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool

MODES = ["rule_open", "curve_open", "wm_open", "rule_retry", "adaptive_retry", "curve_retry", "wm_retry"]


def worker(idxs):
    from echo import ManipulationEnv
    from echo.pushtask import make_push_task, run_push_episode, fit_response_curves
    from echo.worldmodel import WorldModel
    wm = WorldModel.load("checkpoints/wm_push.pt"); curves = fit_response_curves()
    env = ManipulationEnv(seed=0); rows = []
    for i in idxs:
        instr, layout, subj = make_push_task(i)
        for mode in MODES:
            env.reset(slippery=True, layout=layout)
            ep = run_push_episode(env, instr, mode, wm, curves=curves)
            rows.append({"i": i, "mode": mode, "subject": subj, "success": ep.success, "pushes": ep.pushes,
                         "first_ok": ep.first_push_ok})
    env.close()
    return rows


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


if __name__ == "__main__":
    n = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tag = sys.argv[3] if len(sys.argv) > 3 else "push"
    t = time.time()
    with Pool(workers) as pool:
        rows = [r for part in pool.map(worker, [list(range(w, n, workers)) for w in range(workers)]) for r in part]
    json.dump(rows, open(f"results/{tag}_results.json", "w"))
    print(f"{len(rows)} runs in {time.time()-t:.0f}s\n")
    by = collections.defaultdict(dict)
    for r in rows: by[r["mode"]][r["i"]] = r
    print(f"{'mode':11s} {'success':>8s} {'95% CI':>14s} {'first push ok':>14s} {'mean pushes':>12s}")
    for m in MODES:
        rs = list(by[m].values()); k = sum(r["success"] for r in rs); lo, hi = wilson(k, len(rs))
        print(f"{m:14s} {k/len(rs):8.2f} [{lo:.2f},{hi:.2f}] {np.mean([r['first_ok'] for r in rs]):14.2f} "
              f"{np.mean([r['pushes'] for r in rs]):12.2f}")
    print("\nby pushed object (success):")
    for subj in ("red_cube", "green_cube", "blue_cube"):
        print(f"  {subj:11s} " + "  ".join(f"{m}={np.mean([r['success'] for r in by[m].values() if r['subject']==subj]):.2f}" for m in MODES))
    print("\npaired (A-only wins / B-only wins, sign-test p) on identical tasks:")
    for a, b in (("wm_open", "rule_open"), ("wm_open", "curve_open"), ("wm_retry", "rule_retry"),
                 ("wm_retry", "adaptive_retry"), ("wm_retry", "curve_retry")):
        ids = sorted(set(by[a]) & set(by[b]))
        wa = sum(by[a][i]["success"] and not by[b][i]["success"] for i in ids)
        wb = sum(by[b][i]["success"] and not by[a][i]["success"] for i in ids)
        print(f"  success  {a} vs {b}: {wa} / {wb}  p={sign_p(wa, wb):.4f}")
    for other in ("rule_retry", "adaptive_retry", "curve_retry"):
        ids = sorted(set(by["wm_retry"]) & set(by[other]))
        fewer = sum(by["wm_retry"][i]["pushes"] < by[other][i]["pushes"] for i in ids)
        more = sum(by["wm_retry"][i]["pushes"] > by[other][i]["pushes"] for i in ids)
        print(f"  pushes needed  wm_retry fewer / more than {other}: {fewer} / {more}  p={sign_p(fewer, more):.4f}")

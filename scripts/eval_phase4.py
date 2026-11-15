"""Phase 4 evaluation: modes x failure scenarios, paired layouts/injections.
Usage: python scripts/eval_phase4.py <episodes_per_cell> [workers]
"""
import sys, pathlib, json, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool

TASKS = [
    "Pick up the red cube, place it inside the blue container, and move the green cube to the left of the red cube.",
    "Put the green cube in the blue container",
    "Move the red cube to the right of the green cube",
]
FIRST_SUBJECT = ["red_cube", "green_cube", "red_cube"]
SCENARIOS = ["nominal", "grasp_fail", "displace", "obstacles"]
MODES = ["open_loop", "verify_retry", "no_wm", "echo_detect_only", "echo"]


def make_injection(scenario, i):
    from echo.recovery import Injection
    rng = np.random.default_rng(50000 + i)
    if scenario == "grasp_fail":
        return Injection(grasp_fail_pick=int(rng.integers(0, 2)) if i % 3 == 0 else 0)
    if scenario == "displace":
        ang = rng.uniform(0, 2 * np.pi)
        return Injection(disturb_after=2, disturb_obj=FIRST_SUBJECT[i % 3],
                         disturb_dxy=(0.10 * float(np.cos(ang)), 0.10 * float(np.sin(ang))))
    return Injection()


def worker(args):
    idxs, thr, ckpt, modes, use_percep = args
    from echo import ManipulationEnv
    from echo.recovery import run_episode
    from echo.worldmodel import WorldModel
    wm = WorldModel.load(ckpt)
    env = ManipulationEnv(seed=0)
    from echo.perception import Perception
    per = Perception(env.img_size) if use_percep else None
    rows = []
    for i in idxs:
        for sc in SCENARIOS:
            for mode in modes:
                env.rng = np.random.default_rng(1000 + i)          # identical layout for every mode
                env.reset(obstacles=2 if sc == "obstacles" else 0)
                ep = run_episode(env, TASKS[i % 3], mode, wm, thr, make_injection(sc, i), perception=per)
                rows.append({"i": i, "scenario": sc, "mode": mode, "success": ep.success, "n_actions": ep.n_actions,
                             "detections": ep.detections, "replans": ep.replans, "injected": ep.injected,
                             "flags": ep.step_flags})
    env.close()
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/wm.pt"
    thr = float(sys.argv[4]) if len(sys.argv) > 4 else json.load(open("detector.json"))["threshold_m"]
    modes = sys.argv[5].split(",") if len(sys.argv) > 5 else MODES
    tag = sys.argv[6] if len(sys.argv) > 6 else "phase4"
    use_percep = "--perception" in sys.argv
    start = int(sys.argv[7]) if len(sys.argv) > 7 else 0      # first episode index (use a fresh range for held-out layouts)
    chunks = [list(range(start + w, start + n, workers)) for w in range(workers)]
    t = time.time()
    with Pool(workers) as pool:
        rows = [r for part in pool.map(worker, [(c, thr, ckpt, modes, use_percep) for c in chunks]) for r in part]
    json.dump(rows, open(f"{tag}_results.json", "w"))
    print(f"{len(rows)} runs in {time.time()-t:.0f}s, detector threshold {thr*100:.1f} cm\n")

    print(f"{'scenario':11s} {'mode':17s} {'success':>9s} {'actions':>8s} {'detects':>8s}")
    for sc in SCENARIOS:
        for mode in modes:
            r = [x for x in rows if x["scenario"] == sc and x["mode"] == mode]
            print(f"{sc:11s} {mode:17s} {np.mean([x['success'] for x in r]):9.2f} "
                  f"{np.mean([x['n_actions'] for x in r]):8.1f} {np.mean([x['detections'] for x in r]):8.2f}"
                  f"   (n={len(r)})")
        print()
    # step-level detector quality (world-model monitor, full system)
    for mode in [m for m in ("no_wm", "echo") if m in modes]:
        fl = [f for x in rows if x["mode"] == mode for f in x["flags"]]
        bad = [f[0] for f in fl if f[1]]; good = [f[0] for f in fl if not f[1]]
        print(f"[{mode}] step-level detection: recall on bad steps {np.mean(bad):.2f} (n={len(bad)}), "
              f"false-alarm rate on good steps {np.mean(good):.2f} (n={len(good)})")

"""Push transitions on the task distribution (aligned two-cube layouts), for the push world model.
Usage: python scripts/gen_push_task_data.py <n_episodes> <out.npz> [seed0] [workers]
Layout seeds are drawn from a range disjoint from the evaluation tasks (eval uses task indices 0..999).
"""
import sys, pathlib, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool


def worker(args):
    seed, n_eps = args
    from echo import ManipulationEnv
    from echo.pushtask import make_push_task
    from echo.wm_data import encode_state, encode_action_v2
    rng = np.random.default_rng(seed)
    env = ManipulationEnv(seed=seed)
    S, A, S2, EP = [], [], [], []
    for ep in range(n_eps):
        instr, layout, subj = make_push_task(1_000_000 + seed * 1000 + ep)
        ref = next(k for k in layout if k.endswith("_cube") and k != subj and instr.split("the ")[-1].startswith(k.split("_")[0]))
        env.reset(slippery=True, layout=layout)
        for k in range(3):
            st = env.get_state()
            y_s, y_r = st.objects[subj].position[1], st.objects[ref].position[1]
            toward = -1.0 if y_s > y_r else 1.0
            if k > 0 and rng.random() < 0.3:
                toward = -toward                                    # sometimes push away (recovery-like situations)
            sgn = toward
            cmd = float(rng.uniform(0.03, 0.26)) if k == 0 else float(rng.uniform(0.03, 0.20))
            if abs(y_s + sgn * (cmd + 0.10)) > 0.34:
                sgn = -sgn
            act = {"type": "push", "object": subj, "direction": (0.0, sgn), "distance": cmd}
            s0, a = encode_state(st), encode_action_v2(act, st)
            st2, _ = env.step(act)
            S.append(s0); A.append(a); S2.append(encode_state(st2)); EP.append(seed * 100000 + ep)
    env.close()
    return np.array(S), np.array(A), np.array(S2), np.array(EP)


if __name__ == "__main__":
    n_eps = int(sys.argv[1]); out = sys.argv[2]
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 11000
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    per = int(np.ceil(n_eps / workers)); t = time.time()
    with Pool(workers) as pool:
        parts = pool.map(worker, [(seed0 + i, per) for i in range(workers)])
    S, A, S2, EP = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(out, S=S, A=A, S2=S2, ep=EP, ok=np.ones(len(S), dtype=bool))
    print(f"{len(S)} push transitions from {workers * per} episodes in {time.time()-t:.0f}s -> {out}")

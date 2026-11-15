"""Generate push transitions in slippery-physics scenes for the push world model.
Usage: python scripts/gen_push_data.py <n_episodes> <out.npz> [seed0] [workers]
Pushes are along +/-y with commanded distance U[0.04, 0.35]; objects have hidden, colour-dependent friction.
"""
import sys, pathlib, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool

CUBES = ["red_cube", "green_cube", "blue_cube"]


def worker(args):
    seed, n_eps, pushes = args
    from echo import ManipulationEnv
    from echo.wm_data import encode_state, encode_action_v2
    rng = np.random.default_rng(seed)
    env = ManipulationEnv(seed=seed)
    S, A, S2, EP = [], [], [], []
    for ep in range(n_eps):
        env.rng = np.random.default_rng(seed * 1000 + ep)
        env.reset(slippery=True)
        for _ in range(pushes):
            st = env.get_state()
            name = str(rng.choice(CUBES))
            y = st.objects[name].position[1]
            cmd = float(rng.uniform(0.04, 0.35))
            sgn = 1.0 if rng.random() < 0.5 else -1.0
            if abs(y + sgn * (cmd + 0.12)) > 0.36:          # keep the sweep inside the reachable workspace
                sgn = -sgn
            act = {"type": "push", "object": name, "direction": (0.0, sgn), "distance": cmd}
            s0, a = encode_state(st), encode_action_v2(act, st)
            st2, _ = env.step(act)
            S.append(s0); A.append(a); S2.append(encode_state(st2)); EP.append(seed * 100000 + ep)
    env.close()
    return np.array(S), np.array(A), np.array(S2), np.array(EP)


if __name__ == "__main__":
    n_eps = int(sys.argv[1]); out = sys.argv[2]
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, (os.cpu_count() or 2) - 1)
    per = int(np.ceil(n_eps / workers)); t = time.time()
    with Pool(workers) as pool:
        parts = pool.map(worker, [(seed0 + i, per, 3) for i in range(workers)])
    S, A, S2, EP = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(out, S=S, A=A, S2=S2, ep=EP, ok=np.ones(len(S), dtype=bool))
    print(f"{len(S)} push transitions from {workers * per} episodes in {time.time()-t:.0f}s -> {out}")

"""Generate (state, action, next_state) transitions for the world model.
Usage: python scripts/gen_data.py <n_episodes> <out.npz> [seed0] [workers]
"""
import sys, pathlib, time, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from multiprocessing import Pool

CUBES = ["red_cube", "green_cube", "blue_cube"]


def worker(args):
    seed, n_eps, steps = args
    from echo import ManipulationEnv
    from echo.wm_data import encode_state, encode_action
    rng = np.random.default_rng(seed)
    env = ManipulationEnv(seed=seed)
    S, A, S2, OK, EP = [], [], [], [], []
    for ep in range(n_eps):
        env.reset()
        for _ in range(steps):
            st = env.get_state()
            if rng.random() < 0.15:                   # unexpected displacement of a free object
                name = str(rng.choice(CUBES))
                if not st.objects[name].held:
                    env.disturb(name, dxy=tuple(rng.uniform(-0.08, 0.08, 2)))
                    st = env.get_state()
            held = env._held
            if held is not None:
                if rng.random() < 0.4:
                    act = {"type": "place_in", "container": "blue_container"}
                else:
                    act = {"type": "place", "target": (float(rng.uniform(0.35, 0.75)), float(rng.uniform(-0.3, 0.3)))}
            else:
                if rng.random() < 0.9:
                    act = {"type": "pick", "object": str(rng.choice(CUBES))}
                    env.force_grasp_fail = bool(rng.random() < 0.15)   # injected failed grasp
                else:
                    act = {"type": "place", "target": (float(rng.uniform(0.35, 0.75)), float(rng.uniform(-0.3, 0.3)))}
            s0 = encode_state(st); a = encode_action(act, st)
            st2, info = env.step(act)
            S.append(s0); A.append(a); S2.append(encode_state(st2)); OK.append(info["success"])
            EP.append(seed * 100000 + ep)
    env.close()
    return np.array(S), np.array(A), np.array(S2), np.array(OK), np.array(EP)


if __name__ == "__main__":
    n_eps = int(sys.argv[1]); out = sys.argv[2]
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else max(1, (os.cpu_count() or 2) - 1)
    steps = 4
    per = int(np.ceil(n_eps / workers))
    t = time.time()
    with Pool(workers) as pool:
        parts = pool.map(worker, [(seed0 + i, per, steps) for i in range(workers)])
    S, A, S2, OK, EP = (np.concatenate(x) for x in zip(*parts))
    np.savez_compressed(out, S=S, A=A, S2=S2, ok=OK, ep=EP)
    print(f"{len(S)} transitions from {workers * per} episodes in {time.time()-t:.0f}s "
          f"(action success rate {OK.mean():.2f}) -> {out}")

"""Multi-step (open-loop rollout) prediction: feed the model its own predictions back in.
Evaluated on held-out test episodes, on windows where the real trajectory is continuous
(no external disturbance between steps) and every action in the window succeeded.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo.worldmodel import WorldModel
from echo.wm_data import OBJECTS

d = np.load("data_train.npz"); S, A, S2, ok, ep = d["S"], d["A"], d["S2"], d["ok"], d["ep"]
eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps); n = len(eps)
test_eps = eps[int(.9 * n):]
wm = WorldModel.load("checkpoints/wm_nominal.pt")
pos = [4 * i + j for i in range(len(OBJECTS)) for j in range(3)]
H = 4

windows = []                       # (start index, horizon) windows inside one episode, continuous + successful
for e in test_eps:
    idx = np.where(ep == e)[0]
    for a in range(len(idx)):
        for h in range(1, H + 1):
            seg = idx[a:a + h]
            if len(seg) < h: break
            continuous = all(np.abs(S[seg[k + 1]] - S2[seg[k]]).max() < 1e-4 for k in range(h - 1))
            if continuous and ok[seg].all():
                windows.append((a, h, seg))
            else:
                break

res = {}
print(f"{'horizon':>8s} {'windows':>8s} {'rollout err (cm)':>18s} {'copy-state err (cm)':>20s}")
for h in range(1, H + 1):
    err, copy = [], []
    for _, hh, seg in windows:
        if hh != h: continue
        s = S[seg[0]][None]
        for k in range(h):                                   # roll the model forward on its own predictions
            s, _ = wm.predict(s, A[seg[k]][None])
        truth = S2[seg[-1]]
        e = np.linalg.norm((s[0] - truth)[pos].reshape(-1, 3), axis=1)
        c = np.linalg.norm((S[seg[0]] - truth)[pos].reshape(-1, 3), axis=1)
        err.append(e.mean()); copy.append(c.mean())
    if err:
        res[h] = {"n": len(err), "rollout_cm": float(np.mean(err) * 100), "copy_cm": float(np.mean(copy) * 100)}
        print(f"{h:8d} {len(err):8d} {res[h]['rollout_cm']:18.2f} {res[h]['copy_cm']:20.2f}")
json.dump(res, open("results/multistep.json", "w"), indent=1)

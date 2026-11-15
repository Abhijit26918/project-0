"""Train the push world model and compare with the "moves as commanded" assumption.
Usage: python scripts/train_push_wm.py data_push.npz [epochs]
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo.worldmodel import WorldModel
from echo.wm_data import ACTION_DIM_V2, OBJECTS

d = np.load(sys.argv[1]); epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
S, A, S2, ep = d["S"], d["A"], d["S2"], d["ep"]
eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps); n = len(eps)
m_tr = np.isin(ep, eps[:int(.8 * n)]); m_te = np.isin(ep, eps[int(.9 * n):])
print(f"push transitions: train {m_tr.sum()}  test {m_te.sum()}")

wm = WorldModel(n_models=5, action_dim=ACTION_DIM_V2)
wm.fit_norm(S[m_tr], A[m_tr], S2[m_tr]); wm.train(S[m_tr], A[m_tr], S2[m_tr], epochs=epochs, verbose=False)
pathlib.Path("checkpoints").mkdir(exist_ok=True); wm.save("checkpoints/wm_push.pt")

pred, _ = wm.predict(S[m_te], A[m_te])
obj = A[m_te][:, 4:8].argmax(1)                     # pushed object index (after 4 type slots)
cmd = A[m_te][:, -1]                                # signed commanded displacement along y
yi = lambda k: 4 * k + 1
rows = np.arange(m_te.sum())
y0 = S[m_te][rows, [yi(k) for k in obj]]
true_dy = S2[m_te][rows, [yi(k) for k in obj]] - y0
pred_dy = pred[rows, [yi(k) for k in obj]] - y0
print(f"\n{'object':10s} {'n':>4s} | err of 'moves as commanded' | err of world model   (cm, along-y displacement)")
res = {}
for k in range(3):
    s = obj == k
    e_cmd = np.abs(true_dy[s] - cmd[s]).mean() * 100; e_wm = np.abs(true_dy[s] - pred_dy[s]).mean() * 100
    res[OBJECTS[k]] = {"n": int(s.sum()), "assume_commanded_cm": float(e_cmd), "world_model_cm": float(e_wm)}
    print(f"{OBJECTS[k]:10s} {s.sum():4d} | {e_cmd:24.1f} | {e_wm:18.1f}")
e_cmd = np.abs(true_dy - cmd).mean() * 100; e_wm = np.abs(true_dy - pred_dy).mean() * 100
print(f"{'all':10s} {len(cmd):4d} | {e_cmd:24.1f} | {e_wm:18.1f}")
res["all"] = {"assume_commanded_cm": float(e_cmd), "world_model_cm": float(e_wm)}
json.dump(res, open("results/push_wm.json", "w"), indent=1)

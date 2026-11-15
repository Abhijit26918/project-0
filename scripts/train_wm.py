"""Train and evaluate the world model. Usage: python scripts/train_wm.py data_train.npz [epochs]"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo.worldmodel import WorldModel
from echo.wm_data import OBJECTS

d = np.load(sys.argv[1])
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NOMINAL = "--nominal" in sys.argv          # train on successful transitions only: predicts the *expected* outcome
CKPT = "checkpoints/wm_nominal.pt" if NOMINAL else "checkpoints/wm.pt"
S, A, S2, ok, ep = d["S"], d["A"], d["S2"], d["ok"], d["ep"]

# split by episode so no episode leaks between train and test
eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps)
n = len(eps); tr_e, va_e, te_e = eps[:int(.8*n)], eps[int(.8*n):int(.9*n)], eps[int(.9*n):]
m_tr, m_va, m_te = (np.isin(ep, e) for e in (tr_e, va_e, te_e))
print(f"transitions: train {m_tr.sum()}  val {m_va.sum()}  test {m_te.sum()}")

if "--eval" in sys.argv:                      # skip training, load saved checkpoint
    wm = WorldModel.load(CKPT)
else:
    wm = WorldModel(n_models=5)
    m_fit = m_tr & ok if NOMINAL else m_tr
    print(f"fitting on {m_fit.sum()} transitions ({'successful only' if NOMINAL else 'all outcomes'})")
    wm.fit_norm(S[m_fit], A[m_fit], S2[m_fit])
    wm.train(S[m_fit], A[m_fit], S2[m_fit], epochs=epochs)
    pathlib.Path("checkpoints").mkdir(exist_ok=True)
    wm.save(CKPT)

# ---- evaluation ---------------------------------------------------------------
pos_idx = [4 * i + j for i in range(len(OBJECTS)) for j in range(3)]   # xyz of every object


def obj_err(pred, true):          # mean Euclidean position error per object (metres)
    p = pred[:, pos_idx].reshape(len(pred), -1, 3); t = true[:, pos_idx].reshape(len(true), -1, 3)
    return np.linalg.norm(p - t, axis=2)      # (N, n_objects)


def report(name, m):
    pred, sd = wm.predict(S[m], A[m])
    e_model = obj_err(pred, S2[m]); e_copy = obj_err(S[m], S2[m])
    print(f"\n[{name}] N={m.sum()}")
    print(f"  mean object position error  model: {e_model.mean()*100:.2f} cm   copy-state baseline: {e_copy.mean()*100:.2f} cm")
    for lab, mk in (("successful actions", ok[m]), ("failed actions", ~ok[m])):
        if mk.sum():
            print(f"  {lab:20s} (n={mk.sum():4d})  model {e_model[mk].mean()*100:.2f} cm   copy {e_copy[mk].mean()*100:.2f} cm")
    # does ensemble disagreement track actual error?
    unc = sd[:, pos_idx].mean(1); err = e_model.mean(1)
    # manual Pearson (np.corrcoef crashes here: BLAS DLL clash between pip torch and conda numpy)
    corr = float(np.mean((unc - unc.mean()) * (err - err.mean())) / (unc.std() * err.std() + 1e-12))
    print(f"  corr(ensemble uncertainty, error): {corr:.2f}")
    return {"model_cm": float(e_model.mean() * 100), "copy_cm": float(e_copy.mean() * 100)}


res = {"val": report("val", m_va), "test": report("test", m_te)}
json.dump(res, open("wm_results_nominal.json" if NOMINAL else "wm_results.json", "w"), indent=1)

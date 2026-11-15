"""Calibrate the prediction-discrepancy failure detector.
discrepancy e_t = max over objects of || predicted position - actual position ||  (metres).
Threshold chosen on the validation split (max Youden J), reported on the test split.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo.worldmodel import WorldModel
from echo.wm_data import OBJECTS

d = np.load("data_train.npz")
S, A, S2, ok, ep = d["S"], d["A"], d["S2"], d["ok"], d["ep"]
eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps)   # same split as train_wm.py
n = len(eps); va_e, te_e = eps[int(.8*n):int(.9*n)], eps[int(.9*n):]
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/wm.pt"
DET = sys.argv[2] if len(sys.argv) > 2 else "detector.json"
wm = WorldModel.load(CKPT)
pos_idx = [4 * i + j for i in range(len(OBJECTS)) for j in range(3)]


def discrepancy(S_, A_, S2_):
    pred, _ = wm.predict(S_, A_)
    p = pred[:, pos_idx].reshape(len(pred), -1, 3); t = S2_[:, pos_idx].reshape(len(S2_), -1, 3)
    return np.linalg.norm(p - t, axis=2).max(1)


def auroc(score, bad):   # rank-based, no BLAS
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    nb, ng = bad.sum(), (~bad).sum()
    return float((ranks[bad].sum() - nb * (nb + 1) / 2) / (nb * ng))


def split(e_list):
    # "surprising failure" = a pick that failed (the model cannot foresee a slipped grasp).
    # Failed places with nothing held are *predictable* from the state (model correctly predicts no change),
    # so they are not detection targets and are counted as ordinary actions here.
    m = np.isin(ep, e_list)
    surprising = (~ok[m]) & (A[m][:, 0] == 1)      # action type one-hot index 0 == pick
    return discrepancy(S[m], A[m], S2[m]), surprising


e_va, bad_va = split(va_e); e_te, bad_te = split(te_e)
cands = np.linspace(0.01, 0.25, 241)
# lowest threshold whose false-alarm rate on validation is <= 2%: most sensitive detector that stays quiet.
# (Youden's J picks a threshold at the edge of the failed-grasp errors and would miss smaller surprises
#  such as an object knocked ~10 cm.)
thr = next(float(t) for t in cands if (e_va[~bad_va] > t).mean() <= 0.02)
print(f"validation: chosen threshold {thr*100:.1f} cm  (AUROC {auroc(e_va, bad_va):.3f})")
for name, e, bad in (("val", e_va, bad_va), ("test", e_te, bad_te)):
    print(f"[{name}] AUROC {auroc(e, bad):.3f} | failed-pick recall {(e[bad] > thr).mean():.2f} "
          f"(n={bad.sum()}) | false-alarm rate on other actions {(e[~bad] > thr).mean():.2f} (n={(~bad).sum()})")
    print(f"       discrepancy median  good {np.median(e[~bad])*100:.1f} cm   failed {np.median(e[bad])*100:.1f} cm")
json.dump({"threshold_m": thr}, open(DET, "w"))

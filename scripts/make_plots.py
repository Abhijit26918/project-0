"""Figures for the report -> results/figures/.  Run from project root with the echo env."""
import sys, pathlib, json, math, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from echo.worldmodel import WorldModel
from echo.wm_data import OBJECTS

OUT = pathlib.Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# validated categorical slots 1-5, fixed order (never cycled); one colour per entity
C = {"open_loop": "#2a78d6", "verify_retry": "#eb6834", "no_wm": "#1baf7a", "echo_detect_only": "#eda100", "echo": "#e87ba4"}
LABEL = {"open_loop": "Open loop (Baseline A)", "verify_retry": "Verify + retry", "no_wm": "Rules + recovery",
         "echo_detect_only": "World-model detect only", "echo": "ECHO (world model + recovery)"}
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "font.size": 10, "legend.frameon": False})

d = np.load("data_train.npz"); S, A, S2, ok, ep = d["S"], d["A"], d["S2"], d["ok"], d["ep"]
eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps); n = len(eps)
te = np.isin(ep, eps[int(.9 * n):])                      # held-out test episodes (never trained on)
wm = WorldModel.load("checkpoints/wm_nominal.pt")
pred, _ = wm.predict(S[te], A[te]); St, At, S2t, okt = S[te], A[te], S2[te], ok[te]
pos = [4 * i + j for i in range(4) for j in range(3)]
err = np.linalg.norm((pred - S2t)[:, pos].reshape(-1, 4, 3), axis=2)          # (N, objects) metres
disc = err.max(1)
atype = At[:, :3].argmax(1)                                                   # 0 pick, 1 place, 2 place_in

# ---- Fig 1: prediction quality ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
true_move = np.linalg.norm((S2t - St)[:, pos].reshape(-1, 4, 3)[..., :2], axis=2)
pred_move = np.linalg.norm((pred - St)[:, pos].reshape(-1, 4, 3)[..., :2], axis=2)
m = (true_move > 0.01) & okt[:, None]
ax[0].scatter(true_move[m] * 100, pred_move[m] * 100, s=14, color=C["open_loop"], alpha=0.6, linewidths=0)
lim = max(true_move[m].max(), pred_move[m].max()) * 100 * 1.05
ax[0].plot([0, lim], [0, lim], color=INK2, lw=1, ls="--"); ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
ax[0].set_xlabel("actual horizontal displacement (cm)"); ax[0].set_ylabel("predicted displacement (cm)")
ax[0].set_title("Predicted vs actual movement\n(objects that moved >1 cm, successful actions, test set)", fontsize=10, loc="left")
names = ["pick", "place", "place_in"]
data, labs = [], []
for k, nm in enumerate(names):
    s = (atype == k) & okt
    data.append(disc[s] * 100); labs.append(f"{nm}\n(n={s.sum()})")
fp = (atype == 0) & ~okt
data.append(disc[fp] * 100); labs.append(f"failed pick\n(n={fp.sum()})")
bp = ax[1].boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55, showfliers=False,
                   medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for i, b in enumerate(bp["boxes"]):
    b.set_facecolor(C["echo"] if i == 3 else C["open_loop"]); b.set_edgecolor(INK2); b.set_alpha(0.85)
ax[1].set_ylabel("max object position error (cm)")
ax[1].set_title("World-model error by action type\n(a failed grasp is a large, clean surprise)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig1_prediction.png", dpi=160); plt.close(fig)

# ---- Fig 2: detector separation ---------------------------------------------------
thr = 5.0
fig, ax = plt.subplots(figsize=(6.5, 4))
bins = np.linspace(0, 30, 61)
good = disc[~((atype == 0) & ~okt)] * 100                 # everything except failed grasps
ax.hist(good, bins=bins, color=C["open_loop"], alpha=0.85, label=f"normal actions (n={len(good)})")
ax.hist(disc[fp] * 100, bins=bins, color=C["echo"], alpha=0.9, label=f"failed grasps (n={fp.sum()})")
ax.axvline(thr, color=INK, lw=1.2, ls="--"); ax.text(thr + 0.4, ax.get_ylim()[1] * 0.9, f"threshold {thr:.0f} cm", color=INK2)
ax.set_yscale("log"); ax.set_xlabel("prediction discrepancy (cm)"); ax.set_ylabel("number of actions (log)")
ax.set_title("Failure detector: prediction discrepancy", loc="left", fontsize=10); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig2_detector.png", dpi=160); plt.close(fig)


# ---- Fig 3: task success with 95% Wilson CI ------------------------------------------------
def wilson(k, nn, z=1.96):
    p = k / nn; dd = 1 + z * z / nn; c = (p + z * z / (2 * nn)) / dd
    h = z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / dd
    return c - h, c + h


SC = ["nominal", "grasp_fail", "displace", "obstacles"]
SC_LABEL = ["Nominal", "Failed grasp", "Cube knocked", "2 obstacles"]
MODES = ["open_loop", "verify_retry", "no_wm", "echo_detect_only", "echo"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
for ax, (f, title) in zip(axes, (("results/phase5_results.json", "Ground-truth object state"),
                                 ("results/phase5_percep_results.json", "Camera only (perceived state)"))):
    rows = json.load(open(f)); by = collections.defaultdict(list)
    for r in rows: by[(r["scenario"], r["mode"])].append(r["success"])
    w = 0.16
    for j, md in enumerate(MODES):
        x = np.arange(len(SC)) + (j - 2) * w
        p = np.array([np.mean(by[(s, md)]) for s in SC])
        ci = np.array([wilson(sum(by[(s, md)]), len(by[(s, md)])) for s in SC])
        ax.bar(x, p * 100, w * 0.88, color=C[md], label=LABEL[md])
        ax.errorbar(x, p * 100, yerr=[(p - ci[:, 0]) * 100, (ci[:, 1] - p) * 100], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    ax.set_xticks(np.arange(len(SC))); ax.set_xticklabels(SC_LABEL); ax.set_ylim(0, 105)
    ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("task success (%)  [bars: rate, whiskers: 95% CI]")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(OUT / "fig3_success.png", dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

"""Push tasks: "Slide the X cube next to the Y cube" under hidden, object-dependent friction.

The skill `push` takes a commanded distance, but how far an object really travels depends on its (hidden) friction:
blue ~ as commanded, red undershoots, green overshoots and slides. Two ways to choose the command:
  rule  : assume the object moves exactly as commanded (the conventional planner).
  curve : per-object response curve (commanded -> actual distance) fitted on the SAME push data, polynomial regression,
          no world model: ignores other objects and the rest of the state. The control for "is a world model needed?".
  adaptive: rule planner that re-estimates the object's response (gain = travelled / commanded) after each push --
          a no-world-model baseline with online system identification.
  wm    : search candidate commands with the learned world model and choose the one whose PREDICTED outcome best
          satisfies the goal (and does not disturb other objects) -- predictive planning (Baseline B).
Each can run open loop (one push) or with feedback retries (re-check the goal, re-plan the command).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .parser import parse
from .planner import check_goal, NEXT_TO_GAP
from .wm_data import OBJECTS, encode_state, encode_action_v2

MODES = ("rule_open", "rule_retry", "adaptive_retry", "curve_open", "curve_retry", "wm_open", "wm_retry")
CANDIDATES = np.arange(0.02, 0.36, 0.005)


def fit_response_curves(path: str = "data_push_task.npz", degree: int = 3) -> dict:
    """Per-object polynomial fit of travelled distance vs commanded distance on the world model's TRAIN episodes."""
    d = np.load(path); S, A, S2, ep = d["S"], d["A"], d["S2"], d["ep"]
    eps = np.unique(ep); r = np.random.default_rng(0); r.shuffle(eps)
    m = np.isin(ep, eps[:int(.8 * len(eps))])
    obj = A[:, 4:8].argmax(1); cmd = A[:, -1]; rows = np.arange(len(S))
    sgn = np.sign(cmd); y0 = S[rows, 4 * obj + 1]
    moved = (S2[rows, 4 * obj + 1] - y0) * sgn
    return {k: np.poly1d(np.polyfit(np.abs(cmd)[m & (obj == k)], moved[m & (obj == k)], degree)) for k in range(3)}


class PushPlanner:
    def __init__(self, wm=None, curves=None):
        self.wm, self.curves = wm, curves

    def command(self, state, obj: str, ref: str):
        """Returns (direction sign along y, commanded distance, predicted gap or None)."""
        oy, ry = state.objects[obj].position[1], state.objects[ref].position[1]
        side = 1.0 if oy >= ry else -1.0                       # stay on the side we are on; stop next to the reference
        delta = (ry + side * NEXT_TO_GAP) - oy
        sgn, need = (1.0 if delta >= 0 else -1.0), abs(delta)
        if self.curves is not None:
            oi = OBJECTS.index(obj)
            travelled = self.curves[oi](CANDIDATES)
            gap = np.abs((oy + sgn * travelled) - ry)
            b = int(np.argmin(np.abs(gap - NEXT_TO_GAP)))
            return sgn, float(CANDIDATES[b]), float(gap[b])
        if self.wm is None:
            return sgn, need, None                             # assume "moves as commanded"
        s0 = encode_state(state)
        S = np.repeat(s0[None], len(CANDIDATES), 0)
        A = np.stack([encode_action_v2({"type": "push", "object": obj, "direction": (0.0, sgn), "distance": float(c)}, state)
                      for c in CANDIDATES])
        pred, _ = self.wm.predict(S, A)
        oi, ri = OBJECTS.index(obj), OBJECTS.index(ref)
        po, pr = pred[:, 4 * oi:4 * oi + 2], pred[:, 4 * ri:4 * ri + 2]
        gap = np.linalg.norm(po - pr, axis=1)
        cur = s0.reshape(-1)[:16].reshape(4, 4)[:, :2]
        others = [k for k in range(4) if k != oi]
        collateral = np.max([np.linalg.norm(pred[:, 4 * k:4 * k + 2] - cur[k], axis=1) for k in others], axis=0)
        score = np.abs(gap - NEXT_TO_GAP) + 3.0 * np.maximum(collateral - 0.01, 0.0)
        b = int(np.argmin(score))
        return sgn, float(CANDIDATES[b]), float(gap[b])


def make_push_task(i: int):
    """Aligned two-cube layout (same x, separated along y) so that a +/-y push can solve it; other objects stay clear."""
    rng = np.random.default_rng(90000 + i)
    colors = ["red", "green", "blue"]
    a, b = rng.choice(3, 2, replace=False)
    x = float(rng.uniform(0.45, 0.62))
    while True:
        y_ref = float(rng.uniform(-0.30, 0.30)); L = float(rng.uniform(0.20, 0.30)); sg = float(rng.choice([-1, 1]))
        if abs(y_ref + sg * L) <= 0.30: break
    y_sub = y_ref + sg * L
    layout = {f"{colors[a]}_cube": (x + float(rng.uniform(-0.01, 0.01)), y_sub), f"{colors[b]}_cube": (x, y_ref)}
    lo, hi = min(y_ref, y_sub) - 0.10, max(y_ref, y_sub) + 0.10
    placed = list(layout.values())
    remaining = [f"{c}_cube" for k, c in enumerate(colors) if k not in (a, b)] + ["blue_container"]
    for name in remaining:
        for _ in range(500):
            xy = (float(rng.uniform(0.35, 0.75)), float(rng.uniform(-0.30, 0.30)))
            in_corridor = abs(xy[0] - x) < 0.15 and lo < xy[1] < hi
            if not in_corridor and all(np.hypot(xy[0] - q[0], xy[1] - q[1]) > 0.13 for q in placed):
                break
        layout[name] = xy; placed.append(xy)
    return f"Slide the {colors[a]} cube next to the {colors[b]} cube.", layout, f"{colors[a]}_cube"


@dataclass
class PushEpisode:
    success: bool = False
    pushes: int = 0
    first_push_ok: bool = False


def run_push_episode(env, instruction: str, mode: str, wm=None, max_pushes: int = 6, curves=None) -> PushEpisode:
    assert mode in MODES
    g = parse(instruction, scene_objects=list(env.objects)).goals[0]
    assert g.kind == "next_to" and g.method == "push", g
    planner = PushPlanner(wm if mode.startswith("wm") else None, curves if mode.startswith("curve") else None)
    retry = mode.endswith("retry")
    gain = 1.0                                               # adaptive baseline: estimated travelled/commanded ratio
    ep = PushEpisode()
    while ep.pushes < (max_pushes if retry else 1):
        st = env.get_state()
        if ep.pushes > 0 and check_goal(st, g, env.container_inner):
            break
        sgn, dist, _ = planner.command(st, g.subject, g.ref)
        if mode == "adaptive_retry":
            dist = min(dist / gain, 0.35)
        y0 = st.objects[g.subject].position[1]
        env.step({"type": "push", "object": g.subject, "direction": (0.0, sgn), "distance": float(dist)})
        ep.pushes += 1
        if mode == "adaptive_retry":
            moved = (env.get_state().objects[g.subject].position[1] - y0) * sgn
            if dist > 0.02 and moved > 0.01:
                gain = moved / dist
        if ep.pushes == 1:
            ep.first_push_ok = check_goal(env.get_state(), g, env.container_inner)
    ep.success = check_goal(env.get_state(), g, env.container_inner)
    return ep

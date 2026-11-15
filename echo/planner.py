"""Skill library, hierarchical planner (goal -> skills), and goal verifier."""
from __future__ import annotations
import numpy as np
from .task import Goal, Step, TaskSpec
from .env import WORKSPACE, EnvState

REL_DIR = {"left_of": (0, 1), "right_of": (0, -1), "in_front_of": (-1, 0), "behind": (1, 0)}
CLEARANCE = {"cube": 0.07, "container": 0.12, "obstacle": 0.09}


# ------------------------------------------------------------------ planning
def plan(spec: TaskSpec) -> list[Step]:
    """Decompose each goal into primitive skills. Targets that depend on object positions
    are resolved late (at execution time), since earlier steps move objects."""
    steps: list[Step] = []
    for g in spec.goals:
        if g.kind == "picked":
            steps.append(Step("pick", {"object": g.subject}, g.id))
        elif g.kind == "in":
            steps += [Step("pick", {"object": g.subject}, g.id),
                      Step("place_in", {"object": g.subject, "container": g.ref}, g.id)]
        elif g.kind in REL_DIR or g.kind == "next_to":
            if g.method == "push":
                steps.append(Step("push_rel", {"object": g.subject, "ref": g.ref, "relation": g.kind}, g.id))
            else:
                steps += [Step("pick", {"object": g.subject}, g.id),
                          Step("place_rel", {"object": g.subject, "ref": g.ref, "relation": g.kind}, g.id)]
        else:
            raise ValueError(f"no skill for goal kind {g.kind}")
    return steps


def resolve_place_target(state: EnvState, obj: str, ref: str, relation: str):
    """Pick a free table position satisfying `relation` w.r.t. `ref`'s *current* position."""
    rp = np.array(state.objects[ref].position[:2])
    if relation == "next_to":                       # ring around the reference, on the side the object is already on
        return _next_to_target(state, obj, ref, rp)
    d = np.array(REL_DIR[relation], dtype=float)
    perp = np.array([-d[1], d[0]])
    base = 0.10 if state.objects[ref].kind == "cube" else 0.16
    others = [o for n, o in state.objects.items() if n != obj]
    for scale in (1.0, 1.25, 1.5, 1.8):
        for shift in (0.0, 0.03, -0.03, 0.06, -0.06):
            xy = rp + d * base * scale + perp * shift
            # reachable placement region (larger than the object spawn region)
            if not (0.30 <= xy[0] <= 0.78 and abs(xy[1]) <= 0.36 and np.hypot(*xy) <= 0.82):
                continue
            if all(np.linalg.norm(xy - np.array(o.position[:2])) > CLEARANCE[o.kind] for o in others):
                return tuple(xy)
    return None


NEXT_TO_GAP, NEXT_TO_RANGE = 0.085, (0.05, 0.12)     # target centre distance and accepted range (m)


def _next_to_target(state: EnvState, obj: str, ref: str, rp: np.ndarray):
    away = np.array(state.objects[obj].position[:2]) - rp
    base = np.arctan2(away[1], away[0])
    others = [o for n, o in state.objects.items() if n not in (obj, ref)]
    for off in (0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 3.14159):
        a = base + off
        xy = rp + NEXT_TO_GAP * np.array([np.cos(a), np.sin(a)])
        if not (0.30 <= xy[0] <= 0.78 and abs(xy[1]) <= 0.36 and np.hypot(*xy) <= 0.82):
            continue
        if all(np.linalg.norm(xy - np.array(o.position[:2])) > CLEARANCE[o.kind] for o in others):
            return tuple(xy)
    return None


# -------------------------------------------------------------- verification
def check_goal(state: EnvState, g: Goal, container_inner: float = 0.10) -> bool:
    o = state.objects[g.subject]
    p = np.array(o.position)
    if g.kind == "picked":
        return o.held or p[2] > 0.08
    r = np.array(state.objects[g.ref].position)
    if g.kind == "in":
        half = container_inner / 2
        return bool(abs(p[0] - r[0]) < half and abs(p[1] - r[1]) < half and p[2] < 0.09)
    if g.kind == "next_to":
        dist = float(np.linalg.norm((p - r)[:2]))
        return NEXT_TO_RANGE[0] <= dist <= NEXT_TO_RANGE[1] and p[2] < 0.05 and not o.held
    d = np.array(REL_DIR[g.kind])
    diff = (p - r)[:2]
    along, across = float(diff @ d), abs(float(diff @ np.array([-d[1], d[0]])))
    return along > 0.05 and across < 0.07 and p[2] < 0.05 and not o.held

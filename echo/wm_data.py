"""State/action encoding for the world model (fixed scene: 3 cubes + 1 container)."""
from __future__ import annotations
import numpy as np
from .env import EnvState

OBJECTS = ["red_cube", "green_cube", "blue_cube", "blue_container"]
ACTION_TYPES = ["pick", "place", "place_in"]
STATE_DIM = len(OBJECTS) * 4 + 3 + 1          # per object [x,y,z,held] + ee xyz + gripper opening
ACTION_DIM = len(ACTION_TYPES) + len(OBJECTS) + 2   # type one-hot, object one-hot, target xy
ACTION_TYPES_V2 = ACTION_TYPES + ["push"]            # v2 adds the push primitive (used by the push world model)
ACTION_DIM_V2 = len(ACTION_TYPES_V2) + len(OBJECTS) + 2


def encode_state(s: EnvState) -> np.ndarray:
    v = []
    for n in OBJECTS:
        o = s.objects[n]
        v += [*o.position, float(o.held)]
    v += [*s.ee_pos, s.gripper_open]
    return np.asarray(v, dtype=np.float32)


def encode_action(a: dict, s: EnvState) -> np.ndarray:
    """pick: object one-hot. place: target xy. place_in: container one-hot (target xy = container xy)."""
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    v[ACTION_TYPES.index(a["type"])] = 1
    off = len(ACTION_TYPES)
    if a["type"] == "pick":
        v[off + OBJECTS.index(a["object"])] = 1
    elif a["type"] == "place":
        v[off + len(OBJECTS):] = a["target"]
    elif a["type"] == "place_in":
        c = a["container"]
        v[off + OBJECTS.index(c)] = 1
        v[off + len(OBJECTS):] = s.objects[c].position[:2]
    return v


def encode_action_v2(a: dict, s: EnvState) -> np.ndarray:
    """Like encode_action but with a push type. push: object one-hot + signed displacement command (dx, dy)."""
    if a["type"] != "push":
        v1 = encode_action(a, s)
        n1 = len(ACTION_TYPES)
        return np.concatenate([v1[:n1], [0.0], v1[n1:]]).astype(np.float32)
    v = np.zeros(ACTION_DIM_V2, dtype=np.float32)
    v[ACTION_TYPES_V2.index("push")] = 1
    v[len(ACTION_TYPES_V2) + OBJECTS.index(a["object"])] = 1
    d = np.asarray(a["direction"], dtype=float); d = d / np.linalg.norm(d)
    v[-2:] = d * a["distance"]
    return v

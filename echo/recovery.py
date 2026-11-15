"""Phase 4: execution monitoring, failure detection and goal-level replanning/recovery.

Modes
  open_loop        Baseline A: execute the plan once, no monitoring.
  verify_retry     NO step monitoring: execute the plan, then re-plan only for goals that fail final verification.
  no_wm            monitoring WITHOUT a world model: rule checks (skill return flag + goal post-conditions) + recovery.
  echo_detect_only world-model discrepancy monitoring, but stop at the first detection (no recovery).
  echo             world-model discrepancy monitoring + replanning/recovery (full system).

Recovery = goal-level replanning: after a detection, re-check every goal against the true state and
re-plan skills for the goals that are not satisfied (this redoes failed grasps and repairs displaced objects).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .env import ManipulationEnv
from .parser import parse
from .planner import plan, resolve_place_target, check_goal
from .task import Step
from .wm_data import encode_state, encode_action, OBJECTS

MODES = ("open_loop", "verify_retry", "no_wm", "echo_detect_only", "echo")
POS_IDX = [4 * i + j for i in range(len(OBJECTS)) for j in range(3)]


@dataclass
class Injection:
    """Controlled failure scenario. Indices count executed actions / picks across the whole episode."""
    grasp_fail_pick: int | None = None          # the n-th pick attempt fails to close the gripper
    disturb_after: int | None = None            # after the n-th action, knock a cube
    disturb_obj: str = "red_cube"
    disturb_dxy: tuple = (0.0, 0.10)


@dataclass
class Episode:
    success: bool = False
    n_actions: int = 0
    detections: int = 0
    replans: int = 0
    injected: bool = False                      # was a failure actually applied?
    infeasible: bool = False                    # planner found no feasible target (blocked instruction)
    step_flags: list = field(default_factory=list)   # (flagged, truly_bad) per step, for detector metrics
    max_discrepancy: float = 0.0


def step_to_action(env: ManipulationEnv, step: Step, state=None):
    """Skill -> primitive action. With `state` (an observation), targets come from that observation,
    so perception error propagates into execution; without it the simulator's true state is used."""
    a = step.args
    if step.skill == "pick":
        act = {"type": "pick", "object": a["object"]}
        if state is not None:
            act["position"] = tuple(state.objects[a["object"]].position[:2])
        return act
    if step.skill == "place_in":
        act = {"type": "place_in", "container": a["container"]}
        if state is not None:
            act["position"] = tuple(state.objects[a["container"]].position[:2])
        return act
    xy = resolve_place_target(state or env.get_state(), a["object"], a["ref"], a["relation"])
    return None if xy is None else {"type": "place", "target": tuple(map(float, xy))}


def discrepancy(pred: np.ndarray, actual: np.ndarray) -> float:
    """e_t = max over objects of the position error between predicted and actual next state (metres)."""
    p = pred[POS_IDX].reshape(-1, 3); t = actual[POS_IDX].reshape(-1, 3)
    return float(np.linalg.norm(p - t, axis=1).max())


def run_episode(env: ManipulationEnv, instruction: str, mode: str = "echo", wm=None, thr: float = 0.102,
                inject: Injection | None = None, max_actions: int = 24, max_passes: int = 6,
                perception=None) -> Episode:
    """If `perception` is given, all decisions (planning targets, world-model input, monitoring, goal checks)
    use camera observations; success is still scored on the true simulator state."""
    assert mode in MODES
    inject = inject or Injection()
    use_wm = mode.startswith("echo")
    recover = mode in ("verify_retry", "no_wm", "echo")
    spec = parse(instruction, scene_objects=list(env.objects))
    ep = Episode()
    picks = 0
    passes = 0

    def obs():
        return perception.observe(env) if perception is not None else env.get_state()

    def unsatisfied(true: bool = False):
        s = env.get_state() if true else obs()
        return [g for g in spec.goals if not check_goal(s, g, env.container_inner)]

    while True:
        todo = spec.goals if passes == 0 else unsatisfied()
        if not todo or passes >= (1 if mode == "open_loop" else max_passes) or ep.n_actions >= max_actions:
            break
        if passes > 0:
            ep.replans += 1
        passes += 1
        aborted = False
        for g in todo:
            steps = plan(type(spec)(spec.instruction, [g]))
            for st in steps:
                # skill precondition: do not pick an object that is already in the gripper
                s0 = obs()
                if st.skill == "pick" and s0.objects[st.args["object"]].held:
                    continue
                act = step_to_action(env, st, s0 if perception is not None else None)
                if act is None:                         # planner found no feasible target
                    ep.infeasible = True; aborted = True; break
                pred = None
                if use_wm:
                    pred, _ = wm.predict(encode_state(s0)[None], encode_action(act, s0)[None])
                    pred = pred[0]
                if st.skill == "pick":                  # failure injection: slipped grasp
                    if inject.grasp_fail_pick == picks:
                        env.force_grasp_fail = True; ep.injected = True
                    picks += 1
                s1, info = env.step(act)
                ep.n_actions += 1
                disturbed = False
                if inject.disturb_after == ep.n_actions and not s1.objects[inject.disturb_obj].held:
                    env.disturb(inject.disturb_obj, inject.disturb_dxy)
                    disturbed = True; ep.injected = True
                s1 = obs()                              # observation after the action (and any disturbance)
                truly_bad = (not info["success"]) or disturbed

                if mode in ("open_loop", "verify_retry"):
                    flagged = False
                elif use_wm:
                    e = discrepancy(pred, encode_state(s1)); ep.max_discrepancy = max(ep.max_discrepancy, e)
                    flagged = e > thr
                else:                                   # rule-based: only the skill's own return flag
                    flagged = not info["success"]
                ep.step_flags.append((flagged, truly_bad))
                if flagged:
                    ep.detections += 1
                    aborted = True
                    break
                if ep.n_actions >= max_actions:
                    aborted = True; break
            if aborted:
                break
            if recover and mode != "verify_retry" and not check_goal(obs(), g, env.container_inner):
                ep.detections += 1                      # goal post-condition check after finishing a goal
                aborted = True
                break
        if ep.infeasible or (aborted and mode == "echo_detect_only"):
            break
        if aborted and not recover and mode != "open_loop":
            break
    env.park()
    ep.success = not unsatisfied(true=True)      # scored on ground truth, never on perception
    return ep

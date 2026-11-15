"""Language-conditioned execution pipeline: parse -> plan -> execute -> verify.

This is Baseline A (open loop, no world model / monitoring): it executes the plan once
and verifies the goals afterwards.  Phase 4 adds monitoring and recovery on top.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .env import ManipulationEnv
from .parser import parse
from .planner import plan, resolve_place_target, check_goal
from .task import TaskSpec, Step


@dataclass
class RunResult:
    spec: TaskSpec
    steps: list[Step]
    step_ok: list[bool] = field(default_factory=list)
    goal_ok: dict = field(default_factory=dict)
    success: bool = False
    n_actions: int = 0

    def report(self):
        lines = [f"instruction: {self.spec.instruction}", "goals:", str(self.spec), "plan:"]
        lines += [f"  {i}. {s}  -> {'ok' if ok else 'FAILED'}"
                  for i, (s, ok) in enumerate(zip(self.steps, self.step_ok))]
        lines.append("goal verification: " + ", ".join(f"[{k}]={'ok' if v else 'FAIL'}" for k, v in self.goal_ok.items()))
        lines.append(f"TASK SUCCESS: {self.success}   (actions executed: {self.n_actions})")
        return "\n".join(lines)


def run_step(env: ManipulationEnv, step: Step) -> bool:
    a = step.args
    if step.skill == "pick":
        _, info = env.step({"type": "pick", "object": a["object"]})
    elif step.skill == "place_in":
        _, info = env.step({"type": "place_in", "container": a["container"]})
    elif step.skill == "place_rel":
        xy = resolve_place_target(env.get_state(), a["object"], a["ref"], a["relation"])
        if xy is None:
            return False
        _, info = env.step({"type": "place", "target": xy})
    else:
        raise ValueError(step.skill)
    return info["success"]


def execute_instruction(env: ManipulationEnv, instruction: str, stop_on_failure: bool = True) -> RunResult:
    spec = parse(instruction, scene_objects=list(env.objects))
    steps = plan(spec)
    res = RunResult(spec, steps)
    for step in steps:
        ok = run_step(env, step)
        res.step_ok.append(ok); res.n_actions += 1
        if not ok and stop_on_failure:
            break
    env.park()
    state = env.get_state()
    res.goal_ok = {g.id: check_goal(state, g, env.container_inner) for g in spec.goals}
    res.success = all(res.goal_ok.values()) and len(res.step_ok) == len(steps)
    return res

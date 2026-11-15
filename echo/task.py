"""Structured task representation (Phase 2).

Frame convention: robot base at origin facing +x.  left = +y, right = -y,
in_front_of = toward the robot (-x), behind = away from the robot (+x).
"""
from __future__ import annotations
from dataclasses import dataclass, field

RELATIONS = {"in", "left_of", "right_of", "in_front_of", "behind", "picked"}


@dataclass
class Goal:
    """A desired end condition: kind(subject, ref)."""
    id: int
    kind: str                     # one of RELATIONS
    subject: str                  # e.g. "red_cube"
    ref: str | None = None        # e.g. "blue_container"
    depends_on: list[int] = field(default_factory=list)

    def __str__(self):
        return f"[{self.id}] {self.kind}({self.subject}{', ' + self.ref if self.ref else ''})" + \
               (f" after {self.depends_on}" if self.depends_on else "")


@dataclass
class TaskSpec:
    instruction: str
    goals: list[Goal]

    def __str__(self):
        return "\n".join(str(g) for g in self.goals)


@dataclass
class Step:
    """One executable skill invocation produced by the planner."""
    skill: str                    # pick | place_in | place_rel
    args: dict
    goal_id: int

    def __str__(self):
        return f"{self.skill}({', '.join(f'{k}={v}' for k, v in self.args.items())})  <- goal {self.goal_id}"

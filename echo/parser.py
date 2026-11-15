"""Rule-based natural-language instruction parser (no APIs, deterministic)."""
from __future__ import annotations
import re
from .task import Goal, TaskSpec

COLOR = r"(red|green|blue|yellow)"
KIND = r"(cube|block|container|box|bin|bowl|obstacle)"
NP = re.compile(rf"{COLOR}\s+{KIND}")
KIND_NORM = {"cube": "cube", "block": "cube", "container": "container", "box": "container",
             "bin": "container", "bowl": "container", "obstacle": "obstacle"}
RELS = [  # order matters: longest phrases first
    (r"in front of", "in_front_of"), (r"to the left of|left of|on the left of", "left_of"),
    (r"to the right of|right of|on the right of", "right_of"), (r"behind", "behind"),
    (r"next to|beside|adjacent to", "next_to"),
]
CONTAINER_WORDS = r"\b(in|into|inside|within)\b"
PICK_WORDS = r"\b(pick up|pick|grab|take|lift)\b"
PLACE_WORDS = r"\b(place|put|move|set|drop|push|shift)\b"


class ParseError(ValueError):
    pass


def _nps(clause):
    return [f"{c}_{KIND_NORM[k]}" for c, k in NP.findall(clause)]


def parse(instruction: str, scene_objects: list[str] | None = None) -> TaskSpec:
    text = re.sub(r"[.!?]", " ", instruction.lower())
    clauses = [c.strip() for c in re.split(r",|;|\bthen\b|\band\b", text) if c.strip()]
    goals: list[Goal] = []
    current: str | None = None       # most recently mentioned subject, for "it"

    for clause in clauses:
        nps = _nps(clause)
        pronoun = bool(re.search(r"\b(it|them|that)\b", clause))
        if pronoun and current:
            subject, refs = current, nps
        elif nps:
            subject, refs = nps[0], nps[1:]
        else:
            raise ParseError(f"cannot find an object in clause: '{clause}'")

        rel = next((name for pat, name in RELS if re.search(pat, clause)), None)
        if rel:
            if not refs:
                raise ParseError(f"relation without a reference object: '{clause}'")
            method = "push" if re.search(r"\b(push|slide|nudge)\b", clause) else "pick_place"
            goals.append(Goal(len(goals), rel, subject, refs[0], method=method))
        elif re.search(CONTAINER_WORDS, clause) and refs:
            if not refs[0].endswith("container"):
                raise ParseError(f"'{refs[0]}' is not a container: '{clause}'")
            goals.append(Goal(len(goals), "in", subject, refs[0]))
        elif re.search(PICK_WORDS, clause):
            goals.append(Goal(len(goals), "picked", subject))
        else:
            raise ParseError(f"unrecognised action in clause: '{clause}'")
        current = subject

    # a bare "pick up X" immediately followed by a goal on X is redundant (the plan includes the pick)
    kept: list[Goal] = []
    for i, g in enumerate(goals):
        if g.kind == "picked" and i + 1 < len(goals) and goals[i + 1].subject == g.subject:
            continue
        kept.append(g)
    goals = kept
    for i, g in enumerate(goals):       # renumber, then compute dependencies
        g.id = i
    for g in goals:
        g.depends_on = [h.id for h in goals[:g.id]
                        if {h.subject, h.ref} & {g.subject, g.ref} - {None}]

    if scene_objects is not None:
        for g in goals:
            for o in (g.subject, g.ref):
                if o and o not in scene_objects:
                    raise ParseError(f"object '{o}' is not in the scene {sorted(scene_objects)}")
    return TaskSpec(instruction, goals)

"""Run Baseline A over many random layouts; report success rate and failure reasons."""
import sys, pathlib, json, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo import ManipulationEnv
from echo.executor import execute_instruction

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TASKS = [
    "Pick up the red cube, place it inside the blue container, and move the green cube to the left of the red cube.",
    "Put the green cube in the blue container",
    "Move the red cube to the right of the green cube",
]
env = ManipulationEnv(seed=123)
rows, reasons = [], collections.Counter()
for i in range(N):
    task = TASKS[i % len(TASKS)]
    env.reset()                       # random layout from env rng
    res = execute_instruction(env, task)
    if not res.success:
        bad = [str(s) for s, ok in zip(res.steps, res.step_ok) if not ok]
        why = bad[0].split("(")[0] if bad else "goal_not_satisfied:" + ",".join(
            f"{res.spec.goals[k].kind}" for k, v in res.goal_ok.items() if not v)
        reasons[why] += 1
    rows.append({"i": i, "task": i % len(TASKS), "success": res.success, "actions": res.n_actions})
    if (i + 1) % 10 == 0:
        print(f"{i+1}/{N} success so far: {np.mean([r['success'] for r in rows]):.2f}", flush=True)

print("\nOverall success:", np.mean([r["success"] for r in rows]))
for t in range(len(TASKS)):
    sub = [r["success"] for r in rows if r["task"] == t]
    print(f"  task {t}: {np.mean(sub):.2f} ({len(sub)} runs)  {TASKS[t][:60]}")
print("Failure reasons:", dict(reasons))
json.dump(rows, open("batch_eval_baselineA.json", "w"))
env.close()

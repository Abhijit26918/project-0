"""Record annotated demo videos of the full system (world-model monitoring + recovery) -> results/videos/."""
import sys, pathlib, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import cv2
import pybullet as p
from PIL import Image
from echo import ManipulationEnv
from echo.worldmodel import WorldModel
import echo.recovery as R
from eval_phase4 import TASKS, make_injection

OUT = pathlib.Path("results/videos"); OUT.mkdir(parents=True, exist_ok=True)
W, H, EVERY, FPS, THR = 480, 360, 8, 30, 0.05
VIEW = p.computeViewMatrix([0.95, -0.85, 0.75], [0.5, 0.0, 0.08], [0, 0, 1])
PROJ = p.computeProjectionMatrixFOV(55, W / H, 0.05, 3.0)
wm = WorldModel.load("checkpoints/wm_nominal.pt")


def pick_layout(scenario, task_idx, want_detection):
    """First held-out layout where the full system succeeds (and, if asked, a failure was detected)."""
    env = ManipulationEnv(seed=0)
    for i in range(200 + task_idx, 400, 3):
        env.rng = np.random.default_rng(1000 + i); env.reset()
        ep = R.run_episode(env, TASKS[task_idx], "echo", wm, THR, make_injection(scenario, i))
        if ep.success and (ep.detections > 0 or not want_detection) and (not want_detection or ep.injected):
            env.close(); return i
    env.close(); raise RuntimeError("no suitable layout")


def record(name, scenario, task_idx, want_detection):
    i = pick_layout(scenario, task_idx, want_detection)
    env = ManipulationEnv(seed=0)
    env.rng = np.random.default_rng(1000 + i); env.reset()
    frames, state = [], {"line": "start", "alert": "", "count": 0}
    instruction = TASKS[task_idx]

    def grab():
        _, _, rgb, _, _ = p.getCameraImage(W, H, VIEW, PROJ, renderer=p.ER_TINY_RENDERER)
        img = np.reshape(rgb, (H, W, 4))[..., :3].astype(np.uint8)[..., ::-1].copy()      # RGB -> BGR for cv2
        cv2.rectangle(img, (0, 0), (W, 62), (30, 30, 30), -1)
        for k, ln in enumerate(textwrap.wrap(instruction, 62)[:2]):
            cv2.putText(img, ln, (8, 18 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, state["line"], (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 220, 255), 1, cv2.LINE_AA)
        if state["alert"]:
            col = (60, 60, 230) if "FAILURE" in state["alert"] else (80, 200, 80)
            cv2.rectangle(img, (0, H - 34), (W, H), (30, 30, 30), -1)
            cv2.putText(img, state["alert"], (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
        return img

    orig_step = env._step
    def rec_step(n=1):
        for _ in range(n):
            p.stepSimulation(); state["count"] += 1
            if state["count"] % EVERY == 0: frames.append(grab())
    env._step = rec_step

    orig_env_step = env.step
    def step_wrapped(a):
        tgt = a.get("object") or a.get("container") or (tuple(round(x, 2) for x in a["target"]) if "target" in a else "")
        state["line"] = f"action: {a['type']} {tgt}"
        return orig_env_step(a)
    env.step = step_wrapped

    orig_disc = R.discrepancy
    def disc_wrapped(pred, actual):
        e = orig_disc(pred, actual)
        if e > THR:
            state["alert"] = f"FAILURE DETECTED  (prediction error {e*100:.0f} cm) - replanning"
            frames.extend([grab()] * 45)                       # hold the frame ~1.5 s
        else:
            state["alert"] = f"as predicted (error {e*100:.1f} cm)"
        return e
    R.discrepancy = disc_wrapped

    frames.extend([grab()] * 20)
    ep = R.run_episode(env, instruction, "echo", wm, THR, make_injection(scenario, i))
    state["alert"] = "TASK COMPLETE" if ep.success else "task not completed"
    frames.extend([grab()] * 45)
    R.discrepancy = orig_disc; env.close()

    mp4 = OUT / f"{name}.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in frames: vw.write(f)
    vw.release()
    small = [Image.fromarray(cv2.resize(f, (320, 240))[..., ::-1]) for f in frames[::3]]
    small[0].save(OUT / f"{name}.gif", save_all=True, append_images=small[1:], duration=100, loop=0, optimize=True)
    print(f"{name}: layout {i}, {len(frames)} frames ({len(frames)/FPS:.0f}s), success={ep.success}, "
          f"detections={ep.detections}, mp4 {mp4.stat().st_size/1e6:.1f} MB", flush=True)


record("01_language_task", "nominal", 0, want_detection=False)
record("02_failed_grasp_recovery", "grasp_fail", 0, want_detection=True)
record("03_knocked_cube_recovery", "displace", 0, want_detection=True)

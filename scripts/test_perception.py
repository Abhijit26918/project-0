"""Measure perception accuracy against simulator ground truth on random layouts."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from echo import ManipulationEnv
from echo.perception import Perception

env = ManipulationEnv(seed=7)
per = Perception(env.img_size)
errs = {k: [] for k in ("red_cube", "green_cube", "blue_cube", "blue_container")}
missed = 0
for i in range(60):
    env.reset(obstacles=2 if i % 4 == 3 else 0)
    env.park()
    rgb, _, _ = env.render(top_down=True)
    found = per.detect(rgb)
    true = env.get_state()
    for k in errs:
        if k not in found:
            missed += 1; continue
        t = np.array(true.objects[k].position[:2])
        errs[k].append(np.linalg.norm(np.array(found[k]) - t))
for k, v in errs.items():
    v = np.array(v) * 100
    print(f"{k:15s} n={len(v):2d} mean {v.mean():.2f} cm  p95 {np.percentile(v,95):.2f} cm  max {v.max():.2f} cm")
print("missed detections:", missed, "of", 60 * 4)

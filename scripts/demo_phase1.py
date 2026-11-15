"""Phase 1 demo: pick/place, camera views, injected failures. Run from project root."""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from echo import ManipulationEnv


def inside(s, cube, cont):
    r, c = s.objects[cube].position, s.objects[cont].position
    return abs(r[0] - c[0]) < 0.05 and abs(r[1] - c[1]) < 0.05 and r[2] < 0.08, \
        float(np.hypot(r[0] - c[0], r[1] - c[1]))


def save(env, name, **kw):
    env.park()
    rgb, _, _ = env.render(**kw)
    cv2.imwrite(name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


env = ManipulationEnv(gui="--gui" in sys.argv, seed=0)
save(env, "phase1_scene.png"); save(env, "phase1_scene_top.png", top_down=True)

# 1) normal pick + place_in
_, i1 = env.step({"type": "pick", "object": "red_cube"})
s, i2 = env.step({"type": "place_in", "container": "blue_container"})
ok, off = inside(s, "red_cube", "blue_container")
print(f"normal: pick={i1['success']} place={i2['success']} inside={ok} centre_offset={off*100:.1f}cm")
save(env, "phase1_after.png"); save(env, "phase1_after_top.png", top_down=True)

# 2) injected failed grasp
env.force_grasp_fail = True
s, i = env.step({"type": "pick", "object": "green_cube"})
print("failed grasp reported success:", i["success"], "(expected False)")

# 3) unexpected displacement
before = env.get_state().objects["green_cube"].position
env.disturb("green_cube", dxy=(0.08, 0.0))
after = env.get_state().objects["green_cube"].position
print(f"disturbance moved green cube {np.hypot(after[0]-before[0], after[1]-before[1])*100:.1f}cm")

json.dump([{"a": e["a"], "info": e["info"], "s": e["s"].to_dict(), "s_next": e["s_next"].to_dict()} for e in env.log],
          open("phase1_log.json", "w"), indent=1, default=float)
env.close()

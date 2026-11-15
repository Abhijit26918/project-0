"""Phase 2/5: visual perception. Top-down RGB image -> object positions (no simulator ground truth).

Classical pipeline: HSV colour segmentation -> connected components -> blob centre ->
back-projection of the pixel ray onto the object's known top-face plane.
The blue cube and blue container share a colour; the larger blob is the container.
"""
from __future__ import annotations
import cv2
import numpy as np
import pybullet as p
from .env import EnvState, ObjectState

CAM_EYE, CAM_TARGET, CAM_UP = [0.55, 0.0, 0.9], [0.55, 0.0, 0.0], [1, 0, 0]
TOP_Z = {"cube": 0.04, "container": 0.05}          # plane height of the visible top face
CUBE_Z, CONT_Z = 0.02, 0.005                        # object centre heights assumed by the state encoding
MIN_AREA = 25


class Perception:
    def __init__(self, img_size: int = 256):
        self.size = img_size
        view = np.array(p.computeViewMatrix(CAM_EYE, CAM_TARGET, CAM_UP)).reshape(4, 4).T
        proj = np.array(p.computeProjectionMatrixFOV(60, 1.0, 0.05, 2.0)).reshape(4, 4).T
        self.inv = np.linalg.inv(proj @ view)
        self.last: dict[str, tuple] = {}            # last known positions (fallback under occlusion)

    # ------------------------------------------------------------ geometry
    def pixel_to_world(self, u: float, v: float, plane_z: float):
        x = 2 * (u + 0.5) / self.size - 1; y = 1 - 2 * (v + 0.5) / self.size
        pts = []
        for z in (-1.0, 1.0):
            q = self.inv @ np.array([x, y, z, 1.0]); pts.append(q[:3] / q[3])
        a, b = pts
        t = (plane_z - a[2]) / (b[2] - a[2])
        w = a + t * (b - a)
        return float(w[0]), float(w[1])

    # ------------------------------------------------------- segmentation
    @staticmethod
    def _masks(rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat = (s > 140) & (v > 90)                  # saturated: excludes grey table/floor
        return {"red": sat & ((h < 8) | (h > 172)), "green": sat & (h > 45) & (h < 85),
                "blue": sat & (h > 105) & (h < 135)}

    @staticmethod
    def _blobs(mask):
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        out = [(stats[i, cv2.CC_STAT_AREA], cent[i], stats[i]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
        return sorted(out, key=lambda b: -b[0])

    def detect(self, rgb: np.ndarray) -> dict[str, tuple]:
        """Returns {object_name: (x, y)} for the objects found in the image."""
        m = self._masks(rgb); found = {}
        for color in ("red", "green"):
            b = self._blobs(m[color])
            if b:
                c = b[0][1]; found[f"{color}_cube"] = self.pixel_to_world(c[0], c[1], TOP_Z["cube"])
        blue = self._blobs(m["blue"])
        if blue:
            area, _, st = blue[0]                    # largest blue blob = container; use bbox centre (robust to cube inside)
            cx, cy = st[cv2.CC_STAT_LEFT] + st[cv2.CC_STAT_WIDTH] / 2 - 0.5, st[cv2.CC_STAT_TOP] + st[cv2.CC_STAT_HEIGHT] / 2 - 0.5
            found["blue_container"] = self.pixel_to_world(cx, cy, TOP_Z["container"])
            if len(blue) > 1:
                c = blue[1][1]; found["blue_cube"] = self.pixel_to_world(c[0], c[1], TOP_Z["cube"])
        return found

    # --------------------------------------------------------------- state
    def observe(self, env) -> EnvState:
        """Perceived state. Arm pose and gripper come from proprioception; held objects are located by
        the end effector (they are occluded / lifted); everything else comes from the camera."""
        true = env.get_state()
        if env._held is None:
            env.park()                              # keep the arm out of the camera's view
        rgb, _, _ = env.render(top_down=True)
        found = self.detect(rgb)
        objs = {}
        for name, o in true.objects.items():
            if o.kind == "obstacle":
                objs[name] = o                      # obstacles are not perceived (not in the world-model state)
                continue
            if o.held:
                pos = (true.ee_pos[0], true.ee_pos[1], true.ee_pos[2] - 0.0)
            elif name in found:
                z = CUBE_Z if o.kind == "cube" else CONT_Z
                pos = (*found[name], z)
            else:
                pos = self.last.get(name, o.position)   # occluded: fall back to last estimate
            self.last[name] = pos
            objs[name] = ObjectState(name, o.kind, tuple(round(float(c), 5) for c in pos), o.orientation, o.held)
        return EnvState(objs, true.ee_pos, true.gripper_open)

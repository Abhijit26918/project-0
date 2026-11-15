"""Phase 1: PyBullet tabletop manipulation environment for ECHO.

Scene: table, Franka Panda arm, coloured cubes, open-top containers, obstacles.
Observation: structured state (object poses) + optional RGB/depth camera.
Actions: high-level primitives executed through physics (IK + position control + gripper).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pybullet as p
import pybullet_data

PUSH_FRICTION = {"red": 1.5, "green": 0.4, "blue": 0.9, "yellow": 0.8}   # per-colour cube friction (slippery mode)
COLORS = {
    "red": (0.9, 0.1, 0.1, 1),
    "green": (0.1, 0.8, 0.2, 1),
    "blue": (0.1, 0.2, 0.9, 1),
    "yellow": (0.95, 0.85, 0.1, 1),
}
CUBE_SIZE = 0.04            # edge length (m)
TABLE_Z = 0.0               # table top height
WORKSPACE = ((0.35, 0.75), (-0.30, 0.30))  # x range, y range
HOME_POS = (0.45, 0.0, 0.35)
GRASP_ORN = p.getQuaternionFromEuler([np.pi, 0, 0])  # gripper pointing down


@dataclass
class ObjectState:
    name: str
    kind: str                 # cube | container | obstacle
    position: tuple
    orientation: tuple        # quaternion xyzw
    held: bool = False


@dataclass
class EnvState:
    objects: dict = field(default_factory=dict)   # name -> ObjectState
    ee_pos: tuple = (0, 0, 0)
    gripper_open: float = 1.0

    def to_dict(self):
        return asdict(self)

    def vector(self, names: list[str]) -> np.ndarray:
        """Flat feature vector for the world model: [pos(3), quat(4)] per object + ee + gripper."""
        v = []
        for n in names:
            o = self.objects[n]
            v += list(o.position) + list(o.orientation)
        v += list(self.ee_pos) + [self.gripper_open]
        return np.asarray(v, dtype=np.float32)


class ManipulationEnv:
    def __init__(self, gui: bool = False, seed: int | None = None, img_size: int = 256):
        self.gui = gui
        self.rng = np.random.default_rng(seed)
        self.img_size = img_size
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.objects: dict[str, int] = {}
        self.kinds: dict[str, str] = {}
        self.log: list[dict] = []
        self.robot = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, cubes=("red", "green", "blue"), containers=("blue",), obstacles=0,
              layout: dict | None = None, slippery: bool = False):
        """Rebuild the scene. `layout` optionally maps object name -> (x, y).
        slippery=True gives each cube colour-dependent friction (hidden from the planner) for pushing experiments."""
        self.slippery = slippery
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.loadURDF("plane.urdf")
        self.table = p.loadURDF("table/table.urdf", basePosition=[0.55, 0, -0.63], baseOrientation=[0, 0, 0, 1])
        if slippery:
            p.changeDynamics(self.table, -1, lateralFriction=0.8)
        self.robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        self.ee_link, self.finger_joints = 11, (9, 10)
        self.arm_joints = list(range(7))
        home = [0, -0.4, 0, -2.2, 0, 2.0, 0.78]
        for j, q in zip(self.arm_joints, home):
            p.resetJointState(self.robot, j, q)
        self.objects, self.kinds, self.log = {}, {}, []
        self._held, self._grasp_offset = None, (0.0, 0.0)
        self.force_grasp_fail = False   # failure injection: next pick closes nothing

        placed: list[tuple] = []
        def sample(name):
            if layout and name in layout:
                return layout[name]
            for _ in range(200):
                xy = (self.rng.uniform(*WORKSPACE[0]), self.rng.uniform(*WORKSPACE[1]))
                if all(np.hypot(xy[0] - q[0], xy[1] - q[1]) > 0.11 for q in placed):
                    return xy
            return xy

        for c in containers:
            name = f"{c}_container"
            xy = sample(name); placed.append(xy)
            self._add_container(name, xy, COLORS[c])
        for c in cubes:
            xy = sample(f"{c}_cube"); placed.append(xy)
            self._add_cube(f"{c}_cube", xy, COLORS[c])
            if slippery:
                p.changeDynamics(self.objects[f"{c}_cube"], -1, lateralFriction=PUSH_FRICTION[c])
        for i in range(obstacles):
            name = f"obstacle_{i}"
            xy = sample(name); placed.append(xy)
            self._add_obstacle(name, xy)
        self._open_gripper()
        self.move_to(HOME_POS)
        self._settle(60)
        return self.get_state()

    def _add_cube(self, name, xy, rgba):
        h = CUBE_SIZE / 2
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[h] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[h] * 3, rgbaColor=rgba)
        b = p.createMultiBody(0.05, col, vis, [xy[0], xy[1], TABLE_Z + h + 0.001])
        p.changeDynamics(b, -1, lateralFriction=1.2)
        self.objects[name], self.kinds[name] = b, "cube"

    def _add_container(self, name, xy, rgba, inner=0.10, wall=0.01, height=0.05):
        """Open-top box made from 5 boxes, as one compound body."""
        half = inner / 2 + wall
        shapes = [  # (half extents, offset)
            ([half, half, wall / 2], [0, 0, wall / 2]),
            ([wall / 2, half, height / 2], [half - wall / 2, 0, height / 2]),
            ([wall / 2, half, height / 2], [-half + wall / 2, 0, height / 2]),
            ([half, wall / 2, height / 2], [0, half - wall / 2, height / 2]),
            ([half, wall / 2, height / 2], [0, -half + wall / 2, height / 2]),
        ]
        cols = [p.createCollisionShape(p.GEOM_BOX, halfExtents=he) for he, _ in shapes]
        viss = [p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=rgba) for he, _ in shapes]
        b = p.createMultiBody(
            baseMass=0.5, baseCollisionShapeIndex=cols[0], baseVisualShapeIndex=viss[0],
            basePosition=[xy[0], xy[1], TABLE_Z + 0.001],
            linkMasses=[0.1] * 4, linkCollisionShapeIndices=cols[1:], linkVisualShapeIndices=viss[1:],
            linkPositions=[o for _, o in shapes[1:]], linkOrientations=[[0, 0, 0, 1]] * 4,
            linkInertialFramePositions=[[0, 0, 0]] * 4, linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0] * 4, linkJointTypes=[p.JOINT_FIXED] * 4, linkJointAxis=[[0, 0, 1]] * 4)
        self.objects[name], self.kinds[name] = b, "container"
        self.container_inner = inner

    def _add_obstacle(self, name, xy):
        he = [0.03, 0.03, 0.05]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=(0.4, 0.4, 0.4, 1))
        b = p.createMultiBody(0, col, vis, [xy[0], xy[1], TABLE_Z + he[2]])
        self.objects[name], self.kinds[name] = b, "obstacle"

    # ------------------------------------------------------------- simulation
    def _step(self, n=1):
        for _ in range(n):
            p.stepSimulation()
            if self.gui:
                time.sleep(1 / 240)

    def _settle(self, n=120):
        self._step(n)

    def _set_gripper(self, width):
        for j in self.finger_joints:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, width, force=40)

    def _open_gripper(self):
        self._set_gripper(0.04); self._step(60)

    def _close_gripper(self):
        self._set_gripper(0.0); self._step(90)

    def move_to(self, pos, steps=240, tol=0.01):
        """Move end effector to `pos` (gripper down) using IK + position control."""
        for _ in range(steps):
            q = p.calculateInverseKinematics(self.robot, self.ee_link, pos, GRASP_ORN,
                                             maxNumIterations=50, residualThreshold=1e-4)
            for j in self.arm_joints:
                p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, q[j], force=200, maxVelocity=1.5)
            self._step(1)
            if np.linalg.norm(np.array(self.ee_position()) - np.array(pos)) < tol:
                break
        self._step(10)

    def ee_position(self):
        return p.getLinkState(self.robot, self.ee_link)[0]

    # ------------------------------------------------------------------ state
    def get_state(self) -> EnvState:
        objs = {}
        for n, b in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(b)
            objs[n] = ObjectState(n, self.kinds[n], tuple(np.round(pos, 5)), tuple(np.round(orn, 5)),
                                  held=(n == self._held))
        g = p.getJointState(self.robot, self.finger_joints[0])[0] / 0.04
        return EnvState(objs, tuple(np.round(self.ee_position(), 5)), float(np.clip(g, 0, 1)))

    def park(self):
        """Move the arm to the side so it does not occlude the camera."""
        self.move_to((0.25, 0.45, 0.40))

    def render(self, top_down: bool = False):
        """RGB + depth + segmentation from a fixed camera (angled by default, or straight down)."""
        if top_down:
            view = p.computeViewMatrix([0.55, 0.0, 0.9], [0.55, 0.0, 0.0], [1, 0, 0])
        else:
            view = p.computeViewMatrix([1.05, 0.0, 0.7], [0.5, 0.0, 0.0], [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(60, 1.0, 0.05, 2.0)
        w, h, rgb, depth, seg = p.getCameraImage(self.img_size, self.img_size, view, proj,
                                                 renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(rgb, (h, w, 4))[..., :3].astype(np.uint8)
        return rgb, np.reshape(depth, (h, w)), np.reshape(seg, (h, w))

    # ---------------------------------------------------------------- actions
    def step(self, action: dict) -> tuple[EnvState, dict]:
        """Execute a primitive. Actions:
          {"type":"pick","object":name}
          {"type":"place","target":(x,y)}              # place held object at xy on table
          {"type":"place_in","container":name}         # drop held object into container
          {"type":"move","pos":(x,y,z)}
          {"type":"release"}
        Returns (next_state, info). Logs (s, a, s').
        """
        s0 = self.get_state()
        t = action["type"]
        ok = True
        if t == "pick":
            ok = self._pick(action["object"], action.get("position"))
        elif t == "place":
            ok = self._place_xy(action["target"])
        elif t == "place_in":
            c = action.get("position") or p.getBasePositionAndOrientation(self.objects[action["container"]])[0]
            ok = self._place_xy((c[0], c[1]), z_extra=0.03)
        elif t == "move":
            self.move_to(action["pos"])
        elif t == "release":
            self._release()
        elif t == "push":
            ok = self._push(action["object"], action["direction"], action["distance"])
        else:
            raise ValueError(f"unknown action {t}")
        self._settle(30)
        s1 = self.get_state()
        info = {"success": bool(ok)}
        self.log.append({"s": s0, "a": action, "s_next": s1, "info": info})
        return s1, info

    def _pick(self, name, position=None) -> bool:
        """`position` (x, y) overrides the true object position, e.g. a *perceived* position."""
        b = self.objects[name]
        pos = position if position is not None else p.getBasePositionAndOrientation(b)[0]
        self._open_gripper()
        self.move_to((pos[0], pos[1], 0.20))
        self.move_to((pos[0], pos[1], TABLE_Z + CUBE_SIZE / 2 + 0.01))
        if self.force_grasp_fail:
            self.force_grasp_fail = False       # one-shot injected failure
        else:
            self._close_gripper()
        self.move_to((pos[0], pos[1], 0.25))
        # verify grasp by lift: object must follow the hand
        new = p.getBasePositionAndOrientation(b)[0]
        held = new[2] > TABLE_Z + 0.08
        self._held = name if held else None
        if held:  # remember where the cube sits relative to the hand, to centre placements
            ee = self.ee_position()
            self._grasp_offset = (new[0] - ee[0], new[1] - ee[1])
        return held

    def _push(self, name, direction, distance) -> bool:
        """Sweep the closed gripper through the object along `direction` (xy) so it should travel `distance` m.
        Outcome depends on friction/mass the caller does not know: low-friction cubes slide past the target."""
        pos = np.array(p.getBasePositionAndOrientation(self.objects[name])[0])[:2]
        d = np.asarray(direction, dtype=float); d = d / np.linalg.norm(d)
        start, end = pos - d * 0.08, pos + d * (distance - 0.03)       # contact offset: finger 1 cm + half cube 2 cm
        z = TABLE_Z + 0.03
        self._set_gripper(0.0); self._step(60)
        self.move_to((start[0], start[1], 0.20))
        self.move_to((start[0], start[1], z))
        self.move_to((end[0], end[1], z), steps=500, tol=0.005)
        self.move_to((end[0], end[1], 0.20))
        self._open_gripper()
        return True

    def _place_xy(self, xy, z_extra=0.0) -> bool:
        if self._held is None:
            return False
        # shift the hand so the *cube* (not the hand) lands on the target
        tx, ty = xy[0] - self._grasp_offset[0], xy[1] - self._grasp_offset[1]
        self.move_to((tx, ty, 0.25))
        self.move_to((tx, ty, TABLE_Z + CUBE_SIZE + 0.02 + z_extra))
        self._release()
        self.move_to((tx, ty, 0.25))
        return True

    def _release(self):
        self._open_gripper()
        self._held = None

    def disturb(self, name, dxy=(0.08, 0.0)):
        """Failure injection: teleport an object (unexpected displacement)."""
        pos, orn = p.getBasePositionAndOrientation(self.objects[name])
        p.resetBasePositionAndOrientation(self.objects[name], [pos[0] + dxy[0], pos[1] + dxy[1], pos[2]], orn)
        self._settle(30)

    def close(self):
        p.disconnect(self.cid)

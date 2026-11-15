# Project ECHO: Predictive World Models for Failure Detection and Recovery in Language-Conditioned Manipulation

*Technical report. Simulation only (PyBullet). All numbers are reproducible from the scripts in `scripts/`; raw outputs are in `results/`.*

## 1. Summary

ECHO is a simulated tabletop robot (Franka Panda) that parses a natural-language instruction, plans a sequence of skills,
predicts the outcome of each action with a learned world model, compares the prediction with what it observes through a camera,
and replans when they disagree.

**Main findings**

1. **Detection plus recovery is what matters.** Under injected failures, open-loop execution drops to 7-10% task success.
   Detecting the failure *and* replanning restores 75-98%. Detection alone does not help (it stops the run).
2. **The learned world model detects failures well, without privileged signals.** It flags 100% of failed grasps and of bad steps in live runs
   (6-8% false alarms). A rule-based monitor that does not use the world model catches 60-63% of bad steps, and only because it reads the
   simulator's own success flag.
3. **The world model did *not* improve task success over a simple rule-based recovery baseline.** The full system is statistically
   indistinguishable from "rules + recovery" in every scenario (paired sign tests, p >= 0.125). The first research question
   ("does predictive world modelling improve task completion?") is therefore **not supported by this evidence**.
4. **A camera costs little**, except after a cube is knocked (87% -> 75%).

## 2. System

| Module | Implementation |
|---|---|
| Environment | PyBullet, Franka Panda, 3 coloured cubes, 1 open container, optional obstacles; IK + position control |
| Language | Rule-based parser (deterministic, no API) -> goals such as `in(red_cube, blue_container)`, `left_of(green_cube, red_cube)` |
| Planning | Goal -> skills (`pick`, `place_in`, `place_rel`); placement targets resolved at execution time and checked for clearance |
| Perception | OpenCV colour segmentation on a top-down camera, back-projected to table coordinates |
| World model | Ensemble of 5 MLPs (3x256) predicting the normalised state change `s' - s` from `(s, a)` |
| Monitor | `e_t = max over objects of ||predicted - observed position||`; failure if `e_t` > 5 cm |
| Recovery | Goal-level replanning: re-check every goal against the observed state, re-plan skills for unmet goals |

Frame convention: robot at the origin facing +x; left = +y; "in front of" = toward the robot.

### Design decision: train on expected outcomes
Grasps fail at random about 15% of the time in the training data. A model trained on all outcomes learns to hedge (predict a
half-lifted cube), so *normal* grasps look surprising: 13-15% false alarms live. Training on **successful transitions only** makes the model
predict the *expected* outcome, so a failed grasp becomes a clean surprise of about 22 cm against about 1 cm for a normal action.

## 3. World model

Data: 6,000 transitions from 1,500 random episodes (random picks and placements, 15% injected grasp failures, 15% random object
displacements), split 80/10/10 **by episode**.

![prediction quality](../results/figures/fig1_prediction.png)

| Test set (600 transitions) | Mean object position error | "Nothing changes" baseline |
|---|---|---|
| All actions | 1.49 cm | 6.44 cm |
| Successful actions | 0.78 cm | 7.49 cm |

**Multi-step rollout** (the model fed its own predictions; test windows with a continuous, successful trajectory):

| Horizon | Windows | Rollout error | Copy-state baseline |
|---|---|---|---|
| 1 | 515 | 0.78 cm | 7.49 cm |
| 2 | 295 | 1.33 cm | 9.17 cm |
| 3 | 148 | 1.97 cm | 13.64 cm |
| 4 | 56 | 2.53 cm | 12.99 cm |

Error grows roughly linearly rather than compounding. Uncertainty: ensemble disagreement correlates with error (0.49 val, 0.60 test), which is
informative but not strong.

## 4. Failure detector

![detector](../results/figures/fig2_detector.png)

On held-out data the discrepancy separates failed grasps (all around 22 cm) from normal actions (AUROC 0.994). At the deployed 5 cm threshold:
failed-grasp recall is 1.00 (n=54). The false-alarm rate on the *random-action* test data is **12%** (pick 10%, place 21%, place-in 4%); the high place figure
comes from random placements that collide with other objects, which the planner avoids in real task runs. In live runs the step-level false-alarm rate was 6-8%.

## 5. End-to-end evaluation

60 held-out layouts per cell (layouts 200-259, never used for tuning), three task types, identical layouts and injected failures across all
variants. Success is always scored on the true simulator state.

Variants: **Open loop** (Baseline A) executes the plan once. **Verify + retry** has no step monitoring and only re-plans after a failed final check.
**Rules + recovery** monitors with the skill's own success flag and goal checks. **WM detect only** uses the world-model detector but aborts on detection.
**ECHO** uses the world-model detector plus replanning.

![task success](../results/figures/fig3_success.png)

Ground-truth object state:

| Scenario | Open loop | Verify + retry | Rules + recovery | WM detect only | ECHO |
|---|---|---|---|---|---|
| Nominal | 0.98 | 0.98 | 0.98 | 0.95 | 0.98 |
| Failed grasp | 0.10 | 0.98 | 0.98 | 0.10 | 0.98 |
| Cube knocked after placing | 0.07 | 0.83 | 0.90 | 0.07 | 0.87 |
| 2 obstacles (unseen arrangement) | 0.92 | 0.92 | 0.92 | 0.88 | 0.92 |

Camera only (perceived state; perception error 0.43 cm mean, 0.9 cm max, no missed objects in 240):

| Scenario | Open loop | Verify + retry | Rules + recovery | WM detect only | ECHO |
|---|---|---|---|---|---|
| Nominal | 0.98 | 0.98 | 0.98 | 0.97 | 0.98 |
| Failed grasp | 0.10 | 0.95 | 0.97 | 0.10 | 0.97 |
| Cube knocked after placing | 0.07 | 0.77 | 0.82 | 0.07 | 0.75 |
| 2 obstacles | 0.88 | 0.90 | 0.90 | 0.87 | 0.90 |

95% Wilson intervals and all paired tests are in `results/phase5_summary.txt` and `results/phase5_percep_summary.txt`.

**Paired comparisons (ECHO-only wins / other-only wins)**

| Comparison | Failed grasp | Cube knocked |
|---|---|---|
| ECHO vs open loop (true state) | 53 / 0 | 48 / 0 |
| ECHO vs open loop (camera) | 52 / 0 | 41 / 0 |
| ECHO vs rules + recovery (true state) | 0 / 0 | 0 / 2 (p = 0.50) |
| ECHO vs rules + recovery (camera) | 0 / 0 | 0 / 4 (p = 0.125) |

## 6. Discussion

**What the evidence supports.** Prediction-based failure detection is accurate and needs no privileged signal; replanning turns detections into recovered tasks.
**What it does not support.** A task-success advantage for the world model over simpler monitors. In the knocked-cube scenario the point estimate
favours the rule-based variant (87% vs 90% true state, 75% vs 82% camera), and the false alarms of the world-model detector plausibly explain that
(each wastes a replan), though the differences are within noise at n=60.

Why the world model adds little here: the failures are *visible after the fact* in a few state variables (is the cube lifted, did it move), so cheap checks
suffice once a monitor exists. A world model should matter when it can *prevent* failures by evaluating alternatives before acting. That needs a
richer action space than this task has, which is why predictive planning (Baseline B) was not built: the planner already rejects collision-prone placements
geometrically, so a model scoring those same candidates would pick the same ones.

**Residual failures are physical**: a target region fully blocked by other objects, and re-grasps that close the fingers on a container wall.

## 7. Limitations and threats to validity

- The rule-based baseline reads simulator-provided success flags and ground-truth goal checks, which a real robot would not have. This flatters it.
- The 5 cm detection threshold was tuned on layouts 0-23; reported results use layouts 200-259.
- One scene family (3 cubes + 1 container), one simulator, one failure model (slipped grasp, one knocked object). Obstacles are not part of the world model state.
- Perception is classical colour segmentation: it needs distinctly coloured objects and a fixed camera.
- No physical-robot validation. Camera-only results are simulated RGB, not real images.
- The experiments are small (n=60 per cell); differences below about 10 points are not resolvable.

## 8. Reproducing

See `README.md`. Videos of nominal execution, failed-grasp recovery and knocked-cube recovery are in `results/videos/`.

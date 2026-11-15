"""Phase 2 demo: parser tests (no simulator) + language-conditioned execution."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cv2
from echo import ManipulationEnv
from echo.parser import parse, ParseError
from echo.executor import execute_instruction

# ---- parser unit checks -------------------------------------------------------
EXPECT = {
    "Pick up the red cube, place it inside the blue container, and move the green cube to the left of the red cube.":
        ["in(red_cube,blue_container)", "left_of(green_cube,red_cube)"],
    "Put the green block in the blue bin": ["in(green_cube,blue_container)"],
    "Grab the red cube": ["picked(red_cube)"],
    "Move the blue cube behind the green cube then put the red cube to the right of the blue cube":
        ["behind(blue_cube,green_cube)", "right_of(red_cube,blue_cube)"],
}
for text, want in EXPECT.items():
    got = [f"{g.kind}({g.subject},{g.ref})" if g.ref else f"{g.kind}({g.subject})" for g in parse(text).goals]
    assert got == want, f"\n  text: {text}\n  got:  {got}\n  want: {want}"
for bad in ["Dance around", "Put the purple cube in the blue container", "Move the red cube to the left"]:
    try:
        parse(bad, scene_objects=["red_cube", "blue_container"]); raise SystemExit(f"should have failed: {bad}")
    except ParseError as e:
        pass
print("parser checks passed\n")

# ---- end-to-end in simulation -------------------------------------------------
INSTR = "Pick up the red cube, place it inside the blue container, and move the green cube to the left of the red cube."
env = ManipulationEnv(seed=0)
res = execute_instruction(env, INSTR)
print(res.report())
env.park()
rgb, _, _ = env.render(top_down=True)
cv2.imwrite("phase2_result_top.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
env.close()

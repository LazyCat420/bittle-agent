---
part: Digital twin viewer
status: shipped
updated: 2026-09-28
---

# The digital twin viewer, and the seven plans that never reached this shelf

Chapter 00 says the documentation exists because work kept being tracked "in
loose `PLAN-*.md` files nobody could find." This repo had eight of them sitting
in its own root the whole time.

The daily documentation review on **2026-09-28** reported bittle-agent as
healthy: nine chapters, every one classified, nothing stale, nothing unfiled.
It also reported, in a section nobody had acted on, `bittle-agent: (repo root)
(10)` — ten markdown documents beside the chapters that the audit reads but
this shelf never indexed. Eight of them are plans. **Seven of those eight are
about a subsystem that has no chapter at all.**

Chapters 01–08 are all `part: RL walking trainer`. They document MuJoCo Warp,
Brax PPO, terrain, gaits, the fly-brain A/B. They say nothing about the
**browser digital twin** — the Three.js viewer in `static/js/`, its obstacle
courses, its ground-contact solver, its root motion. That subsystem is 1,332
lines of `static/js/viewer.js` plus `obstacle_course.js`, `app/motion/obstacles.py`
and seven test files, and a reader of this documentation would not learn it
exists.

## What the root plans were, and what became of them

All seven shipped, all on **2026-09-06**, each with tests landed in the same
commit. Verified by reading the commits and the current code, not by trusting
the plan files' own status lines — `PLAN_FIX_TERRAIN_RENDER.md` still says
"Awaiting User Approval" at the top of a fix that shipped the same day.

| Root plan | What it asked for | Shipped as | Tests |
|---|---|---|---|
| `PLAN_OBSTACLE_COURSE_AND_HIERARCHICAL_CONTROL.md` | 3D obstacle courses, terrain clearance, the 3-tier control architecture | `ed8aee4` — `app/motion/obstacles.py`, `static/js/obstacle_course.js` | `tests/test_obstacles.py` |
| `PLAN_FIX_TERRAIN_RENDER.md` | terrain visible and framed in the viewport | `db66b71` — obstacles aligned to +X, camera auto-framing | `tests/test_obstacles.py` |
| `PLAN_GRAVITY_AND_GROUND_CONTACT.md` | the torso drops when it sits instead of kicking in mid-air | `ede4acd` — gravity ground-contact solver | `tests/test_ground_contact.py` |
| `PLAN_GROUND_PLANE_COLLISION_FIX.md` | paws stop sinking 1–15 mm through the floor | `18cf260` — supremum contact solver | `tests/test_ground_plane_collision.py` |
| `PLAN_FORWARD_BACKWARD_LOCOMOTION.md` | stop walking in place; translate through the scene | `652a597` — root motion on the D-pad and keyboard | `tests/test_locomotion_displacement.py` |
| `PLAN_FIX_FORWARD_SIDEWAYS_LOCOMOTION.md` | stop crab-walking: the mesh faced −Z while displacement went +X | `cd1c511` — `YXZ` → `YZX` Euler order, one heading axis | `tests/test_mesh_locomotion_alignment.py` |
| `PLAN_WALK_AND_STAIR_TRAINING.md` | kinematic root motion + pitch conformance so it can reach the stairs | `ede4acd` — stair climbing, autonomous episodes | `tests/test_scene_locomotion_and_training.py` |

The eighth, `PLAN_FLY_BRAIN_AB_BALANCE_ROCKS_AND_BALL.md` (2026-09-12), is the
one plan that *was* filed: chapter 08 cites it by name. It is the exception that
shows the rule — the fly-brain plan got a chapter because a session wrote one,
and the other seven did not because nothing makes that happen.

## What the founding plan now says that is not true

`PLAN.md` (2026-07-27) is still the repo's architecture and safety document, and
§1, §2 and §4 still describe the shipped system: the per-joint clamp against
`firmware_limit ∩ wire_range`, the latching e-stop, `app/safety.py` as the one
place policy lives.

**§3 is superseded.** It reads:

> **Out (deliberate):** PyBullet. The original plan called for it, but a physics
> sim is a multi-week project on its own […] and none of it makes the *hardware*
> safer.

A physics sim did land — it was simply not PyBullet. `trainer/` is MuJoCo Warp
with Brax PPO, it is the subject of chapters 01–08, and it was a multi-week
project exactly as §3 predicted. The reasoning in §3 was sound and the verdict
was overturned on purpose; what is wrong is only that the document still states
the verdict as current. The same substitution answers Milestone 2 of
`PLAN_OBSTACLE_COURSE_AND_HIERARCHICAL_CONTROL.md` ("PyBullet / Rigid-Body
Physics Sim"), which is shipped under a different engine.

## What is left

Two things, both from the hierarchical-control plan's roadmap:

1. **Tier 2 never reaches the robot.** The plan's 3-tier architecture is
   firmware reflexes ⇄ neural locomotion policy ⇄ GLM choreographer. Tier 3
   exists (`app/agent.py`), tier 1 exists (the OpenCat firmware), and tier 2 is
   trained and exported — but only to `.npz`, by `trainer/policy/export.py`.
   Nothing in `app/` or `static/js/` reads it: `grep -rn onnx app/ static/js/`
   returns nothing, and the only policy path into the viewer is
   `GET /runs/{run_id}/rollout`, which replays a **recorded** episode. So the
   trained policy cannot yet drive either backend live. A reader who saw
   chapters 01–08 alone would reasonably assume it could.
2. **Milestone 4 (Vision-Language-Action) is untouched.** Listed as a roadmap
   endpoint, never started. Recorded here so it is a known gap rather than an
   open question.

## Why this chapter, rather than eight

The plans are shipped history. Copying seven of them onto this shelf would add
seven chapters that nobody needs to read and that would then age here instead of
in the root. What was missing was not the plans — they are in git, dated, beside
their tests — it was the **index**: nothing on this shelf admitted the viewer
existed, so the audit's clean bill of health for this repo was true of the
chapters and false of the repo.

The root files stay where they are. They are the primary record of their own
fixes, they are cited above, and deleting them would take the "why" with them.

---

*Filed by the daily documentation review, 2026-09-28. Evidence is the commit
range and the current tree; no test run was made in this pass, so the plans are
`shipped`, not `verified`.*

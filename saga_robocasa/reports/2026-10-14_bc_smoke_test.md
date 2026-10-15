# BC smoke test on the controlled seed-0 episode — 2026-10-14

**Scope.** This is a pipeline/overfitting test, not a policy result. The environment is a
*controlled benchmark episode*: seed-0 `PickPlaceCounterToSink` metadata replayed with the target
object replaced by `canned_food_18` and repositioned (`controlled_episode.py`), not a randomly
sampled RoboCasa episode.

## Data (code `e48d111`, schema `saga_spot_demo_v1`)

| set | episodes | transitions | notes |
|---|---|---|---|
| fixed-base manipulation | 6 (`demo_000`–`005`, perturb seeds −1, 1–5) | 4,252 | all validated, bit-exact replay |
| navigation + manipulation | 1 (`nav_000`, start 1.42 m from pregrasp) | 872 | validated, bit-exact replay |

## Model

PointNet (6-channel points: base-frame xyz + oracle object/sink/robot labels, 1,024 deterministic
points) + MLP on proprioception (arm qpos, gripper qpos, TCP in base frame) and oracle goal (object
and sink centre in base frame). Predicts an 8-step sequence of `action_policy` (desired base-frame
arm delta, desired base-frame base velocity, gripper); only the first action is executed. 0.56 M
parameters, 3,000 steps, batch 64, CPU (this environment's PyTorch has no CUDA). Phase is not an
input. Trained on `nav_000` only (`e9b73b9`).

## 1. Offline one-demo overfit — passed

Normalized MSE 0.394 → 0.0015. Gripper sign accuracy 871/872.

| dim | MAE (active steps) | vs zero | vs train mean |
|---|---|---|---|
| eef translation (x, y, z) | 0.0047, 0.0034, 0.0071 | 6.2×, 24.8×, 7.7× | 6.9×, 25.5×, 8.1× |
| eef rotation (x, y, z) | 0.0002, 0.0005, 0.0007 | 9.2×, 7.3×, 18.6× | 9.0×, 7.2×, 21.2× |
| base fwd / side / yaw (159 active steps) | 0.0027, 0.0049, 0.0044 | 111×, 44×, 37× | 102×, 44×, 30× |

Every expert phase has active-component MAE ≤ 0.009.

## 2. Closed-loop replay from the training initial state — failed

Same initial simulator state as `nav_000` (hash match). Horizon 1,200 steps; abort on collision,
joint limit or object drop.

| run | outcome | min base error | first gripper close | grasp / place / `_check_success` |
|---|---|---|---|---|
| expert (`nav_000`) | success, 872 steps | 0.9 cm (gate ≤ 1 cm) | step 255 | yes / yes / True |
| BC, unfiltered | collision abort at step 248 (base vs fixture) | 2.45 cm | step 173 (empty) | no / no / False |
| BC + safety filter* | collision abort at step 334 (gripper vs sink) | 1.51 cm | step 183 (empty) | no / no / False |

\*Deployment-only diagnostic, model unchanged: normalized predictions clipped to the training range
and rate-limited to the largest per-step change in the training actions. Clipping was active on
228 / 334 steps (the rate limit never engaged); out-of-range components appeared from step 0 and were
largest on arm rotation (|normalized| up to 2.4).

**Diagnosis.** Navigation tracks the expert to ~1 cm for the first ~100 steps, then the final
alignment lags (policy 4 cm vs expert 0.9 cm near step 165). The policy never reaches the expert's
navigation gate, closes the gripper prematurely (~step 175–185 vs 255), and from there the states
are outside the single demonstration: without the filter the base command grows to 2.5× the training
range and drives into a fixture; with the filter the base stays bounded but the arm moves into the
sink. This is the expected compounding-error failure of single-demo behavior cloning; clipping does
not address it.

## 3. Generalization to perturbed starts — not tested

## Next

1. ≥ 10 navigation + manipulation demos with starts concentrated around the final approach (far
   starts with lateral/yaw offsets; near-goal starts 1–10 cm, inside/outside the 1 cm / 0.01 rad gate).
2. DAgger-style recovery demos: run the policy, detect the first divergence (premature gripper close
   or base command disagreeing with the expert at the visited state), restore that exact simulator
   state, let the expert recover, and record the expert transitions separately from policy failures.
3. Phase/action-balanced sampling, goal-relative conditioning (pregrasp position/yaw error), and the
   A (navigation) → B (manipulation) → C (combined) evaluation progression before any diffusion/
   action-horizon policy.

---

# Update: more data, recovery demos, and the A/B/C progression

## Data added (code `6780801`, schema `saga_spot_demo_v1`, all validated with bit-exact replay)

| set | episodes | notes |
|---|---|---|
| navigation + manipulation, far starts | 6 (`nav_far_001`–`006`) | 1.4–1.9 m, yaw offsets up to ±1.3 rad |
| near-goal starts | 4 (`nav_near_001`–`004`) | 2.7–8.2 cm, yaw up to ±0.11 rad |
| just inside / outside the 1 cm / 0.01 rad gate | 2 + 2 (`nav_gate_in_*`, `nav_gate_out_*`) | 0.7–0.8 cm vs 1.2–1.3 cm |
| DAgger-style recoveries | 8 (`recover_*`) | exact simulator states visited by the two failed BC rollouts (divergence at step 139 ± 20, and the premature-close steps 173 / 183), expert recovers |

In every expert episode navigation ends inside the gate (≤ 9.8 mm, ≤ 9.6 mrad) before the first
gripper close. Failed policy rollouts are kept in `data/evaluations`, separate from expert data.

## Training changes (`7b3ae89`)

Source- and phase-balanced sampling (fixed-base 83% of samples → 47% of sampling mass; NAV_ALIGN /
NAV_SETTLE 0.8% → 8.5%), goal-relative conditioning (pregrasp position error and yaw error), phase
subsets, episode-level validation splits. Phase is never an input. Single-step execution only.

## Results (closed loop, no safety filter unless stated)

**A. Navigation only** (`bc_A_nav`; held out: `nav_far_006`, `nav_near_004`, `nav_gate_out_002`).
Success = 1 cm / 0.01 rad gate held for 10 steps, any gripper close = failure.

| start | result | note |
|---|---|---|
| far 6 (held-out demo) | timeout (600) | position 3.1 mm but yaw stalls at 24 mrad |
| far 7 (unseen) | **gate reached** (step 170) | |
| far 8 (unseen) | timeout (600) | position 5.5 mm but yaw stalls at 22 mrad |
| near 4 (held-out demo) | **gate reached** (step 57) | |
| near 5 (unseen) | **gate reached** (step 49) | |
| near 6 (unseen) | **gate reached** (step 29) | |

4/6; no collisions; the gripper never closed during navigation. Failure mode: small yaw corrections
are predicted as ~0, so the base stops 2× outside the yaw gate.

**B. Manipulation only from a valid pregrasp state** (perturb seeds 5 = held-out demo, 6–8 unseen).

| model | base handling | success | notes |
|---|---|---|---|
| with fixed-base demos | policy output | 0/4 (1 grasp) | base drifted 3.8–16 cm in every run |
| without fixed-base demos | policy output | 0/4 (1 grasp) | same |
| with fixed-base demos | zero (`--zero-base`) | **1/4** | 2 arm–cabinet collisions, 1 wrist (wr0) limit |
| without fixed-base demos | zero (`--zero-base`) | **2/4** | 1 arm–cabinet collision, 1 wrist limit |

The base drift with policy output is a deployment interaction: tiny predicted base velocities
(≈0.03 m/s) are turned by the deadband-compensating `BaseActionModel` into ≥ 0.25 commands that
really move the base. 4 starts per model is too few to rank the two training sets.

**C. Combined** (held out: `demo_005`, `nav_far_006`, `nav_near_004`, `nav_gate_out_002`; starts far 6,
far 7, near 5; base output raw or with a 0.03 m/s / 0.03 rad/s deadzone): **0/12 success**, 1 grasp.
Navigation gets to 0.1–4.3 cm, but several runs close the gripper 2.4–6.4 cm from the pregrasp pose
(outside the gate), and runs end in wrist-limit aborts, arm/base–counter or cabinet collisions, or
timeouts.

## Summary of what is and is not established

- Offline fitting: works on all splits (held-out gripper accuracy ≥ 0.99).
- Closed loop: navigation-only reaches the gate on 4/6 starts; manipulation-only succeeds on 1/4 and
  2/4 with the base held fixed; the full combined task does not succeed (0/12).
- Not established: anything about generalization beyond this single controlled layout.

---

# Hybrid autonomous deployment (Week 1 expectation 3)

**This is a HYBRID system**: learned navigation (Stage A checkpoint `bc_A_nav.pt`) → **rule-based**
transition (pregrasp error ≤ 0.01 m and ≤ 0.01 rad for 10 consecutive steps; permanent switch) →
learned manipulation (Stage B checkpoint `bc_B_manip_without_fixed.pt`, the better of the two
B models). No expert actions after reset; the expert code only defines the pregrasp goal and the
safety thresholds. Rules: during navigation the gripper is forced open; during manipulation the
base velocity is forced to exactly zero before `BaseActionModel`. Collision, joint-limit (0.05 rad
margin) and object-drop aborts are unchanged. Script: `policy/deploy_hybrid.py`. Every rollout has
an external-camera video and full metrics in `data/evaluations/hybrid_*`.

Seen in training: start far 1 is a training episode of both checkpoints (`nav_far_001`); far 6 is
A's held-out validation episode and unseen by B; far 7, far 8, near 5, near 6 were never seen by
either checkpoint.

## Variant 1 — arm rule "zero OSC delta" (as first implemented)

| start | gate | grasp | placement | `_check_success` | outcome |
|---|---|---|---|---|---|
| far 1 (known) | no | no | no | False | gate not reached in 600 steps |
| far 6 (A val) | no | no | no | False | gate not reached in 600 steps |
| far 7 (unseen) | no | no | no | False | gate not reached in 600 steps |
| far 8 (unseen) | no | no | no | False | gate not reached in 600 steps |
| near 5 (unseen) | yes (52) | no | no | False | gripper–sink collision at step 601 |
| near 6 (unseen) | yes (28) | no | no | False | gripper–sink collision at step 553 |

Diagnosis: a zero delta does **not** suppress arm motion. With the OSC in `achieved` goal-update
mode there is no restoring force, so while the base accelerates and turns the arm drifts by up to
0.95 rad (TCP 41 cm relative to the base; under the Stage A policy's own arm actions the drift was
≤ 0.09 rad). The navigation policy's observations leave the training distribution after ~82 steps
and it stalls ~4.8 cm from the goal.

## Variant 2 — arm rule "hold the post-reset gripper pose relative to the base" (fixed stabilizer, no task information)

| start | gate (step) | grasp (step) | placement | `_check_success` | failure (step) |
|---|---|---|---|---|---|
| far 1 (known to A and B) | **yes** (184) | **yes** (323) | no | False | wrist `wr0` joint limit, margin 0.038 (394) |
| far 6 (A val, unseen by B) | **yes** (167) | **yes** (307) | no | False | `wr0` joint limit, margin 0.042 (388) |
| far 7 (unseen) | **yes** (159) | no | no | False | gripper–sink collision (1174); can pushed 31 cm |
| far 8 (unseen) | **yes** (166) | **yes** (297) | no | False | `wr0` joint limit, margin 0.039 (376) |
| near 5 (unseen) | **yes** (51) | no | no | False | gripper–sink collision (1067); can pushed 31 cm |
| near 6 (unseen) | **yes** (29) | **yes** (167) | no | False | `wr0` joint limit, margin 0.044 (237) |

**Navigation gate 6/6, grasp 4/6, placement 0/6, full `_check_success()` 0/6.**

Diagnosis: every grasping rollout lifts the can ~10.5 cm and then, during transport towards the
sink, drives the wrist pitch `wr0` to ≈ +1.79 rad (limit 1.833; abort at 0.05 rad margin). Across
the 15 expert navigation demos `wr0` never exceeds 1.608 rad. The remaining two rollouts never
close on the can, push it along the counter and hit the sink.

## Week 1 status

1. Whole-task data with navigation + manipulation: done (15 navigation demos, 8 recovery demos,
   6 fixed-base demos; all validated with bit-exact replay).
2. Imitation-learning policy: PointNet + MLP behaviour cloning built and evaluated in stages; the
   diffusion policy has not been started.
3. Autonomous completion: the hybrid system navigates autonomously to the pregrasp gate on 6/6
   starts and grasps on 4/6, but has **not yet completed the task (0/6)**; the blocking failure is
   the wrist joint limit during learned transport.

---

# Stage B2: transport-focused manipulation retraining (2026-10-15)

## Policy-input inspection (before training)

Stage B (`bc_B_manip_without_fixed.pt`) inputs: arm joint positions incl. `wr0` — **yes** (6);
joint-limit margins — **no** (only implicit in qpos); TCP pose — **position only** (base frame), no
orientation; object-to-TCP pose — **no** explicit offset (both positions present), no object
orientation; sink goal — interior-centre **position** (base frame). B2 adds
`proprio_mode="extended"` (+18): joint-limit margins (6), TCP orientation (first two rotation-matrix
columns, base frame, 6), object−TCP offset (3), object up-axis (3). The original 10-D mode is
unchanged and still the default.

## Hybrid transport recovery data (`data/hybrid_transport_recovery`)

States restored 20/40/60 steps before each abort of the six arm-hold hybrid rollouts; provenance
(source rollout, step, initial `wr0`, object pose, failure type, grasped-at-restore) in each
demo's `recovery` metadata. New expert capability: mid-carry start at GRASP_HOLD
(`prepare_restored_manipulation_start`), lifting relative to the nominal grasp height.

| source failure | restores | expert recovery | validated (exact replay + `_check_success`) |
|---|---|---|---|
| wrist limit: far 1, far 6, far 8, near 6 | 12 (`wr0` ≈ 1.63 at restore, can grasped) | **12/12 success** | 12/12 |
| gripper–sink collision: far 7, near 5 | 6 (can pushed ~31 cm to the sink edge) | 0/6 (2 refused: gripper 2.5 mm from the sink; 4 no feasible transport plan) | — (kept in `data/failures`) |

## Training (`bc_B2_manip.pt`)

Manipulation phases of the fixed-base demos, the full-task demos and DAgger recoveries, plus the 12
transport recoveries; held out by episode: `demo_005`, `nav_far_004`, `htr_far_8_*` (3).
Balanced sampling with TRANSIT/RELEASE_DESCEND ×3, RETRACT/RETRACT_UP/RETRACT_BACK ×2 and
transport-recovery source ×3; 6,000 steps. Not phase-conditioned (phase is not an input).

| source | samples | sampling mass |
|---|---|---|
| fixed_base | 3,525 | 15.8% |
| navigation (full-task demos) | 9,892 | 16.0% |
| recovery (navigation DAgger) | 5,676 | 15.9% |
| hybrid_transport_recovery | 4,604 | 52.3% |

By phase: TRANSIT 11.2%, RELEASE_DESCEND 11.3%, RETRACT 10.2%, RETRACT_BACK 9.5%, RETRACT_UP 9.0%,
APPROACH_HOVER 2.1%, DESCEND 2.0%, CLOSE 2.4%.

Auxiliary joint-limit penalty (weight 0.5, band 0.10 rad inside the demonstrated joint range):
predicted joint motion `M @ a` (arm Jacobian pseudo-inverse composed with the OSC delta scaling,
precomputed at every recorded state, `policy/compute_joint_maps.py`) is penalized when it moves a
joint further outward inside the band. Validated against recorded motion: per-joint correlation
0.88–1.00 (`wr0` 0.99), sign agreement 0.93–0.94; actual one-step motion ≈ 25–30% of `M @ a` (the
OSC covers part of the goal offset per step), so the map is conservative. Offline: train MSE
0.0030, held-out 0.0195, held-out gripper accuracy 0.994.

## Deployment safety layer

`spot_utils.joint_safety_projection` (a safety layer around the learned policy, not an expert
action): predicts next joint margins with the live map and, if any would fall below 0.08 rad
while the action reduces it, scales only the arm rotation delta. Raw and projected actions are
logged separately.

## Stage B2 closed-loop evaluation (fixed base, joint-safety projection 0.08 rad)

| start | seen | grasp | placed | `_check_success` | outcome (step) | projection active |
|---|---|---|---|---|---|---|
| perturb 5 | held-out demo | no | no | False | arm–upper-cabinet collision (175) after an insecure close and empty lift | 0 |
| perturb 6 | unseen | yes | no | False | object dropped outside the sink (279) | 0 |
| perturb 7 | unseen | yes | **yes** | False | gripper–sink collision while retracting (571) | 0 |
| perturb 8 | unseen | no | no | False | cabinet-door collision (771) | 0 (2 unresolved) |
| perturb 9 | unseen | no | no | False | gripper–counter collision (165), closed 7.8 cm beside the can | 0 |
| perturb 10 | unseen | no | no | False | arm–upper-cabinet collision (176) after an empty lift | 0 |

**0/6 — Stage B2 did not meet the prerequisite** (at least one held-out/unseen success), so the
hybrid rerun was **not** performed. B2 is worse than the original Stage B on the overlapping starts
(original B with base held at zero completed perturb 6 and 8). The projection never activated:
B2 no longer approaches the wrist limit, but it now fails earlier, mostly in grasp alignment
(insecure or missed closes followed by lifting into the upper cabinets). Likely cause (not yet
isolated): the transport-recovery episodes start mid-carry and, with the source and phase boosts,
take 52% of the sampling mass while approach/descend/close get ~2% each.

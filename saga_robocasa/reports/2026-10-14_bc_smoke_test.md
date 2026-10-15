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

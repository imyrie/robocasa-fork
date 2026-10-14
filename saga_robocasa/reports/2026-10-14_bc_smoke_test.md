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

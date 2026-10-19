# Controlled CounterToSink benchmark `counter_to_sink_v1` (frozen)

One controlled RoboCasa `PickPlaceCounterToSink` scene with `SpotWithArmFloating`:

- the seed-0 layout (layout 52, style 13), with the target replaced by `canned_food_18`;
- fixed fixtures, distractors, sink and external camera (`saga_external_v1`), 20 Hz control.

Each episode starts the robot far from the counter. The expert, or the policy, must:
1. navigate to an object-relative pregrasp pose;
2. grasp the can;
3. place it in the sink;
4. retract.

Success is RoboCasa's `_check_success()`. This is **not** a randomly sampled RoboCasa episode, and it
does not test generalization across kitchens.

| | |
|---|---|
| manifest | `manifests/counter_to_sink_v1.json` (100 episodes) |
| generation summary | `manifests/counter_to_sink_v1_generation.json` (every rejected candidate: episode, attempt, candidate seed, reason) |
| checksums | `manifests/counter_to_sink_v1.sha256`; verify with `sha256sum -c counter_to_sink_v1.sha256` from `manifests/` |
| manifest SHA-256 | `a6f728dd8c93933b4a63a61034d3a7c0ff8f5e0f2ae6841be156b6cf136320bf` |
| **generator commit** | `db1e63f`: the code that produced the manifest, recorded inside it as `code_commit` |
| **manifest / data-definition commit** | the commit that adds these files (`git log --format=%h -1 -- saga_robocasa/benchmark/manifests/counter_to_sink_v1.json`) |
| generated on | rlab7, AMD EPYC 7702, master seed 20261018, 5,062 s |

**Split, fixed before any collection.**

| split | episodes |
|---|---|
| training | `episode_000`–`episode_049` |
| held-out | `episode_050`–`episode_099` |

Held-out episodes are never used for any of the following:
- gradient updates;
- recovery demonstrations;
- normalization statistics;
- hyperparameter choice;
- early stopping;
- checkpoint selection.

Every collection and training run must verify the manifest checksum first. A frozen manifest is never edited; any change of distribution becomes a new version.

## Episode definition (`episodes.py`)

- **Object pose.** The can's XY is shifted from the reference pose, and the can is rotated about the vertical axis.
  - The can is axisymmetric: radius 2.67–2.75 cm, with its collision centre at the body origin.
  - The yaw rotation therefore changes its appearance, but not the top-down grasp.
- **Pregrasp goal.**
  - Position: the reference pregrasp pose, expressed relative to the object, (−0.543, +0.083) m in the goal-yaw frame, re-applied to the settled perturbed object.
  - Yaw: stays at the counter-facing reference yaw.
  - Arm ready pose, hover and grasp targets: recomputed from the object.
  - Transport, release and retract plans: computed online by the expert.
- **Start pose.** Generated relative to that goal, never relative to the robot's spawn pose:
  - distance `d`;
  - bearing `b` measured from "directly behind the goal";
  - yaw offset `ψ` from the goal yaw.
- **Rejection sampling.** For each episode index, attempts draw parameters from `default_rng(SeedSequence([20261018, index, attempt]))`. The first candidate that passes every initial-condition check is accepted. The checks are:
  - the object lies inside the counter-top region with at least 2 cm margin;
  - the object settles by ≤ 3 mm and tilts by ≤ 3°;
  - the object touches nothing but the counter top;
  - IK succeeds for the ready, hover and grasp poses, with joint margin ≥ 0.10 rad;
  - goal clearance is at least 0.05 m for the base and 0.02 m for the robot, with no contacts;
  - the start footprint is collision-free and inside the scene;
  - start clearance is at least 0.15 m for the base and 0.10 m for the robot;
  - an A* path exists.

  Nothing looks at expert or policy outcomes, or at the split.

## Proposal ranges vs realized benchmark support

The **proposal** distribution is uniform over:

| parameter | proposal range |
|---|---|
| start distance | 1.4–2.4 m |
| start bearing | ±1.2 rad |
| start yaw offset | ±π/2 |
| object x and y shift | ±3 cm each |
| object yaw | ±π |

The **benchmark distribution** is this proposal, conditioned on collision-free free space and valid initial conditions. Its realized support over the 100 accepted episodes:

| parameter | realized min | realized max | mean |
|---|---|---|---|
| start distance (m) | 1.402 | 2.133 | 1.642 |
| start bearing (rad) | +0.710 | +1.199 | +1.012 |
| start yaw offset (rad) | −1.519 | +1.563 | −0.028 |
| object dx (m) | −0.0285 | +0.0290 | +0.002 |
| object dy (m) | −0.0294 | +0.0297 | −0.001 |
| object yaw (rad) | −3.066 | +3.133 | +0.233 |

The training and held-out splits share the same support: distance 1.41–2.13 vs 1.40–2.07 m, and bearing 0.73–1.18 vs 0.71–1.20 rad.

**Rejection counts:** 1,670 candidates in total.
- 1,408 rejected by the fixture-footprint pre-check.
- 262 sent to the full checks: 162 rejected for start collision risk, 100 accepted.
- No other check rejected anything.
- Attempts per episode: median 13, maximum 67.

Acceptance by bearing (reconstructed from the candidate seeds):

| proposal bearing (rad) | candidates passing the footprint check | accepted |
|---|---|---|
| −1.2 to +0.6 | 0–3% | 0 |
| +0.6 to +0.9 | 72% | 22 |
| +0.9 to +1.2 | 49% | 78 |

Acceptance by distance, for bearing ≥ 0.6:

| distance (m) | accepted |
|---|---|
| 1.4–1.6 | 59% |
| 1.6–1.8 | 46% |
| 1.8–2.0 | 12% |
| 2.0–2.2 | 5% |
| 2.2–2.4 | 0% |

## What this benchmark does and does not measure

1. **One approach corridor.** The kitchen geometry restricts valid far starts to a single corridor on one side of the pregrasp pose: bearing ≈ +0.71 to +1.20 rad, distance ≈ 1.40–2.13 m.
2. **Straight-line paths.** Every planned A* path in the manifest has exactly two waypoints (start and goal), so the navigation is effectively straight-line driving with simultaneous yaw correction.
3. **What it measures:** whole-task performance (navigation, grasp, transport, placement and retraction) from varied start poses within that one corridor. The start yaw spans the full ±π/2, and the object pose spans the full ±3 cm and ±π.
4. **What it does not establish:** generalization across kitchen layouts, obstacle configurations, objects, or multiple approach corridors, and obstacle-avoiding path planning.
5. **The cheap pre-check does not change the accepted distribution.**
   - It applies the same fixture-footprint rule as the full check (`nav_expert.footprint_blocked` at the same transit clearance), at the goal implied by the commanded object pose.
   - It only avoids building the simulator state for candidates that this rule would reject anyway.
   - The settled and commanded object poses agree to within 0.0000 m (the maximum settling shift over the 100 episodes), and the full check rejected no candidate on footprint.

## Determinism and nodes

- `reset_to_spec(params)` reproduces each entry's `initial_state_sha256`.
- Bit-exact physics and IK hashes were verified on rlab3 and rlab4 (AMD EPYC 7352) and rlab7 (AMD EPYC 7702). They also match the earlier demonstrations recorded on ilab1 and rlab3.
- rlab2 produced different hashes, most likely from CPU-dependent BLAS kernels in the numpy IK. Benchmark jobs therefore run only on verified nodes, and every output records its host and CPU.

## Tools

| script | purpose |
|---|---|
| `make_manifest.py` | generate a new manifest version (refuses to overwrite an existing one) |
| `audit_expert.py` / `summarize_audit.py` | expert audit without saving images |
| `verify_rendering.py` | progress-camera rendering fix; policy-camera invariance and calibration |
| `compare_demos.py` | array-by-array comparison of two recorded demonstrations |
| `tests/test_benchmark.py` | backward compatibility (bit-exact old start paths), goal and start geometry, determinism, manifest split |

Generated audit output and trajectories live outside Git, under `/common/users/im484/saga_benchmark/`.

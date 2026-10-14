# DriveACT: architecture and implementation plan

This document records what was inspected before implementation, which parts of the references are genuinely ACT,
what was reused or adapted, and how the system is put together. Measured results live in `results/` and the README;
nothing here is a result.

## 1. Reference analysis

| Reference | Inspected | Genuinely ACT? | What this project takes from it |
|---|---|---|---|
| **ACT** (Zhao et al., RSS 2023), `tonyzhaozh/act` (MIT; `detr/` Apache-2.0) | paper; `detr/models/detr_vae.py`, `transformer.py`, `policy.py`, `utils.py`, `imitate_episodes.py` | yes, the reference | **Primary reference.** CVAE posterior over [CLS, proprioception, action chunk] with sinusoidal positions and padding mask; latent 32 via reparameterisation; zero latent at inference; DETR encoder-decoder with positions added to q/k in every layer, post-norm, zero decoder targets and learned queries; L1 + β·KL (KL summed over latent dims, averaged over the batch); mean/std normalisation; zero-padded chunks with `is_pad`; temporal ensembling with w_i = exp(−0.01 i), **oldest prediction weighted highest**. |
| **LeRobot ACT** (Apache-2.0) | `configuration_act.py`, `modeling_act.py` | yes (re-implementation) | Cross-check. Confirms that the reference decoder stacks all layer outputs and the policy reads index 0, so only the **first** of 7 decoder layers affects the actions (tonyzhaozh/act issue #25). |
| **QuantumTracer UGV** (no license found) | project page; linked repo `maxboels/QuantumTracer` | **no**: `SimpleACTModel` is a CNN + Transformer encoder + MLP that outputs a chunk but trains with MSE, has no CVAE / KL / queries, and deployment executes only the first action per frame; LeRobot-ACT configs in the repo appear unused | Only conventions: steering in [−1, 1], throttle in [0, 1] with no reverse or brake. No code used (no license). |
| **urban-vla-language-act** (Hugging Face, no license found) | model card, `config.json`, `src/urban_act/*` | ACT-like (CVAE + Transformer decoder, zero latent at inference), language-conditioned | Sanity check of driving-ACT settings: chunk 20 at 10 Hz, receding-horizon execution (replan every 3), state = speed, steering, previous commands. Language conditioning is out of scope. No code used. |
| **ITPS** (MIT) | `itps/common/policies/act/*`, `interact_maze2d.py` | yes (LeRobot ACT copy, `n_decoder_layers = 1`) | Confirms ACT works with low-dimensional observations for navigation (2-D maze, chunk 64, 10 Hz). Not reused: it samples z ~ N(0, I) at inference for steering; we keep ACT's z = 0. |

**What was adapted for Isaac Lab and driving:**

* **Images → LiDAR tokens.** ACT's ResNet-18 feature-map cells become 15 LiDAR tokens from a 1-D convolution
  (circular padding) over 120 range sectors, with sinusoidal azimuth positions (the analogue of ACT's 2-D sine
  positions).
* **Extra route token.** Heading error, lateral offset, 8 look-ahead route points, next-intersection manoeuvre and
  goal distance are encoded by an MLP into one more conditioning token (learned position, like ACT's latent and
  proprioception tokens).
* **Proprioception.** Speed, steering angle, yaw rate and the two previous commands play the role of ACT's joint
  positions, in both the CVAE encoder and the policy.
* **Decoder depth.** All decoder layers are used (`act.model.dec_layers`, default 4). `dec_layers: 1` reproduces the
  effective depth of the reference model.
* **L1 normalisation.** We divide by the number of valid (non-padded) entries (LeRobot), whereas the reference
  divides by all entries including padding.
* **Sampling.** Uniform over all timesteps of all training episodes (vectorised on device), instead of one random
  start per episode per epoch.
* **Size.** hidden 256, FFN 1024, 4/4/4 layers (≈ 10.7 M parameters), chunk 20 at 10 Hz, instead of 512 / 3200 /
  chunk 100 at 50 Hz for bimanual manipulation.

## 2. Installed environment (inspected)

* Isaac Sim 4.5.0 (pip), Isaac Lab 2.1.1, PyTorch 2.7.0 (CUDA 12.8), Python 3.10, NumPy 1.26, h5py, OpenCV,
  imageio, TensorBoard. Nothing was reinstalled.
* **Vehicle assets.** Isaac Sim 4.5 ships no full-size car. `Isaac/Robots/Leatherback/leatherback.usd` is an RC car
  (0.42 m × 0.28 m, 3.6 kg, full suspension linkage); `Isaac/Robots/Vehicle/basic_vehicle_m.usd` is a PhysX-Vehicle
  sample scene (vehicle wizard API, not an Isaac Lab articulation). Scaling an articulation ×10 is unreliable (mass,
  inertia, joint drives and contact offsets do not scale consistently). Therefore the car is a **procedural
  full-size articulation** authored with `pxr` (`driveact/vehicle/car_usd.py`).
* **Other assets.** Cones, barriers, trees and street lamps exist in the NVIDIA asset library but are large,
  materials-heavy and remote; all props are generated procedurally so the scene builds offline and quickly. Only the
  invisible ground plane and the HDRI sky are remote assets.

## 3. System overview

```
configs/*.yaml ──> TrackLayout (tracks/layout.py, generators.py)            pure NumPy, deterministic
                     │  route (right-lane centre), curbs, walls, surfaces, junctions
                     ├──> USD scene (environments/scene_builder.py) ──> Isaac Lab (isaac_backend.py)
                     │                                                    PhysX articulation car @ 100 Hz
                     ├──> obstacles (tracks/obstacles.py) ─ feasibility ─> expert planner (experts/planner.py)
                     └──> DriveEnv (environments/drive_env.py)            backend-agnostic, @ 10 Hz
                            LiDAR (sensors/raycast.py), route projection, observations, collisions, metrics
                                 │ obs                       ▲ action [steer, accel] in [-1, 1]
                                 ▼                           │
                     ScriptedExpert / ACT / BC controller (experts/, models/controller.py)
                                 │ (obs, expert action, executed action)
                                 ▼
                     HDF5 demos (datasets/hdf5_store.py) ─> training (training/trainer.py) ─> checkpoints
```

**Backends.** `IsaacBackend` (the real simulator) and `KinematicBackend` (a CPU test double: kinematic bicycle
with the same actuator model) implement the same 5-method interface (`set_scene`, `reset`, `step`, `state`,
`close`). Unit tests and pipeline dry runs use the test double on any machine; data collection, evaluation and all
reported results use Isaac Lab.

### Timing

| Quantity | Value |
|---|---|
| physics step | 0.01 s (100 Hz) |
| control / observation / LiDAR | 0.1 s (10 Hz), decimation 10 |
| ACT chunk | 20 actions = 2.0 s |
| replanning | every 2 control steps (0.2 s, configurable 1–5) or temporal ensembling every step |
| video | rendered every 4 physics steps = 25 fps (visual mode only) |

### Vehicle and control

* Footprint 4.40 m × 1.80 m, wheelbase 2.70 m, track 1.56 m, wheel radius 0.33 m, 1446 kg; chassis, two steering
  knuckles and four wheels (7 rigid bodies, 6 revolute joints). Cylinder tyre colliders, box body colliders.
* Action `[steering, longitudinal]` in [−1, 1]: steering × 35° (bicycle angle), longitudinal × 3 m/s² (≥ 0) or
  × 6 m/s² braking (< 0).
* Actuator model (`vehicle/control.py`, shared by both backends): steering slew limit 45°/s; acceleration
  integrated into a speed command clamped to [0, 14] m/s (no reverse) with ±1 m/s anti-windup; Ackermann split into
  left/right knuckle position targets; per-wheel speed targets from the turning radius (all-wheel drive, no scrub).
  PhysX implicit drives realise the targets; motion comes only from tyre forces.
* Minimum turning radius (rear axle) 3.86 m.

### Tracks (`configs/tracks/*.yaml`)

Road pieces: straights, circular curves, 4-way intersections (dead-end arms on unused directions). Cross-section:
two 3.5 m lanes (the route uses the right lane), 0.15 m curbs, 3 m sidewalks, 1.1 m concrete walls. Intersection
corners have 6 m curb radii; the route arcs concentrically (right turn 7.75 m, left 11.25 m). Every layout is
checked for overlaps between non-adjacent road corridors.

| Track | Geometry | Obstacles per episode |
|---|---|---|
| 1 | straights, two gentle curves, two 90° corners (≈ 330 m) | none |
| 2 | straights and two bends (≈ 255 m) | 1 parked car in the right lane |
| 3 | four bends incl. an 80° corner (≈ 460 m) | 1–3 parked cars, 1–2 cone tapers, 1–2 barrier groups, 0–1 car parked in the opposite lane |
| 4 | procedural city route, 3–4 intersections (≥ 1 left and ≥ 1 right turn), 1–3 curves (≈ 380–800 m) | 1–3 parked cars, 1–2 cone tapers, 1–2 barrier groups, 0–1 opposite-lane car |

Layout seeds: Tracks 1–3 seed 0 is the canonical recipe, other seeds jitter lengths/radii/angles and may mirror.
Track 4 train seeds 0–11 and test seeds 1000–1005 plus the hand-designed "showcase" (−1) are disjoint. Obstacle
placements come from two disjoint random streams (collection vs evaluation).

**Feasibility.** An obstacle configuration is accepted only if the expert's lateral planner finds a C1-smooth
offset profile that clears every obstacle (≥ 0.45 m) and both curbs, with lane changes that do not start inside
an intersection or before the start, and do not end after the goal.

### Sensing and observations (`environments/observations.py`, 150-D)

* **LiDAR**: 720 exact rays (0.5°) cast analytically at 0.5 m height against wall segments, obstacle boxes and cone
  circles (the same geometry that is spawned in USD), min-pooled to 120 sectors, max range 40 m.
* **Vehicle state** (5): speed, measured steering angle, yaw rate, previous steering and longitudinal commands.
* **Route** (25): heading error (sin, cos), lateral offset from the right-lane centre, 8 look-ahead route points
  (4–40 m) in the vehicle frame, next intersection manoeuvre one-hot (left/straight/right/none) within 60 m and its
  distance, remaining distance to the goal.
* Not observed: obstacle positions, the expert's planned offsets or speeds, future actions.

### Expert (`experts/`)

Privileged: lateral offset profile around obstacles (pass in the left lane, cosine lane changes over 20 m,
merged passes), speed profile (curvature limit 2.2 m/s², 6.5 m/s while shifted, stop at the goal, forward/backward
passes), pure pursuit from the rear axle (look-ahead 0.7 s · v in [5, 14] m), P speed control. 50 % of collected
episodes add Ornstein–Uhlenbeck noise to the executed action (DART); labels are always the clean expert action.

### Dataset (`datasets/hdf5_store.py`)

One HDF5 file per collection run, one group per successful episode: `obs (T, 150)`, `actions (T, 2)` (labels),
`executed (T, 2)`, `state (T, 7)`, `time (T)`, plus outcome, layout seed, obstacle JSON and the full configuration.
Alignment `obs[t+1].previous_command == executed[t]` is unit-tested.

### Evaluation (`evaluation/`)

Goal completion (reached the goal, stopped, no collision and no curb strike), collision rate by type (wall,
vehicle, cone, barrier), curb strikes, lane departure (car centre > 1.75 m from the expert's reference offset),
obstacle avoidance (groups passed / groups encountered), route tracking error, travel time, mean speed, steering /
acceleration smoothness (mean |Δ command| per step, steering jerk), minimum body clearance (from the LiDAR), and
inference latency. Rates carry 95 % Wilson intervals.

## 4. Stages and status

| Stage | Content | Status |
|---|---|---|
| 1 | Isaac Lab, vehicle physics, simple road | done: all `scripts/test_vehicle.py` checks pass (`results/stage1/`) |
| 2 | scripted expert on Tracks 1–4 | done: 64/64 on training layouts, 48/48 on each held-out test set |
| 3 | demonstrations, ACT implementation and unit tests | done: 1,083 episodes / 691k timesteps (`results/data/`) |
| 4–6 | curriculum: Track 1 → 1–2 → 1–3 | done: 48/48, 33/48, 32/48 on held-out layouts of the newest track |
| 5 (diagnosis) | copycat failure of the first ACT | found and fixed with previous-command dropout/noise (`results/ablation_copycat/`) |
| 7 | final ACT vs. BC vs. expert, temporal ensembling, no-previous-command ablation | done (`results/stage7/`) |
| 8 | demonstration videos | rendered in Isaac Sim (`videos/`) |

**Geometry placement rule.** Per-environment static geometry is authored in world coordinates under
`/World/Tracks/env_i` (never under the cloned `/World/envs/env_i`, which carries the grid translation); each layout's
bounding box is centred on its environment grid point; start-up checks verify that tracks do not overlap and that
each environment's wall mesh coincides with the analytic walls.

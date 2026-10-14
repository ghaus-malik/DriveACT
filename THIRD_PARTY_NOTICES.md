# Third-party notices

All source code in this repository was written for this project. No code was copied from the reference
repositories listed below; the ACT model follows the architecture described in the ACT paper and in its reference
implementation, which were studied but not copied. The project depends on the following third-party software and
remote assets, which are installed or downloaded separately and are not redistributed here.

## Software

| Component | Use in this project | License |
|---|---|---|
| NVIDIA Isaac Sim 4.5 | physics simulation (PhysX) and RTX rendering | NVIDIA Isaac Sim additional software and materials license |
| Isaac Lab 2.1 (`isaaclab`) | scene, articulation, rigid-object, contact-sensor, camera and marker APIs | BSD-3-Clause |
| PyTorch | neural networks, batched geometry | BSD-3-Clause |
| NumPy, PyYAML, h5py | numerics, configuration, demonstration storage | BSD-3-Clause; MIT; BSD-3-Clause |
| OpenUSD (`pxr`, bundled with Isaac Sim or `usd-core`) | procedural car, obstacle and track geometry | Tomorrow Open Source Technology License (Apache-2.0 based) |
| Matplotlib | top-down plots | Matplotlib license (PSF-based) |
| OpenCV (`opencv-python`), imageio, imageio-ffmpeg (FFmpeg) | video overlays and MP4 writing | Apache-2.0; BSD-2-Clause; FFmpeg under LGPL/GPL depending on the build |
| TensorBoard (optional) | training curves | Apache-2.0 |

## Remote assets

| Asset | Use | Source |
|---|---|---|
| `Isaac/Environments/Grid/default_environment.usd` | ground plane (invisible collider) | Isaac Sim 4.5 asset library |
| `Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr` | sky light in the visual mode | Isaac Sim 4.5 asset library (Poly Haven, CC0) |

The car, parked cars, traffic cones, concrete barriers, roads, sidewalks, walls, buildings, trees and street lamps
are generated procedurally by this project (`driveact/vehicle/car_usd.py`, `driveact/environments/scene_builder.py`).

## References (studied, not copied)

| Reference | What was used | License |
|---|---|---|
| T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023 ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)), code [github.com/tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT architecture: CVAE encoder, DETR-style Transformer, action chunking, temporal ensembling, loss | MIT (c) 2023 Tony Z. Zhao; `detr/` subfolder Apache-2.0 (c) Facebook, Inc. |
| LeRobot ACT ([github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)) | cross-check of the reference implementation (decoder-layer note, temporal ensembler) | Apache-2.0 |
| QuantumTracer UGV ([maxboels.com/projects/quantumtracer-ugv](https://maxboels.com/projects/quantumtracer-ugv)) | control conventions of an RC car driven by a learned policy (discussion only) | no license found; not used |
| Mayank022/urban-vla-language-act (Hugging Face) | configuration of a driving ACT variant (chunk 20 at 10 Hz, receding horizon; discussion only) | no license found; not used |
| ITPS ([arXiv:2411.16627](https://arxiv.org/abs/2411.16627), [github.com/yanweiw/itps](https://github.com/yanweiw/itps)) | ACT for 2-D maze navigation (discussion only) | MIT (c) 2024 Yanwei Wang |
| L. Laskey et al. *DART: Noise Injection for Robust Imitation Learning.* CoRL 2017 | disturbance injection during data collection | paper |

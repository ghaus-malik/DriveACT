"""Run episodes of any controller (expert, ACT, BC) in a :class:`DriveEnv`, with optional recording and noise."""

from __future__ import annotations

import time

import numpy as np
import torch


class OUNoise:
    """DART-style disturbance: Ornstein-Uhlenbeck noise added to the *executed* action of noisy episodes only.

    The recorded label stays the clean expert action, so the dataset contains recoveries from perturbed states.
    """

    def __init__(self, ncfg, num_envs: int, device, seed: int = 0):
        self.sigma = torch.tensor([float(ncfg.steer_sigma), float(ncfg.accel_sigma)], device=device)
        self.theta = torch.tensor([float(ncfg.steer_theta), float(ncfg.accel_theta)], device=device)
        self.z = torch.zeros(num_envs, 2, device=device)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def reset(self, env_ids) -> None:
        self.z[env_ids] = 0.0

    def __call__(self, actions: torch.Tensor, env) -> torch.Tensor:
        eps = torch.randn(self.z.shape, generator=self.gen).to(self.z.device)
        self.z = self.z - self.theta * self.z + self.sigma * eps
        noisy = env.noisy[:, None]
        return torch.where(noisy, (actions + self.z).clamp(-1.0, 1.0), actions)


def run_episodes(env, controller, episodes: list[int], noise: OUNoise | None = None, noisy_flags: dict | None = None,
                 recorder=None, trace: bool = False, on_step=None, progress: bool = False) -> list:
    """Run the given episode ids, ``env.N`` at a time. Returns :class:`EpisodeRecord` s in completion order.

    ``noisy_flags`` maps episode id -> bool (disturbed episode). ``trace`` stores per-episode x/y/yaw/speed arrays
    in ``record.extra["trace"]``. ``on_step(env, obs, actions)`` is called every control step (visualisation).
    """
    queue = list(episodes)
    records = []
    traces: dict[int, list] = {}
    latency = []
    first = queue[: env.N]
    queue = queue[env.N :]
    slots = list(range(len(first)))
    idle = torch.ones(env.N, dtype=torch.bool, device=env.device)
    idle[: len(first)] = False
    if first:
        flags = [bool((noisy_flags or {}).get(e, False)) for e in first]
        env.reset(slots, first, noisy=flags)
        controller.reset(torch.tensor(slots, device=env.device))
        if noise is not None:
            noise.reset(torch.tensor(slots, device=env.device))
    # inactive environments (fewer episodes than envs) are parked as finished
    if len(first) < env.N:
        env.outcome[len(first):] = 1
    t0 = time.time()
    done_count = 0
    while not bool(idle.all()):
        obs = env.observe()
        tic = time.perf_counter()
        labels = controller.act(obs, env)
        if env.device.type == "cuda":
            torch.cuda.synchronize(env.device)
        latency.append((time.perf_counter() - tic) / max(1, int((~idle).sum())))
        executed = noise(labels, env) if noise is not None else labels
        if recorder is not None:
            recorder.add_step(obs, labels, executed, env, ~idle)
        if trace:
            st = env.state
            for i in torch.nonzero(~idle).flatten().tolist():
                traces.setdefault(i, []).append(
                    [float(st["pos"][i, 0]), float(st["pos"][i, 1]), float(st["yaw"][i]), float(st["v_long"][i])]
                )
        info = env.step(executed)
        if on_step is not None:
            on_step(env, obs, executed)
        finished = torch.nonzero(info["newly_done"] & ~idle).flatten().tolist()
        for i in finished:
            rec = env.episode_record(i)
            if trace:
                rec.extra["trace"] = np.asarray(traces.pop(i, []), dtype=np.float32)
            if recorder is not None:
                recorder.finish_episode(i, rec)
            records.append(rec)
            done_count += 1
            if progress:
                print(f"  [{done_count}/{len(episodes)}] env {i} episode {rec.episode}: {rec.outcome} "
                      f"({rec.time_s:.1f} s, progress {rec.progress:.2f})", flush=True)
            if queue:
                ep = queue.pop(0)
                env.reset([i], [ep], noisy=[bool((noisy_flags or {}).get(ep, False))])
                controller.reset(torch.tensor([i], device=env.device))
                if noise is not None:
                    noise.reset(torch.tensor([i], device=env.device))
            else:
                idle[i] = True
    if records:
        lat = float(np.mean(latency)) if latency else float("nan")
        for r in records:
            r.extra["latency_ms_per_env"] = 1000.0 * lat
            r.extra["wall_time_s"] = time.time() - t0
    return records

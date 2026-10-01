"""Deterministic checkpoint A/B evaluation; never writes deployment models."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from export_rl_mpc_policy import load_actor  # noqa: E402
from validate_rl_mpc import (  # noqa: E402
  _place_all_terrain_types_at_level,
  _set_exact_manual_command,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import (  # noqa: E402
  syncai_g23_pure_mpc_validation_env_cfg,
)

parser = argparse.ArgumentParser()
parser.add_argument(
  "--models", nargs="+", default=["pure_mpc", "700", "2400", "4000", "5399"]
)
parser.add_argument("--levels", nargs="+", type=int, default=[0])
parser.add_argument("--seeds", nargs="+", type=int, default=[42])
parser.add_argument("--output", default="screening.json")
parser.add_argument("--terrain-bank", action="store_true")
parser.add_argument("--positive-axes-only", action="store_true")
args = parser.parse_args()
torch.set_num_threads(1)
run = ROOT / "logs/rsl_rl/g23_rl_mpc_foot_state_history_v3/2026-09-10_11-11-39"
cfg = syncai_g23_pure_mpc_validation_env_cfg()
cfg.scene.num_envs = 3 if args.terrain_bank else 1
cfg.episode_length_s = 20.0
cfg.auto_reset = False
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
results = []
commands = [
  (0.2, 0, 0),
  (-0.2, 0, 0),
  (0, 0.1, 0),
  (0, -0.1, 0),
  (0, 0, 0.3),
  (0, 0, -0.3),
  (0.2, 0, 0.3),
]
if args.positive_axes_only:
  commands = [(0.2, 0, 0), (0, 0.1, 0), (0, 0, 0.3)]
try:
  for name in args.models:
    checkpoint = None if name == "pure_mpc" else run / f"model_{name}.pt"
    actor = (
      None
      if checkpoint is None
      else load_actor(checkpoint)[0].as_onnx(verbose=False).eval()
    )
    digest = (
      None
      if checkpoint is None
      else hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    for level in args.levels:
      for seed in args.seeds:
        for command in commands:
          _place_all_terrain_types_at_level(env, level)
          env.reset(seed=seed)
          term = env.action_manager.get_term("foot_placement")
          robot = env.scene["robot"]
          cmds = env.command_manager.get_term("twist")
          phases = []
          failed = False
          max_roll_pitch = np.zeros(env.num_envs)
          for phase, duration, target in [
            ("stand", 2, (0, 0, 0)),
            ("walk", 4, command),
            ("stop", 3, (0, 0, 0)),
          ]:
            for e in range(env.num_envs):
              _set_exact_manual_command(cmds, e, target)
            records = []
            for _tick in range(round(duration / env.step_dt)):
              obs = term.state_history.clone()
              yaw = obs[:, 2]
              obs[:, 61] = yaw.cos() * target[0] - yaw.sin() * target[1]
              obs[:, 62] = yaw.sin() * target[0] + yaw.cos() * target[1]
              obs[:, 63] = target[2]
              term._observation_history.frames[:, 0, 61:64] = obs[:, 61:64]
              with torch.inference_mode():
                raw = torch.zeros((env.num_envs, 8)) if actor is None else actor(obs)
              if not bool(torch.isfinite(raw).all()):
                raise RuntimeError(f"nonfinite action: {name}")
              _, _, done, timeout, _ = env.step(raw.clamp(-1, 1))
              data = robot.data
              max_roll_pitch = np.maximum(
                max_roll_pitch, term.state_history[:, :2].abs().max(1).values.numpy()
              )
              contact = (
                env.scene["feet_ground_contact"]
                .data.found.reshape(env.num_envs, 4, -1)
                .gt(0)
                .any(-1)
              )
              records.append(
                {
                  "velocity": torch.cat(
                    (data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2:3]),
                    1,
                  ).tolist(),
                  "pose_mse": (data.joint_pos - data.default_joint_pos)
                  .square()
                  .mean(-1)
                  .tolist(),
                  "clip": (raw.abs() > 1).float().mean(-1).tolist(),
                  "contact": contact.all(-1).float().tolist(),
                  "modes": term.gait_modes.tolist(),
                }
              )
              if bool(done.any() or timeout.any()):
                failed = True
                break
            tail = records[len(records) // 2 :]
            phases.append(
              {
                "phase": phase,
                "seconds": len(records) * env.step_dt,
                "tracking_mae": np.abs(
                  np.array([r["velocity"] for r in tail]) - np.array(target)
                )
                .mean(0)
                .tolist(),
                "pose_rmse_rad": np.sqrt(
                  np.array([r["pose_mse"] for r in tail]).mean(0)
                ).tolist(),
                "clip_fraction": np.array([r["clip"] for r in tail]).mean(0).tolist(),
                "four_contact_fraction": np.array([r["contact"] for r in tail])
                .mean(0)
                .tolist(),
                "stand_fraction": (np.array([r["modes"] for r in tail]) == 2)
                .mean(0)
                .tolist(),
                "terminated": done.tolist(),
                "timeout": timeout.tolist(),
              }
            )
            if failed:
              break
          row = {
            "model": name,
            "checkpoint_sha256": digest,
            "level": level,
            "seed": seed,
            "command": command,
            "terrain_types": env.scene.terrain.terrain_types.tolist(),
            "failed": failed,
            "max_roll_pitch_rad": max_roll_pitch.tolist(),
            "solver_failure_count": [
              c.mpc_solver_failure_count for c in term._controllers
            ],
            "phases": phases,
          }
          results.append(row)
          (Path(__file__).parent / args.output).write_text(
            json.dumps(results, indent=2)
          )
          print(
            f"{name} level={level} seed={seed} command={command} failed={failed}",
            flush=True,
          )
finally:
  env.close()

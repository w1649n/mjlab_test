"""Score checkpoints' behavior with the StairGoals training reward, term by term.

Each env climbs or descends a straight 20 cm flight, or walks and stops on flat
ground. Per-term reward rates from the reward manager are integrated over a fixed
window (until termination, as an episode would end), so behaviors from different
checkpoints can be compared under the reward that training optimizes.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv

TASK = "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-J60-StairGoals"
PER_COLUMN = 128
WINDOW = (1.0, 21.0)
STOP_AT = 5.0
STOP_WINDOW = (6.0, 16.0)
STOP_ON_STAIRS = False


def build(height: float, device: str):
  cfg = load_env_cfg(TASK, play=True)
  stair_kw = dict(
    proportion=1.0,
    step_height_range=(height, height),
    step_width=0.30,
    platform_width=2.0,
    border_width=1.0,
  )
  assert cfg.scene.terrain is not None
  # Sub-terrain names match training so terrain-aware rewards stay active.
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=20.0,
    num_rows=1,
    num_cols=3,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=1.0),
      "pyramid_stairs_inv": pyramid_stairs_inv(**stair_kw),
      "pyramid_stairs": pyramid_stairs(**stair_kw),
    },
  )
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.scene.num_envs = 3 * PER_COLUMN
  cfg.events.pop("randomize_terrain", None)
  base = cfg.events["reset_base"]
  base.func = mdp.reset_root_state_uniform
  base.params = {
    "pose_range": {
      "x": (-0.1, 0.1),
      "y": (-0.3, 0.3),
      "z": (0.01, 0.03),
      "yaw": (-math.radians(3), math.radians(3)),
    },
    "velocity_range": {},
  }
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def conditions():
  conds = []
  for col in ("flat", "up", "down"):
    for k in range(PER_COLUMN):
      if col == "flat":
        conds.append(
          {"name": "flat_stop", "sched": [(0.0, 0.0), (1.0, 0.6), (STOP_AT, 0.0)]}
        )
      else:
        if STOP_ON_STAIRS:
          sched = [(0.0, 0.0), (1.0, 0.5), (STOP_AT, 0.0)]
          conds.append({"name": f"{col}_stop", "sched": sched})
          continue
        vx = 0.3 if k < PER_COLUMN // 2 else 0.8
        conds.append({"name": f"{col}_v{vx}", "sched": [(0.0, 0.0), (1.0, vx)]})
  return conds


def rollout(env, vec, runner, checkpoint: str, noise: bool, device: str) -> dict:
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  std = (
    torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"][
      "distribution.std_param"
    ]
    .float()
    .mean()
    .item()
  )
  conds = conditions()
  cmd_term = env.command_manager.get_term("twist")
  rm = env.reward_manager
  names = list(rm.active_terms)
  dt = env.step_dt
  steps = round(WINDOW[1] / dt)
  changes: dict[int, list] = {}
  for i, c in enumerate(conds):
    for t0, vx in c["sched"]:
      changes.setdefault(round(t0 / dt), []).append((i, (vx, 0.0, 0.0)))
  torch.manual_seed(0)
  vec.reset()
  n = env.num_envs
  alive = torch.ones(n, dtype=torch.bool, device=device)
  window_sum = torch.zeros(n, len(names), device=device)
  stop_sum = torch.zeros(n, len(names), device=device)
  start_x = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
  max_x = torch.zeros(n, device=device)
  hipx = torch.zeros(n, device=device)
  hipx_count = 0
  robot = env.scene["robot"]
  sign = torch.tensor(
    [1.0 if nm.startswith(("FL", "HL")) else -1.0 for nm in robot.joint_names],
    device=device,
  )
  hipx_ids = [i for i, nm in enumerate(robot.joint_names) if "HipX" in nm]
  origins = env.scene.env_origins[:, 0].clone()
  obs = vec.get_observations()
  for t in range(steps):
    for i, cmd in changes.get(t, []):
      cmd_term.set_manual_command(i, cmd)
    with torch.inference_mode():
      action = policy(obs)
      if noise:
        action = action + std * torch.randn_like(action)
    obs, _, dones, _ = vec.step(action)
    rate = rm._step_reward.clone()
    time = t * dt
    if WINDOW[0] <= time < WINDOW[1]:
      window_sum += rate * dt * alive[:, None]
    if STOP_WINDOW[0] <= time < STOP_WINDOW[1]:
      stop_sum += rate * dt * alive[:, None]
      err = (robot.data.joint_pos - robot.data.default_joint_pos) * sign
      hipx += err[:, hipx_ids].mean(1)
      hipx_count += 1
    x = env.scene["robot"].data.root_link_pos_w[:, 0] - origins
    max_x = torch.where(alive, torch.maximum(max_x, x), max_x)
    alive &= ~dones.bool()
  del start_x
  out = {"std": std, "groups": {}}
  for gname in sorted({c["name"] for c in conds}):
    ids = torch.tensor(
      [i for i, c in enumerate(conds) if c["name"] == gname], device=device
    )
    source = stop_sum if gname.endswith("_stop") else window_sum
    per_term = source[ids].mean(0)
    out["groups"][gname] = {
      "total": float(per_term.sum()),
      "terms": {nm: float(v) for nm, v in zip(names, per_term.tolist(), strict=True)},
      "survived": float(alive[ids].float().mean()),
      "max_x_m": float(max_x[ids].mean()),
      "reached_top": float((max_x[ids] > 5.3).float().mean()),
      "stop_hipx_outward_deg": float(
        torch.rad2deg(hipx[ids] / max(hipx_count, 1)).mean()
      ),
    }
  return out


def main():
  global STOP_ON_STAIRS, STOP_AT, STOP_WINDOW
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoints", nargs="+", required=True)
  parser.add_argument("--height", type=float, default=0.20)
  parser.add_argument("--device", default="cuda:1")
  parser.add_argument("--output", required=True)
  parser.add_argument("--stop-on-stairs", action="store_true")
  parser.add_argument("--stop-at", type=float, default=STOP_AT)
  args = parser.parse_args()
  STOP_ON_STAIRS = args.stop_on_stairs
  STOP_AT = args.stop_at
  STOP_WINDOW = (STOP_AT + 1.0, STOP_AT + 11.0)
  env = build(args.height, args.device)
  agent = asdict(load_rl_cfg(TASK))
  agent["algorithm"]["symmetry_cfg"] = None
  vec = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None
  runner = runner_cls(vec, agent, device=args.device)
  results = {}
  for ck in args.checkpoints:
    for noise in (False, True):
      key = f"{Path(ck).parent.name}/{Path(ck).name}|{'noise' if noise else 'det'}"
      results[key] = rollout(env, vec, runner, ck, noise, args.device)
      print("done", key, flush=True)
  Path(args.output).write_text(json.dumps(results, indent=1))
  env.close()


if __name__ == "__main__":
  main()

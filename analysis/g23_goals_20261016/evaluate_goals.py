"""Evaluate a G23 History6 velocity checkpoint against the flat/stair goals.

All conditions run in one batched rollout. Every env walks along world +x from
its sub-terrain center; columns hold flat ground or straight pyramid flights with
a fixed riser height. Writes per-env metrics and a summary JSON.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.asset_zoo.robots import G23_J60_RATED_TORQUE, get_g23_j60_robot_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.config.g23.env_cfgs import (
  syncai_g23_rough_proprio_history6_env_cfg,
)
from mjlab.terrains import TerrainGeneratorCfg
from mjlab.terrains.config import flat, pyramid_stairs, pyramid_stairs_inv
from mjlab.utils.lab_api.string import resolve_matching_names_values

TASK_ID = "Mjlab-Velocity-Rough-SyncAI-G23-Proprio-History6-StairYaw"
TILE = 12.0
PLATFORM = 2.0
TILE_BORDER = 1.0
STEP_WIDTH = 0.30
STAIR_START = PLATFORM / 2  # Distance from tile center to the first riser.
STAIR_END = TILE / 2 - TILE_BORDER  # Distance to the outer edge of the flight.
GOAL_X = STAIR_END + 0.5
HEIGHTS = (0.05, 0.10, 0.15, 0.18, 0.20, 0.22)
STAIR_SPEEDS = (0.3, 0.5, 0.8, 1.0)
FLAT_COMMANDS = (
  (0.3, 0.0, 0.0),
  (0.6, 0.0, 0.0),
  (1.0, 0.0, 0.0),
  (-0.5, 0.0, 0.0),
  (0.0, 0.2, 0.0),
  (0.0, 0.0, 0.5),
  (0.6, 0.0, 0.4),
  (0.6, 0.15, 0.0),
)
PER_COLUMN = 64
FLAT_COLUMNS = ("flat_a", "flat_b")
DURATION = 35.0
T_START = 1.0
FLAT_WALK_END = 9.0
STAIR_WALK_END = 33.0
J60_PEAK = {r".*_Hip[XY]_joint": 19.94, r".*_Knee_joint": 30.5}


def build_env_cfg(actuator: str, domain_rand: bool):
  cfg = syncai_g23_rough_proprio_history6_env_cfg(play=True)
  if actuator == "j60":
    cfg.scene.entities = {"robot": get_g23_j60_robot_cfg()}
  sub_terrains = {name: flat(proportion=1.0) for name in FLAT_COLUMNS}
  stair_kw = dict(
    proportion=1.0,
    step_width=STEP_WIDTH,
    platform_width=PLATFORM,
    border_width=TILE_BORDER,
  )
  for h in HEIGHTS:
    tag = f"{round(h * 100)}"
    sub_terrains[f"up_{tag}"] = pyramid_stairs_inv(step_height_range=(h, h), **stair_kw)
    sub_terrains[f"down_{tag}"] = pyramid_stairs(step_height_range=(h, h), **stair_kw)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(TILE, TILE),
    border_width=20.0,
    num_rows=1,
    num_cols=len(sub_terrains),
    curriculum=True,
    sub_terrains=sub_terrains,
  )
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.scene.num_envs = PER_COLUMN * len(sub_terrains)
  cfg.events.pop("randomize_terrain", None)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.3, 0.3),
    "z": (0.01, 0.03),
    "yaw": (-math.radians(3.0), math.radians(3.0)),
  }
  if not domain_rand:
    for name in (
      "foot_friction_slide",
      "foot_friction_spin",
      "foot_friction_roll",
      "encoder_bias",
      "base_com",
    ):
      cfg.events.pop(name, None)
  return cfg, list(sub_terrains)


def build_conditions(columns: list[str]) -> list[dict]:
  conds = []
  for col in columns:
    for k in range(PER_COLUMN):
      if col.startswith("flat"):
        offset = 0 if col == "flat_a" else PER_COLUMN // 2
        cmd = FLAT_COMMANDS[(k + offset) % len(FLAT_COMMANDS)]
        sched = [(0.0, (0.0, 0.0, 0.0)), (T_START, cmd), (FLAT_WALK_END, (0, 0, 0))]
        conds.append({"column": col, "kind": "flat", "cmd": cmd, "schedule": sched})
      else:
        kind, tag = col.split("_")
        vx = STAIR_SPEEDS[k % len(STAIR_SPEEDS)]
        cmd = (vx, 0.0, 0.0)
        sched = [(0.0, (0.0, 0.0, 0.0)), (T_START, cmd), (STAIR_WALK_END, (0, 0, 0))]
        conds.append(
          {
            "column": col,
            "kind": kind,
            "height": int(tag) / 100,
            "cmd": cmd,
            "schedule": sched,
          }
        )
  return conds


def rollout(args) -> dict:
  device = args.device
  env_cfg, columns = build_env_cfg(args.actuator, not args.no_dr)
  conds = build_conditions(columns)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  agent_cfg = load_rl_cfg(TASK_ID)
  vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK_ID)(vec, asdict(agent_cfg), device=device)
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  terrain = env.scene.terrain
  assert terrain is not None
  expected = torch.arange(env.num_envs, device=device) // PER_COLUMN
  assert torch.equal(terrain.terrain_types, expected), "unexpected env layout"

  robot = env.scene["robot"]
  cmd_term = env.command_manager.get_term("twist")
  contact = env.scene["feet_ground_contact"]
  height_scan = env.scene["base_height_scan"]
  tm = env.termination_manager
  fail_terms = [n for n in tm.active_terms if n != "time_out"]

  vec.reset()
  steps = round(DURATION / env.step_dt)
  changes: dict[int, list[tuple[int, tuple]]] = {}
  for i, c in enumerate(conds):
    for t0, cmd in c["schedule"]:
      changes.setdefault(round(t0 / env.step_dt), []).append((i, cmd))

  n = env.num_envs
  origins = env.scene.env_origins.clone()
  buf = {
    "pos": torch.zeros(steps, n, 3, device=device),
    "heading": torch.zeros(steps, n, device=device),
    "grav": torch.zeros(steps, n, 3, device=device),
    "jerr": torch.zeros(steps, n, 12, device=device),
    "tau": torch.zeros(steps, n, 12, device=device),
    "contact": torch.zeros(steps, n, 4, dtype=torch.bool, device=device),
    "vel": torch.zeros(steps, n, 3, device=device),
    "base_h": torch.zeros(steps, n, device=device),
    "cmd": torch.zeros(steps, n, 3, device=device),
  }
  first_fail = torch.full((n,), steps, dtype=torch.long, device=device)
  fail_reason = torch.full((n,), -1, dtype=torch.long, device=device)
  # Optional deployment-side heading hold: straight commands get a yaw-rate
  # correction toward the heading captured when the segment started.
  hold = torch.zeros(n, dtype=torch.bool, device=device)
  ref_heading = torch.zeros(n, device=device)
  manual_cmd = cmd_term._manual_command_b
  obs = vec.get_observations()
  for t in range(steps):
    for i, cmd in changes.get(t, []):
      cmd_term.set_manual_command(i, cmd)
      hold[i] = args.heading_hold > 0 and cmd[2] == 0 and (cmd[0] or cmd[1])
      ref_heading[i] = robot.data.heading_w[i]
    if t in changes:
      obs = vec.get_observations()
    if args.heading_hold > 0:
      err = torch.remainder(ref_heading - robot.data.heading_w + math.pi, 2 * math.pi)
      wz = (args.heading_hold * (err - math.pi)).clamp(-0.5, 0.5)
      manual_cmd[hold, 2] = wz[hold]
    with torch.inference_mode():
      actions = policy(obs)
    # Record the pre-step command, then the post-step state.
    buf["cmd"][t] = cmd_term.command[:, :3]
    obs, _, _, _ = vec.step(actions)
    d = robot.data
    buf["pos"][t] = d.root_link_pos_w - origins
    buf["heading"][t] = d.heading_w
    buf["grav"][t] = d.projected_gravity_b
    buf["jerr"][t] = d.joint_pos - d.default_joint_pos
    buf["tau"][t] = d.actuator_force
    buf["contact"][t] = contact.data.found.reshape(n, 4, -1).gt(0).any(-1)
    buf["vel"][t, :, :2] = d.root_link_lin_vel_b[:, :2]
    buf["vel"][t, :, 2] = d.root_link_ang_vel_b[:, 2]
    h = height_scan.data.heights
    buf["base_h"][t] = h.reshape(n, -1).mean(-1)
    for k, name in enumerate(fail_terms):
      hit = tm.get_term(name) & (first_fail == steps)
      first_fail[hit] = t
      fail_reason[hit] = k
  # Tilt beyond 60 degrees or a collapsed torso also counts as a fall.
  tilt_fall = buf["grav"][..., 2] > -0.5
  low_fall = buf["base_h"] < 0.12
  fell = tilt_fall | low_fall
  t_idx = torch.arange(steps, device=device)[:, None].expand(steps, n)
  fall_t = torch.where(fell, t_idx, steps).min(0).values
  fall_new = fall_t < first_fail
  fail_reason[fall_new] = len(fail_terms)
  first_fail = torch.minimum(first_fail, fall_t)

  actuator_names = list(robot.actuator_names)
  joint_names = list(robot.joint_names)
  rated = np.zeros(len(actuator_names))
  peak = np.zeros(len(actuator_names))
  for table, out in ((G23_J60_RATED_TORQUE, rated), (J60_PEAK, peak)):
    ids, _, values = resolve_matching_names_values(table, actuator_names)
    out[ids] = values
  data = {k: v.cpu().numpy() for k, v in buf.items()}
  env.close()
  return {
    "conds": conds,
    "data": data,
    "first_fail": first_fail.cpu().numpy(),
    "fail_reason": fail_reason.cpu().numpy(),
    "fail_names": fail_terms + ["fall_tilt_or_height"],
    "actuator_names": actuator_names,
    "joint_names": joint_names,
    "rated": rated,
    "peak": peak,
    "dt": env.step_dt,
  }


def wrap(a):
  return (a + np.pi) % (2 * np.pi) - np.pi


def analyze(r: dict) -> list[dict]:
  d, dt = r["data"], r["dt"]
  steps = d["pos"].shape[0]
  rated, peak = r["rated"], r["peak"]
  i_start = round(T_START / dt)
  rows = []
  for i, c in enumerate(r["conds"]):
    ff = int(r["first_fail"][i])
    reason = r["fail_names"][r["fail_reason"][i]] if ff < steps else None
    pos = d["pos"][:, i]
    heading = d["heading"][:, i]
    drift = np.degrees(wrap(heading - heading[i_start]))
    grav = d["grav"][:, i]
    nose_up = np.degrees(np.arcsin(np.clip(-grav[:, 0], -1, 1)))
    roll = np.degrees(np.arcsin(np.clip(grav[:, 1], -1, 1)))
    tau = d["tau"][:, i]
    row = {
      "env": i,
      "column": c["column"],
      "kind": c["kind"],
      "cmd": c["cmd"],
      "fail_step_s": None if ff >= steps else ff * dt,
      "fail_reason": reason,
    }
    if c["kind"] == "flat":
      i_end = round(FLAT_WALK_END / dt)
      w0 = round((T_START + 2.0) / dt)
      walk_ok = ff >= i_end
      row["walk_survived"] = bool(walk_ok)
      row["survived"] = bool(ff >= steps)
      if walk_ok:
        err = np.abs(d["vel"][w0:i_end, i] - np.array(c["cmd"])).mean(0)
        row["track_mae"] = err.tolist()
        row["heading_drift_max_deg"] = float(np.abs(drift[i_start:i_end]).max())
        row["heading_drift_end_deg"] = float(drift[i_end - 1])
        row["walk_dist_m"] = float(
          np.linalg.norm(pos[i_end - 1, :2] - pos[i_start, :2])
        )
        seg = slice(w0, i_end)
        row["walk_tau_rms_ratio"] = (np.sqrt((tau[seg] ** 2).mean(0)) / rated).tolist()
      for label, (a, b) in (("stop_4_7s", (13.0, 16.0)), ("stop_end", (32.0, 35.0))):
        ia, ib = round(a / dt), round(b / dt)
        if ff < ib:
          continue
        jerr = np.degrees(d["jerr"][ia:ib, i])
        row[label] = {
          "joint_rmse_deg": float(np.sqrt((jerr**2).mean())),
          "joint_max_abs_deg": float(np.abs(jerr).max()),
          "joint_mean_err_deg": jerr.mean(0).tolist(),
          "four_contact_frac": float(d["contact"][ia:ib, i].all(-1).mean()),
          "base_speed": float(np.linalg.norm(d["vel"][ia:ib, i, :2], axis=-1).mean()),
          "yaw_rate": float(np.abs(d["vel"][ia:ib, i, 2]).mean()),
          "tau_rms_ratio": (np.sqrt((tau[ia:ib] ** 2).mean(0)) / rated).tolist(),
        }
    else:
      x = pos[:, 0]
      reach = np.nonzero(x >= GOAL_X)[0]
      t_reach = int(reach[0]) if len(reach) else steps
      success = t_reach < ff and t_reach < round(STAIR_WALK_END / dt)
      row["height"] = c["height"]
      row["success"] = bool(success)
      row["max_x_m"] = float(x[: min(ff, steps)].max()) if ff > 0 else 0.0
      row["stuck"] = bool(not success and ff >= round(STAIR_WALK_END / dt))
      end = min(t_reach, ff, steps)
      on = (x[:end] > STAIR_START + 0.2) & (x[:end] < STAIR_END - 0.2)
      if on.sum() > 5:
        idx = np.nonzero(on)[0]
        row["stair_time_s"] = float(len(idx) * dt)
        row["heading_drift_max_deg"] = float(np.abs(drift[i_start:end]).max())
        row["heading_drift_stair_max_deg"] = float(np.abs(drift[idx]).max())
        row["lateral_drift_m"] = float(pos[end - 1, 1] - pos[i_start, 1])
        row["nose_up_mean_deg"] = float(nose_up[idx].mean())
        row["nose_up_min_deg"] = float(nose_up[idx].min())
        row["nose_up_max_deg"] = float(nose_up[idx].max())
        row["nose_up_std_deg"] = float(nose_up[idx].std())
        row["roll_abs_max_deg"] = float(np.abs(roll[idx]).max())
        rms = np.sqrt((tau[idx] ** 2).mean(0))
        row["tau_rms_ratio"] = (rms / rated).tolist()
        row["tau_peak_ratio"] = (np.abs(tau[idx]).max(0) / peak).tolist()
        row["mean_speed"] = float(
          (x[idx[-1]] - x[idx[0]]) / max((idx[-1] - idx[0]) * dt, dt)
        )
    rows.append(row)
  return rows


def summarize(rows: list[dict], actuator_names: list[str]) -> dict:
  out: dict = {"flat": {}, "stairs": {}}
  flat = [r for r in rows if r["kind"] == "flat"]
  for cmd in FLAT_COMMANDS:
    rs = [r for r in flat if tuple(r["cmd"]) == cmd]
    if not rs:
      continue
    ok = [r for r in rs if r["walk_survived"]]
    s: dict = {
      "n": len(rs),
      "walk_survival": len(ok) / len(rs),
      "survival_25s": float(np.mean([r["survived"] for r in rs])),
    }
    if ok:
      s["track_mae_vx_vy_wz"] = (
        np.mean([r["track_mae"] for r in ok], 0).round(3).tolist()
      )
      if cmd[2] == 0.0:
        dm = np.array([abs(r["heading_drift_max_deg"]) for r in ok])
        s["heading_drift_max_deg_p50_p90_max"] = [
          float(np.percentile(dm, 50)),
          float(np.percentile(dm, 90)),
          float(dm.max()),
        ]
        s["heading_within_2deg"] = float((dm <= 2.0).mean())
        s["walk_dist_m_mean"] = float(np.mean([r["walk_dist_m"] for r in ok]))
    stop = [r["stop_4_7s"] for r in rs if "stop_4_7s" in r]
    if stop:
      rm = np.array([x["joint_rmse_deg"] for x in stop])
      mx = np.array([x["joint_max_abs_deg"] for x in stop])
      fc = np.array([x["four_contact_frac"] for x in stop])
      sp = np.array([x["base_speed"] for x in stop])
      good = (rm <= 3.0) & (mx <= 6.0) & (fc >= 0.95) & (sp <= 0.05)
      s["stop_joint_rmse_deg_p50_p90_max"] = [
        float(np.percentile(rm, 50)),
        float(np.percentile(rm, 90)),
        float(rm.max()),
      ]
      s["stop_joint_maxabs_deg_p50_p90_max"] = [
        float(np.percentile(mx, 50)),
        float(np.percentile(mx, 90)),
        float(mx.max()),
      ]
      s["stop_four_contact_mean"] = float(fc.mean())
      s["stop_base_speed_mean"] = float(sp.mean())
      s["stop_pass_rate"] = float(good.mean())
      s["stop_mean_joint_err_deg"] = (
        np.mean([x["joint_mean_err_deg"] for x in stop], 0).round(2).tolist()
      )
    out["flat"][str(cmd)] = s

  stairs = [r for r in rows if r["kind"] != "flat"]
  for kind in ("up", "down"):
    for h in HEIGHTS:
      for vx in STAIR_SPEEDS:
        rs = [
          r
          for r in stairs
          if r["kind"] == kind and abs(r["height"] - h) < 1e-6 and r["cmd"][0] == vx
        ]
        if not rs:
          continue
        ok = [r for r in rs if r["success"]]
        reasons: dict[str, int] = {}
        for r in rs:
          if not r["success"]:
            key = r["fail_reason"] or ("stuck" if r["stuck"] else "slow")
            reasons[key] = reasons.get(key, 0) + 1
        s = {
          "n": len(rs),
          "success_rate": len(ok) / len(rs),
          "fail_reasons": reasons,
          "max_x_mean": float(np.mean([r["max_x_m"] for r in rs])),
        }
        meas = [r for r in ok if "heading_drift_max_deg" in r]
        if meas:
          dm = np.array([r["heading_drift_max_deg"] for r in meas])
          s["heading_drift_max_deg_p50_p90_max"] = [
            float(np.percentile(dm, 50)),
            float(np.percentile(dm, 90)),
            float(dm.max()),
          ]
          s["heading_within_2deg"] = float((dm <= 2.0).mean())
          s["lateral_drift_abs_mean_m"] = float(
            np.mean([abs(r["lateral_drift_m"]) for r in meas])
          )
          s["nose_up_mean_deg"] = float(np.mean([r["nose_up_mean_deg"] for r in meas]))
          s["nose_up_min_deg_worst"] = float(
            np.min([r["nose_up_min_deg"] for r in meas])
          )
          s["nose_up_max_deg_best"] = float(
            np.max([r["nose_up_max_deg"] for r in meas])
          )
          s["roll_abs_max_deg_worst"] = float(
            np.max([r["roll_abs_max_deg"] for r in meas])
          )
          ratio = np.array([r["tau_rms_ratio"] for r in meas])
          s["tau_rms_ratio_mean_per_actuator"] = dict(
            zip(actuator_names, ratio.mean(0).round(2).tolist(), strict=True)
          )
          s["tau_rms_ratio_max_actuator_mean"] = float(ratio.max(1).mean())
          s["tau_peak_ratio_max"] = float(
            np.max([max(r["tau_peak_ratio"]) for r in meas])
          )
          s["stair_time_s_mean"] = float(np.mean([r["stair_time_s"] for r in meas]))
          s["mean_speed"] = float(np.mean([r["mean_speed"] for r in meas]))
        out["stairs"][f"{kind}_{round(h * 100)}cm_v{vx}"] = s
  return out


def main():
  global HEIGHTS, PER_COLUMN, FLAT_COLUMNS
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--actuator", choices=("train", "j60"), default="train")
  parser.add_argument("--no-dr", action="store_true")
  parser.add_argument("--device", default="cuda:1")
  parser.add_argument("--output", required=True)
  parser.add_argument("--heading-hold", type=float, default=0.0)
  parser.add_argument("--heights", type=float, nargs="+", default=list(HEIGHTS))
  parser.add_argument("--per-column", type=int, default=PER_COLUMN)
  parser.add_argument("--no-flat", action="store_true")
  args = parser.parse_args()
  HEIGHTS = tuple(args.heights)
  PER_COLUMN = args.per_column
  FLAT_COLUMNS = () if args.no_flat else ("flat_a", "flat_b")
  r = rollout(args)
  rows = analyze(r)
  summary = summarize(rows, r["actuator_names"])
  result = {
    "checkpoint": args.checkpoint,
    "actuator": args.actuator,
    "domain_rand": not args.no_dr,
    "heading_hold": args.heading_hold,
    "actuator_names": r["actuator_names"],
    "joint_names": r["joint_names"],
    "rated_torque": r["rated"].tolist(),
    "summary": summary,
    "rows": rows,
  }
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=1))
  print(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()

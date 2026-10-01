"""Aggregate equal-weight command trials without mixing units."""

import json
from pathlib import Path

import numpy as np

folder = Path(__file__).parent
out = {}
for path in [folder / "screening.json", folder / "terrain.json"]:
  if not path.exists():
    continue
  rows = json.loads(path.read_text())
  if path.stem == "terrain" and (folder / "terrain_700.json").exists():
    rows += json.loads((folder / "terrain_700.json").read_text())
  summaries = {}
  for name in dict.fromkeys(r["model"] for r in rows):
    group = [r for r in rows if r["model"] == name]
    walks = [p for r in group for p in r["phases"] if p["phase"] == "walk"]
    stops = [p for r in group for p in r["phases"] if p["phase"] == "stop"]
    summaries[name] = {
      "command_trials": len(group),
      "failed_trials": sum(r["failed"] for r in group),
      "mean_tracking_mae_xyz": np.concatenate([p["tracking_mae"] for p in walks])
      .mean(0)
      .tolist(),
      "mean_stop_pose_rmse_deg": float(
        np.rad2deg(np.concatenate([p["pose_rmse_rad"] for p in stops])).mean()
      )
      if stops
      else None,
      "worst_stop_pose_rmse_deg": float(
        np.rad2deg(np.concatenate([p["pose_rmse_rad"] for p in stops])).max()
      )
      if stops
      else None,
      "mean_walk_raw_clip_fraction": float(
        np.concatenate([p["clip_fraction"] for p in walks]).mean()
      ),
      "min_stop_stand_fraction": float(
        np.concatenate([p["stand_fraction"] for p in stops]).min()
      )
      if stops
      else None,
      "min_stop_four_contact_fraction": float(
        np.concatenate([p["four_contact_fraction"] for p in stops]).min()
      )
      if stops
      else None,
      "max_roll_pitch_deg": float(
        np.rad2deg(np.concatenate([r["max_roll_pitch_rad"] for r in group])).max()
      ),
      "solver_failure_count": int(sum(sum(r["solver_failure_count"]) for r in group)),
    }
  out[path.stem] = summaries
(folder / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

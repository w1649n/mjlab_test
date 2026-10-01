"""Summarize the latest run using trailing 100-iteration means."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parents[2]
run = root / "logs/rsl_rl/g23_rl_mpc_foot_state_history_v3/2026-09-10_11-11-39"
ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
ea.Reload()
tags = [
  "Train/mean_reward",
  "Metrics/raw_action_clip_fraction",
  "Metrics/twist/error_vel_xy",
  "Metrics/twist/error_vel_yaw",
  "Curriculum/terrain_levels/mean",
  "Episode_Reward/stopped_joint_pose_l2",
]
labels = [
  "Mean reward",
  "Raw action clipping fraction",
  "Logged XY error (not MAE)",
  "Logged yaw error (not MAE)",
  "Mean terrain level",
  "Weighted stopped-pose reward",
]
fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
summary = {}
for ax, tag, label in zip(axes.flat, tags, labels, strict=True):
  values = ea.Scalars(tag)
  x = np.array([v.step for v in values])
  y = np.array([v.value for v in values])
  smooth = np.convolve(y, np.ones(100) / 100, mode="valid")
  ax.plot(x[99:], smooth, lw=1.4)
  ax.set_title(label)
  ax.grid(alpha=0.25)
  summary[tag] = {}
  for checkpoint in [700, 2400, 4000, 5399]:
    mask = (x <= checkpoint) & (x > checkpoint - 100)
    summary[tag][str(checkpoint)] = float(y[mask].mean()) if mask.any() else None
    ax.axvline(checkpoint, color="gray", alpha=0.25, lw=0.8)
for ax in axes[-1]:
  ax.set_xlabel("Iteration")
fig.suptitle("G23 RLMPC 2026-09-10_11-11-39 — trailing 100 samples")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "training_curves.png", dpi=160)
(Path(__file__).parent / "checkpoint_training_windows.json").write_text(
  json.dumps(summary, indent=2)
)
print(json.dumps(summary, indent=2))

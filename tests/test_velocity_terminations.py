"""Tests for velocity task termination helpers."""

from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.terminations import illegal_contact


def test_illegal_contact_can_require_multiple_history_hits() -> None:
  force_history = torch.zeros((4, 2, 4, 3))
  force_history[0, 0, :2, 0] = 11.0
  force_history[1, 0, :3, 0] = 11.0
  force_history[2, 1, 0, 0] = 11.0
  force_history[3, 0, (0, 1, 3), 0] = 11.0
  env = SimpleNamespace(
    scene={
      "touch": SimpleNamespace(data=SimpleNamespace(force_history=force_history))
    }
  )

  terminated = illegal_contact(
    env,
    sensor_name="touch",
    force_threshold=10.0,
    history_count_threshold=3,
  )

  assert terminated.tolist() == [False, True, False, False]

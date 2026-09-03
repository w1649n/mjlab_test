"""Tests for resume-specific training behavior."""

from pathlib import Path

import pytest

from mjlab.scripts.train import _should_randomize_initial_episode_lengths


@pytest.mark.parametrize(
  ("resume_path", "expected"),
  [
    (None, True),
    (Path("logs/rsl_rl/run/model_100.pt"), False),
  ],
)
def test_initial_episode_length_randomization_is_disabled_on_resume(
  resume_path: Path | None, expected: bool
):
  assert _should_randomize_initial_episode_lengths(resume_path) is expected

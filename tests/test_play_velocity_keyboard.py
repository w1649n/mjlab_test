"""Tests for terminal keyboard velocity commands during play."""

from __future__ import annotations

import math
import os
import sys
from threading import Event
from unittest.mock import MagicMock

import pytest

from mjlab.scripts.play import (
  _dispatch_terminal_key,
  _TerminalKeyReader,
  _VelocityKeyboardController,
)
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_M,
  KEY_Q,
  KEY_R,
  KEY_S,
  KEY_SPACE,
  KEY_W,
  KEY_X,
)
from mjlab.viewer.native.viewer import NativeMujocoViewer


class _FakeVelocityCommand:
  """Small stand-in for the command term's per-environment manual API."""

  def __init__(self) -> None:
    self._commands: dict[int, tuple[float, float, float]] = {}

  def set_manual_command(
    self, env_idx: int, command: tuple[float, float, float] | None
  ) -> tuple[float, float, float] | None:
    if command is None:
      self._commands.pop(env_idx, None)
      return None
    self._commands[env_idx] = (
      float(command[0]),
      float(command[1]),
      float(command[2]),
    )
    return self._commands[env_idx]


def _make_controller(
  *, step: float = 0.1
) -> tuple[_VelocityKeyboardController, _FakeVelocityCommand, list[int]]:
  term = _FakeVelocityCommand()
  selected_env = [0]

  def get_env_idx() -> int:
    return selected_env[0]

  controller = _VelocityKeyboardController(term, get_env_idx, step=step)
  return controller, term, selected_env


def test_m_toggles_selected_environment_between_random_and_manual() -> None:
  controller, term, _ = _make_controller()

  assert term._commands.get(0) is None
  assert controller(KEY_M) is True
  assert term._commands[0] == (0.0, 0.0, 0.0)

  assert controller(KEY_M) is True
  assert term._commands.get(0) is None


@pytest.mark.parametrize(
  ("key", "expected"),
  [
    (KEY_W, (0.1, 0.0, 0.0)),
    (KEY_S, (-0.1, 0.0, 0.0)),
    (KEY_A, (0.0, 0.1, 0.0)),
    (KEY_D, (0.0, -0.1, 0.0)),
    (KEY_Q, (0.0, 0.0, 0.1)),
    (KEY_E, (0.0, 0.0, -0.1)),
  ],
)
def test_direction_keys_adjust_manual_command(
  key: int, expected: tuple[float, float, float]
) -> None:
  controller, term, _ = _make_controller()
  controller(KEY_M)

  assert controller(key) is True
  assert term._commands[0] == pytest.approx(expected)


def test_direction_keys_accumulate_and_x_zeros_command() -> None:
  controller, term, _ = _make_controller()
  controller(KEY_M)

  controller(KEY_W)
  controller(KEY_W)
  controller(KEY_A)
  controller(KEY_Q)
  assert term._commands[0] == pytest.approx((0.2, 0.1, 0.1))

  assert controller(KEY_X) is True
  assert term._commands[0] == (0.0, 0.0, 0.0)


def test_space_zeros_command_like_x() -> None:
  controller, term, _ = _make_controller()
  controller(KEY_W)
  controller(KEY_A)
  controller(KEY_Q)
  assert term._commands[0] == pytest.approx((0.1, 0.1, 0.1))

  assert controller.handles_key(KEY_SPACE)
  assert controller(KEY_SPACE) is True
  assert term._commands[0] == (0.0, 0.0, 0.0)


def test_terminal_dispatch_defers_space_command_to_viewer_queue() -> None:
  env = MagicMock()
  env.cfg.viewer.env_idx = 0
  env.unwrapped.num_envs = 1
  viewer = NativeMujocoViewer(env, MagicMock())
  controller, term, _ = _make_controller()
  controller(KEY_W)

  assert _dispatch_terminal_key(
    KEY_SPACE, controller, viewer.request_callback, viewer.request_reset
  )
  assert term._commands[0] == pytest.approx((0.1, 0.0, 0.0))
  assert len(viewer._actions) == 1
  assert viewer._actions[0][0] == ViewerAction.CALLBACK

  viewer._process_actions()
  assert term._commands[0] == (0.0, 0.0, 0.0)


def test_terminal_dispatch_r_requests_reset_through_viewer_queue() -> None:
  env = MagicMock()
  env.cfg.viewer.env_idx = 0
  env.unwrapped.num_envs = 1
  viewer = NativeMujocoViewer(env, MagicMock())
  controller, term, _ = _make_controller()

  assert _dispatch_terminal_key(
    KEY_R, controller, viewer.request_callback, viewer.request_reset
  )

  # Terminal dispatch queues RESET; it never touches the env or velocity term.
  env.reset.assert_not_called()
  assert list(viewer._actions) == [(ViewerAction.RESET, None)]
  assert term._commands == {}

  viewer._process_actions()
  env.reset.assert_called_once_with()


def test_unmapped_key_is_not_consumed_or_applied() -> None:
  controller, term, _ = _make_controller()

  assert controller(-12345) is False
  assert term._commands.get(0) is None


def test_manual_commands_are_isolated_per_environment() -> None:
  controller, term, selected_env = _make_controller()

  controller(KEY_M)
  controller(KEY_W)
  selected_env[0] = 1
  assert term._commands.get(1) is None
  controller(KEY_M)
  controller(KEY_S)

  assert term._commands[0] == pytest.approx((0.1, 0.0, 0.0))
  assert term._commands[1] == pytest.approx((-0.1, 0.0, 0.0))

  selected_env[0] = 0
  controller(KEY_M)
  assert term._commands.get(0) is None
  assert term._commands[1] == pytest.approx((-0.1, 0.0, 0.0))


@pytest.mark.parametrize(
  "data",
  [b"\x1b[A", b"\x1b[B", b"\x1b[C", b"\x1b[D"],
  ids=["up", "down", "right", "left"],
)
def test_terminal_reader_ignores_ansi_arrow_keys(data: bytes) -> None:
  assert list(_TerminalKeyReader._key_codes(data)) == []


def test_terminal_reader_ignores_escape_sequence_split_across_reads() -> None:
  state = 0
  keys: list[int] = []
  for data in (b"\x1b", b"[", b"A", b"w"):
    decoded, state = _TerminalKeyReader._decode_key_codes(data, state)
    keys.extend(decoded)

  assert keys == [KEY_W]
  assert state == 0


@pytest.mark.skipif(os.name != "posix", reason="PTY and termios require POSIX")
def test_terminal_reader_reads_without_enter_and_restores_tty(monkeypatch) -> None:
  import pty
  import termios

  master_fd, slave_fd = pty.openpty()
  before = termios.tcgetattr(slave_fd)
  stream = os.fdopen(os.dup(slave_fd), "r")
  monkeypatch.setattr(sys, "stdin", stream)
  received: list[int] = []
  received_event = Event()

  def on_key(key: int) -> bool:
    received.append(key)
    received_event.set()
    return True

  try:
    with _TerminalKeyReader(on_key) as reader:
      assert reader.enabled
      assert not (termios.tcgetattr(slave_fd)[3] & termios.ICANON)
      os.write(master_fd, b"w")
      assert received_event.wait(timeout=1.0)

    assert received == [KEY_W]
    assert termios.tcgetattr(slave_fd) == before
  finally:
    stream.close()
    os.close(master_fd)
    os.close(slave_fd)


@pytest.mark.parametrize("step", [0.0, -0.1, math.inf, -math.inf, math.nan])
def test_step_must_be_positive_and_finite(step: float) -> None:
  term = _FakeVelocityCommand()

  with pytest.raises(ValueError, match="step"):
    _VelocityKeyboardController(term, lambda: 0, step=step)

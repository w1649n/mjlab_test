"""Focused episode-boundary checks for the optional legacy MPC backend."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mjlab.tasks.rl_mpc.controller.backend import (
  LegacyG23MpcController,
  LegacyMpcBackendCfg,
  MpcGaitMode,
)


def _make_controller_or_skip(
  *,
  enable_stand_mode: bool = False,
  stand_command_hold_time: float = 0.20,
  stand_contact_hold_time: float = 0.05,
  stand_contact_loss_time: float = 0.05,
  stand_arm_timeout: float = 0.20,
) -> LegacyG23MpcController:
  try:
    return LegacyG23MpcController(
      LegacyMpcBackendCfg(
        formulation="syncai",
        controller_dt=0.01,
        iterations_between_mpc=3,
        flat_ground=True,
        body_mass=15.7,
        foot_landing_height=0.019,
        enable_stand_mode=enable_stand_mode,
        stand_command_hold_time=stand_command_hold_time,
        stand_contact_hold_time=stand_contact_hold_time,
        stand_contact_loss_time=stand_contact_loss_time,
        stand_arm_timeout=stand_arm_timeout,
      )
    )
  except RuntimeError as exc:
    pytest.skip(f"optional rl-mpc native backend is unavailable: {exc}")


def _nominal_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # Simulator-side signs; the legacy G23 leg adapter applies its own sign map.
  joint_pos = np.tile(np.array([0.0, -0.9, 1.8], dtype=np.float32), 4)
  dof_states = np.column_stack((joint_pos, np.zeros(12, dtype=np.float32)))
  body_states = np.zeros(13, dtype=np.float32)
  body_states[2] = 0.30
  body_states[6] = 1.0  # xyzw quaternion
  return dof_states, body_states, np.zeros(3, dtype=np.float32)


def test_straight_residuals_share_one_sample_across_diagonals() -> None:
  backend = _make_controller_or_skip()
  controller = backend._runner.cMPC
  controller.symmetric_residuals = True
  controller._x_vel_des, controller._y_vel_des, controller._yaw_turn_rate = 0.2, 0, 0
  controller._pending_foot_placement_offsets[:] = [
    [0.08, 0.03],
    [-0.02, 0.01],
    [-0.08, 0.02],
    [0.04, -0.03],
  ]
  controller._trot_iteration = 1
  sample = controller._updateSymmetricResiduals().copy()
  np.testing.assert_allclose(sample[:, 0], [0.03, 0.03, -0.02, -0.02], atol=1e-7)
  np.testing.assert_allclose(sample[:, 1], [0.01, -0.01, 0.025, -0.025], atol=1e-7)
  controller._pending_foot_placement_offsets[:] *= -1
  controller._trot_iteration = 17  # Opposite diagonal; period is 30 ticks here.
  np.testing.assert_array_equal(controller._updateSymmetricResiduals(), sample)
  controller._trot_iteration = 31
  np.testing.assert_allclose(controller._updateSymmetricResiduals(), -sample)
  controller._yaw_turn_rate = 0.2
  controller._trot_iteration = 61
  bounded = controller._updateSymmetricResiduals().copy()
  assert np.max(np.abs(bounded[:, 0])) <= 0.015 + 1e-7
  assert np.max(np.abs(bounded[:, 1])) <= 0.0075 + 1e-7
  for left, right in ((0, 1), (2, 3)):
    assert abs(bounded[left, 0] - bounded[right, 0]) <= 0.005 + 1e-7
    assert abs(bounded[left, 1] + bounded[right, 1]) <= 0.005 + 1e-7
  controller._pending_foot_placement_offsets[:] *= -1
  controller._trot_iteration = 77
  np.testing.assert_array_equal(controller._updateSymmetricResiduals(), bounded)
  backend.reset()
  assert controller._symmetric_cycle == -1
  assert not np.any(controller._symmetric_offsets)


@pytest.mark.parametrize("walking_ticks", [1, 14, 15, 16, 29, 30, 31])
def test_stop_places_both_diagonals_before_stand(
  monkeypatch: pytest.MonkeyPatch,
  walking_ticks: int,
) -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_command_hold_time=0.01,
    stand_contact_hold_time=0.02,
  )
  controller = backend._runner.cMPC
  controller.stop_reposition_steps = 2
  monkeypatch.setattr(controller, "solveDenseMPC", lambda *_: None)
  dofs, body, command = _nominal_inputs()
  offsets = np.full((4, 2), 0.04, dtype=np.float32)
  contacts = np.ones(4, dtype=bool)
  for _ in range(3):
    backend.step(dofs, body, command, offsets, contacts)
  command[0] = 0.2
  last_swing = None
  for _ in range(walking_ticks):
    backend.step(dofs, body, command, offsets, contacts)
    swing = controller.last_gait_inputs["swing_states"].reshape(4)
    if np.any(swing > 0):
      last_swing = np.flatnonzero(swing > 0).tolist()
  assert last_swing is not None
  command[:] = 0
  launched = []
  previous_remaining = None
  for _ in range(100):
    # Stale true contacts must not cancel the commanded liftoff immediately.
    backend.step(dofs, body, command, offsets, contacts)
    if controller._stop_step_active:
      remaining = controller._stop_steps_remaining
      if remaining != previous_remaining:
        launched.append(np.flatnonzero(controller._stopping_swing_legs).tolist())
        previous_remaining = remaining
      assert backend.gait_mode == MpcGaitMode.STOPPING
      assert controller._x_vel_des == controller._y_vel_des == 0
      for leg in np.flatnonzero(controller._stopping_swing_legs):
        np.testing.assert_allclose(
          controller.footSwingTrajectories[leg]._pf[:2],
          body[:2, None] + controller._home_foot_positions[leg, :2],
          atol=1e-6,
        )
    if backend.gait_mode == MpcGaitMode.STAND:
      break
  opposite = [leg for leg in range(4) if leg not in last_swing]
  assert launched == [opposite, last_swing]
  assert backend.gait_mode == MpcGaitMode.STAND
  assert controller._stop_steps_remaining == 0
  backend.reset()
  assert not controller._stop_step_active
  assert not np.any(controller._home_foot_positions)
  assert controller._last_swing_diagonal is None


@pytest.mark.parametrize(
  "command", [(0, 0.1, 0), (0, -0.1, 0), (0, 0, 0.3), (0, 0, -0.3)]
)
def test_maneuver_residual_limits_hold_for_extreme_policy_outputs(command) -> None:
  backend = _make_controller_or_skip()
  controller = backend._runner.cMPC
  controller.symmetric_residuals = True
  controller._x_vel_des, controller._y_vel_des, controller._yaw_turn_rate = command
  rng = np.random.default_rng(42)
  for cycle in range(100):
    controller._pending_foot_placement_offsets[:] = rng.uniform(-0.08, 0.08, (4, 2))
    controller._trot_iteration = 1 + 30 * cycle
    projected = controller._updateSymmetricResiduals().copy()
    assert np.all(np.abs(projected) <= np.array([0.015, 0.0075]) + 1e-7)
    for left, right in ((0, 1), (2, 3)):
      assert abs(projected[left, 0] - projected[right, 0]) <= 0.005 + 1e-7
      assert abs(projected[left, 1] + projected[right, 1]) <= 0.005 + 1e-7
    controller._pending_foot_placement_offsets[:] *= -1
    controller._trot_iteration += 16
    np.testing.assert_array_equal(controller._updateSymmetricResiduals(), projected)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_mpc_interval_is_rejected_before_loading_backend(value: object) -> None:
  with pytest.raises(ValueError, match="iterations_between_mpc"):
    LegacyG23MpcController(
      LegacyMpcBackendCfg(iterations_between_mpc=value)  # type: ignore[arg-type]
    )


def test_reposition_waits_for_touchdown_before_accepting_walk(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_command_hold_time=0.01,
    stand_contact_hold_time=0.01,
  )
  controller = backend._runner.cMPC
  controller.stop_reposition_steps = 2
  monkeypatch.setattr(controller, "solveDenseMPC", lambda *_: None)
  dofs, body, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)
  contacts = np.ones(4, dtype=bool)
  backend.step(dofs, body, command, offsets, contacts)
  command[0] = 0.2
  backend.step(dofs, body, command, offsets, contacts)
  command[0] = 0
  for _ in range(10):
    backend.step(dofs, body, command, offsets, contacts)
    if controller._stop_step_active:
      break
  assert controller._stop_step_active
  contacts[3] = False
  command[0] = 0.2
  for _ in range(40):
    backend.step(dofs, body, command, offsets, contacts)
    assert backend.gait_mode == MpcGaitMode.STOPPING
    assert controller._x_vel_des == 0
  assert controller._stopping_search_depth[3] > 0
  contacts[:] = True
  backend.step(dofs, body, command, offsets, contacts)
  assert backend.gait_mode == MpcGaitMode.TROT
  assert not controller._stop_step_active
  assert controller._stop_steps_remaining == 0


def test_episode_reset_clears_controller_state_and_forces_first_qp_update(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip()
  runner = backend._runner
  locomotion = runner.cMPC
  assert locomotion.iterationsBetweenMPC == 3
  assert locomotion.dtMPC == pytest.approx(0.03)
  solve_iterations: list[int] = []

  def fake_solve_dense_mpc(mpc_table: list[int], data: object) -> None:
    del mpc_table, data
    solve_iterations.append(locomotion.iterationCounter)
    locomotion.f_ff.fill(7.0)

  monkeypatch.setattr(locomotion, "solveDenseMPC", fake_solve_dense_mpc)
  dof_states, body_states, command = _nominal_inputs()

  backend.step(
    dof_states,
    body_states,
    command,
    np.full((4, 2), 0.04, dtype=np.float32),
  )
  assert solve_iterations == [1]
  np.testing.assert_allclose(locomotion.swingTimes, 0.15)
  assert np.any(locomotion.f_ff)
  assert np.any(locomotion.pFoot)
  assert np.any(locomotion.foot_positions)
  assert np.any(locomotion._pending_foot_placement_offsets)
  assert np.any(locomotion._latched_foot_placement_offsets)

  backend.reset()

  assert locomotion.iterationCounter == 0
  assert locomotion.firstRun
  assert locomotion.firstSwing == [True] * 4
  assert locomotion._mpc_update_required
  assert not np.any(locomotion.f_ff)
  assert not np.any(locomotion.pFoot)
  assert not np.any(locomotion.foot_positions)
  assert not np.any(locomotion.swingTimes)
  assert locomotion.swingTimeRemaining == [0.0] * 4
  assert not np.any(locomotion._pending_foot_placement_offsets)
  assert not np.any(locomotion._latched_foot_placement_offsets)
  assert locomotion.last_mpc_inputs is None
  assert locomotion.last_gait_inputs is None

  for trajectory in locomotion.footSwingTrajectories:
    assert trajectory._p0 is None
    assert trajectory._pf is None
    assert not np.any(trajectory._p)
    assert not np.any(trajectory._v)
    assert not np.any(trajectory._a)
  for leg_command in runner._legController.commands:
    for value in vars(leg_command).values():
      if isinstance(value, np.ndarray):
        assert not np.any(value)

  solve_iterations.clear()
  backend.step(
    dof_states,
    body_states,
    command,
    np.zeros((4, 2), dtype=np.float32),
  )
  assert solve_iterations == [1]
  assert not locomotion._mpc_update_required


def test_lower_frequency_trot_keeps_gait_and_solver_cadence_aligned(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip()
  locomotion = backend._runner.cMPC
  solve_iterations: list[int] = []

  def fake_solve_dense_mpc(mpc_table: list[int], data: object) -> None:
    del data
    solve_iterations.append(locomotion.iterationCounter)
    assert len(mpc_table) == locomotion.horizonLength * 4

  monkeypatch.setattr(locomotion, "solveDenseMPC", fake_solve_dense_mpc)
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)

  for _ in range(6):
    backend.step(dof_states, body_states, command, offsets)

  assert solve_iterations == [1, 4]
  assert locomotion.dtMPC == pytest.approx(0.03)
  np.testing.assert_allclose(locomotion.swingTimes, 0.15)
  assert locomotion.last_gait_inputs is not None
  assert locomotion.last_gait_inputs["mpc_table_rows_horizon_cols_legs"].shape == (
    10,
    4,
  )
  gait = locomotion.trotting
  gait.setIterations(3, 1)
  first_segment = np.asarray(gait.getMpcTable()).copy()
  gait.setIterations(3, 2)
  np.testing.assert_array_equal(gait.getMpcTable(), first_segment)
  gait.setIterations(3, 3)
  assert not np.array_equal(gait.getMpcTable(), first_segment)


def test_trot_touchdown_height_follows_local_terrain_scan(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip()
  locomotion = backend._runner.cMPC
  monkeypatch.setattr(locomotion, "solveDenseMPC", lambda *_: None)
  dof_states, body_states, command = _nominal_inputs()
  samples = np.array(
    [
      [-1.0, -1.0, 0.04],
      [-1.0, 1.0, 0.04],
      [1.0, -1.0, 0.04],
      [1.0, 1.0, 0.04],
    ],
    dtype=np.float32,
  )

  backend.step(
    dof_states,
    body_states,
    command,
    np.zeros((4, 2), dtype=np.float32),
    terrain_height_samples=samples,
  )

  for trajectory in locomotion.footSwingTrajectories:
    assert trajectory._pf is not None
    assert trajectory._pf[2, 0] == pytest.approx(0.059, abs=1e-4)


def test_standing_gait_is_time_invariant_all_contact() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  gait = backend._runner.cMPC.standing

  for iteration in (0, 1, 17, 10_000):
    gait.setIterations(3, iteration)
    np.testing.assert_array_equal(gait.getContactState(), np.ones((4, 1)))
    np.testing.assert_array_equal(gait.getSwingState(), np.zeros((4, 1)))
    np.testing.assert_array_equal(gait.getMpcTable(), np.ones(40))
    assert gait.getCurrentSwingTime(0.03, 0) == 0.0


def test_stopping_gait_horizon_predicts_touchdown() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  gait = backend._runner.cMPC.stopping
  gait.configure(
    np.array([True, False, False, False]),
    np.array([0.4, 0.0, 0.0, 0.0]),
    np.array([False, True, True, True]),
  )

  table = np.asarray(gait.getMpcTable()).reshape(10, 4)
  np.testing.assert_array_equal(table[:3, 0], np.zeros(3))
  np.testing.assert_array_equal(table[3:, 0], np.ones(7))
  np.testing.assert_array_equal(table[:, 1:], np.ones((10, 3)))
  assert gait.getCurrentStanceTime(0.03, 0) == pytest.approx(0.30)


def test_stopping_gait_keeps_current_search_foot_out_of_contact() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  gait = backend._runner.cMPC.stopping
  gait.configure(
    np.array([True, False, False, False]),
    np.array([1.0, 0.0, 0.0, 0.0]),
    np.array([False, True, True, True]),
  )

  table = np.asarray(gait.getMpcTable()).reshape(10, 4)
  assert table[0, 0] == 0
  np.testing.assert_array_equal(table[1:, 0], np.ones(9))
  np.testing.assert_array_equal(table[:, 1:], np.ones((10, 3)))


def test_zero_command_starts_in_stand_and_masks_offsets(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  dof_states, body_states, command = _nominal_inputs()
  solved_tables: list[np.ndarray] = []

  def fake_solve_dense_mpc(mpc_table: list[int], data: object) -> None:
    del data
    solved_tables.append(np.asarray(mpc_table).copy())

  monkeypatch.setattr(locomotion, "solveDenseMPC", fake_solve_dense_mpc)
  backend.step(
    dof_states,
    body_states,
    command,
    np.full((4, 2), 0.08, dtype=np.float32),
    np.ones(4, dtype=bool),
  )

  assert backend.gait_mode == MpcGaitMode.STAND
  assert len(solved_tables) == 1
  np.testing.assert_array_equal(solved_tables[0], np.ones(40))
  assert locomotion.last_gait_inputs is not None
  np.testing.assert_array_equal(
    locomotion.last_gait_inputs["contact_states"], np.ones((4, 1))
  )
  np.testing.assert_array_equal(
    locomotion.last_gait_inputs["swing_states"], np.zeros((4, 1))
  )
  assert not np.any(locomotion._pending_foot_placement_offsets)
  assert not np.any(locomotion._latched_foot_placement_offsets)


def test_stand_stop_trot_state_machine_uses_hysteresis_and_real_contacts(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_command_hold_time=0.03,
    stand_contact_hold_time=0.02,
  )
  locomotion = backend._runner.cMPC
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.full((4, 2), 0.04, dtype=np.float32)
  all_contacts = np.ones(4, dtype=bool)
  solve_modes: list[int] = []

  def fake_solve_dense_mpc(mpc_table: list[int], data: object) -> None:
    del mpc_table, data
    solve_modes.append(int(locomotion.gait_mode))

  monkeypatch.setattr(locomotion, "solveDenseMPC", fake_solve_dense_mpc)

  # Commands inside the hysteresis band do not wake a standing controller.
  command[0] = 0.05
  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.STAND

  command[0] = 0.20
  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.TROT
  assert solve_modes[-1] == MpcGaitMode.TROT
  first_start_swing = locomotion.last_gait_inputs["swing_states"].copy()

  command[:] = 0.0
  for _ in range(2):
    backend.step(dof_states, body_states, command, offsets, all_contacts)
    assert backend.gait_mode == MpcGaitMode.TROT
  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.STOPPING
  assert not np.any(locomotion._pending_foot_placement_offsets)

  backend.step(
    dof_states,
    body_states,
    command,
    offsets,
    np.array([True, True, True, False]),
  )
  assert backend.gait_mode == MpcGaitMode.STOPPING

  moving_body = body_states.copy()
  moving_body[7] = 0.2
  backend.step(dof_states, moving_body, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.STOPPING

  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.STOPPING
  assert locomotion._stand_settling
  assert locomotion.last_gait_inputs["gait"] == "Stopping landing"
  np.testing.assert_array_equal(
    locomotion.last_gait_inputs["swing_states"], np.zeros((4, 1))
  )

  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.STAND
  assert not np.any(locomotion._pending_foot_placement_offsets)
  assert not np.any(locomotion._latched_foot_placement_offsets)

  command[0] = 0.20
  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.gait_mode == MpcGaitMode.TROT
  np.testing.assert_allclose(
    locomotion.last_gait_inputs["swing_states"], first_start_swing
  )


def test_stand_mode_backend_requires_explicit_foot_contacts() -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
  )
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)

  with pytest.raises(ValueError, match="foot contacts"):
    backend.step(dof_states, body_states, command, offsets, None)
  assert backend.gait_mode == MpcGaitMode.STAND


def test_stand_settling_does_not_destroy_recoverable_swing_state() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  trajectory = locomotion.footSwingTrajectories[0]
  initial = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
  final = np.array([[0.4], [0.5], [0.6]], dtype=np.float32)
  trajectory.setInitialPosition(initial)
  trajectory.setFinalPosition(final)
  locomotion.firstSwing[0] = False
  locomotion._latched_foot_placement_offsets[0] = (0.04, -0.02)

  locomotion._beginStandSettling()
  locomotion._cancelStandSettling()

  np.testing.assert_array_equal(trajectory._p0, initial)
  np.testing.assert_array_equal(trajectory._pf, final)
  np.testing.assert_allclose(
    locomotion._latched_foot_placement_offsets[0], (0.04, -0.02)
  )
  assert not locomotion.firstSwing[0]
  assert not locomotion._stand_settling


def test_stand_with_move_command_and_missing_support_recovers_before_trot() -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_contact_hold_time=0.01,
  )
  locomotion = backend._runner.cMPC
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)
  all_contacts = np.ones(4, dtype=bool)

  backend.step(dof_states, body_states, command, offsets, all_contacts)
  assert backend.stand_ready

  command[0] = 0.20
  contacts = np.array([False, True, True, True])
  backend.step(dof_states, body_states, command, offsets, contacts)
  assert backend.gait_mode == MpcGaitMode.STOPPING
  assert not backend.stand_ready
  np.testing.assert_array_equal(
    locomotion._stopping_swing_legs,
    np.array([True, False, False, False]),
  )


def test_initial_move_command_arms_four_foot_support_before_trot() -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_contact_hold_time=0.03,
  )
  dof_states, body_states, command = _nominal_inputs()
  command[1] = 0.15
  offsets = np.zeros((4, 2), dtype=np.float32)
  contacts = np.ones(4, dtype=bool)

  for _ in range(2):
    backend.step(dof_states, body_states, command, offsets, contacts)
    assert backend.gait_mode == MpcGaitMode.STAND
    assert not backend.stand_ready

  backend.step(dof_states, body_states, command, offsets, contacts)

  assert backend.gait_mode == MpcGaitMode.TROT


def test_stopping_preserves_active_latch_but_stand_clears_all_offsets() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.TROT
  locomotion._pending_foot_placement_offsets.fill(0.02)
  locomotion._latched_foot_placement_offsets.fill(0.04)

  locomotion._transitionToMode(mode_type.STOPPING)
  assert not np.any(locomotion._pending_foot_placement_offsets)
  np.testing.assert_allclose(locomotion._latched_foot_placement_offsets, 0.04)
  locomotion.setFootPlacementOffsets(np.full((4, 2), -0.08))
  np.testing.assert_allclose(locomotion._latched_foot_placement_offsets, 0.04)

  locomotion._transitionToMode(mode_type.STAND)
  assert not np.any(locomotion._pending_foot_placement_offsets)
  assert not np.any(locomotion._latched_foot_placement_offsets)


def test_trot_to_stopping_debounces_one_frame_loss_of_scheduled_support() -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_contact_loss_time=0.03,
  )
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.TROT
  locomotion.last_gait_inputs = {
    "swing_states": np.array([[0.0], [0.2], [0.2], [0.0]], dtype=np.float32)
  }

  # A single simulator frame can report no contacts exactly when the stop
  # command matures. The scheduled FL/HR support pair must enter the normal
  # contact-loss debounce instead of becoming zero-force swing legs at once.
  locomotion.setFootContacts(np.zeros(4, dtype=bool))
  locomotion._transitionToMode(mode_type.STOPPING)

  np.testing.assert_array_equal(
    locomotion._stopping_contacts,
    np.array([True, False, False, True]),
  )
  np.testing.assert_array_equal(
    locomotion._stopping_swing_legs,
    np.array([False, True, True, False]),
  )
  locomotion.stopping.configure(
    locomotion._stopping_swing_legs,
    locomotion._stopping_swing_progress,
    locomotion._stopping_contacts,
  )
  np.testing.assert_array_equal(
    locomotion.stopping.getContactState().reshape(4),
    np.array([1.0, 0.0, 0.0, 1.0]),
  )

  locomotion._updateStoppingContacts()
  np.testing.assert_array_equal(
    locomotion._stopping_contacts,
    np.array([True, False, False, True]),
  )
  locomotion.setFootContacts(np.array([True, False, False, True]))
  locomotion._updateStoppingContacts()
  np.testing.assert_array_equal(locomotion._stopping_contact_off_counters, 0)


def test_stand_to_stopping_preserves_confirmed_raw_contact_loss() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.STAND
  locomotion.last_gait_inputs = {
    # The all-stance schedule must not revive FL after STAND's 50 ms loss
    # confirmation has already selected the recovery transition.
    "swing_states": np.zeros((4, 1), dtype=np.float32)
  }
  measured = np.array([False, True, True, True])
  locomotion.setFootContacts(measured)

  locomotion._transitionToMode(mode_type.STOPPING)

  np.testing.assert_array_equal(locomotion._stopping_contacts, measured)
  np.testing.assert_array_equal(
    locomotion._stopping_swing_legs,
    np.array([True, False, False, False]),
  )
  assert locomotion._stopping_swing_progress[0] > 0.0


def test_stopping_debounces_each_stance_foot_contact_loss(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_contact_loss_time=0.03,
  )
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.TROT
  locomotion.last_gait_inputs = {
    "swing_states": np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
  }
  locomotion.setFootContacts(np.array([False, True, True, True]))
  locomotion._transitionToMode(mode_type.STOPPING)
  monkeypatch.setattr(locomotion, "solveDenseMPC", lambda *_: None)
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)
  contacts = np.array([False, False, True, True])

  assert locomotion._stand_contact_loss_ticks == 3
  for _ in range(locomotion._stand_contact_loss_ticks - 1):
    backend.step(dof_states, body_states, command, offsets, contacts)
    assert locomotion._stopping_swing_legs[0]
    assert not locomotion._stopping_swing_legs[1]

  backend.step(dof_states, body_states, command, offsets, contacts)
  assert locomotion._stopping_swing_legs[1]
  assert locomotion._stopping_swing_progress[1] > 0.0


def test_stopping_contact_schedule_change_forces_same_tick_mpc_solve(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(
    enable_stand_mode=True,
    stand_contact_hold_time=0.02,
  )
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.TROT
  locomotion.last_gait_inputs = {
    "swing_states": np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
  }
  airborne_contacts = np.array([False, True, True, True])
  locomotion.setFootContacts(airborne_contacts)
  locomotion._transitionToMode(mode_type.STOPPING)
  solve_iterations: list[int] = []
  solved_tables: list[np.ndarray] = []

  def fake_solve_dense_mpc(mpc_table: list[int], data: object) -> None:
    del data
    solve_iterations.append(locomotion.iterationCounter)
    solved_tables.append(np.asarray(mpc_table).reshape(10, 4).copy())

  monkeypatch.setattr(locomotion, "solveDenseMPC", fake_solve_dense_mpc)
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)

  backend.step(dof_states, body_states, command, offsets, airborne_contacts)
  assert solve_iterations == [1]
  assert solved_tables[-1][0, 0] == 0

  moving_body = body_states.copy()
  moving_body[7] = 0.20
  all_contacts = np.ones(4, dtype=bool)
  backend.step(dof_states, moving_body, command, offsets, all_contacts)
  assert solve_iterations == [1, 2]
  np.testing.assert_array_equal(solved_tables[-1][0], np.ones(4))


def test_stopping_touchdown_search_is_bounded(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  mode_type = type(locomotion.gait_mode)
  locomotion.gait_mode = mode_type.TROT
  locomotion.last_gait_inputs = {
    "swing_states": np.array([[1.0], [0.0], [0.0], [0.0]], dtype=np.float32)
  }
  locomotion.setFootContacts(np.array([False, True, True, True]))
  locomotion._transitionToMode(mode_type.STOPPING)
  monkeypatch.setattr(locomotion, "solveDenseMPC", lambda *_: None)
  dof_states, body_states, command = _nominal_inputs()
  offsets = np.zeros((4, 2), dtype=np.float32)
  contacts = np.array([False, True, True, True])

  for _ in range(250):
    backend.step(dof_states, body_states, command, offsets, contacts)

  assert locomotion._stopping_search_depth[0] == pytest.approx(0.08)
  assert np.all(locomotion._stopping_search_depth <= 0.08)


def test_stand_latches_world_targets_and_uses_internal_hipx_hold() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC
  dof_states, body_states, command = _nominal_inputs()
  contacts = np.ones(4, dtype=bool)

  backend.step(dof_states, body_states, command, np.zeros((4, 2)), contacts)
  held_feet = locomotion._stand_foot_hold_targets_world.copy()
  held_com_xy = locomotion._stand_com_hold_xy.copy()

  assert locomotion._stand_foot_targets_valid
  assert locomotion._stand_com_target_valid
  for leg_command in backend._runner._legController.commands:
    np.testing.assert_allclose(np.diag(leg_command.kpCartesian), (80.0, 80.0, 30.0))
    np.testing.assert_allclose(np.diag(leg_command.kpJoint), (12.0, 0.0, 0.0))
    np.testing.assert_allclose(np.diag(leg_command.kdJoint), (1.0, 0.2, 0.2))

  moved_body = body_states.copy()
  moved_body[0:2] = (0.04, -0.03)
  backend.step(dof_states, moved_body, command, np.zeros((4, 2)), contacts)

  np.testing.assert_array_equal(locomotion._stand_foot_hold_targets_world, held_feet)
  np.testing.assert_array_equal(locomotion._stand_com_hold_xy, held_com_xy)
  assert locomotion.last_mpc_inputs is not None
  np.testing.assert_allclose(
    locomotion.last_mpc_inputs["weights"]["qp_weights"][[3, 4, 9, 10]],
    (5.0, 5.0, 3.0, 3.0),
  )
  np.testing.assert_array_equal(
    locomotion.last_mpc_inputs["desired_state"]["desired_com_position"][:2],
    held_com_xy,
  )


def test_invalid_solver_output_uses_finite_fallback_and_is_counted() -> None:
  backend = _make_controller_or_skip(enable_stand_mode=True)
  locomotion = backend._runner.cMPC

  class FakeSolver:
    def __init__(self) -> None:
      self.outputs = [np.arange(12, dtype=np.float32), np.empty(0)]
      self.reset_count = 0

    def compute_contact_forces(self, *args: object) -> np.ndarray:
      del args
      return self.outputs.pop(0)

    def reset_solver(self) -> None:
      self.reset_count += 1

  solver = FakeSolver()
  locomotion._cpp_mpc = solver
  locomotion._body_height = backend._runner._quadruped._bodyHeight
  table = np.ones(locomotion.horizonLength * 4, dtype=np.float32)

  locomotion.solveDenseMPC(table, backend._runner.data)
  valid_force = locomotion.f_ff.copy()
  locomotion.solveDenseMPC(table, backend._runner.data)

  np.testing.assert_array_equal(locomotion.f_ff, valid_force)
  assert np.all(np.isfinite(locomotion.f_ff))
  assert locomotion.mpc_solver_failure_count == 1
  assert locomotion.mpc_solver_consecutive_failures == 1
  assert solver.reset_count == 1
  assert locomotion.last_mpc_inputs is not None
  assert locomotion.last_mpc_inputs["solver"]["used_fallback"]
  assert (
    locomotion.last_mpc_inputs["solver"]["fallback"]
    == "last_valid_same_contact_schedule"
  )


def test_controller_output_is_equivariant_to_absolute_yaw() -> None:
  """The controller's internal XY frame and residuals are yaw-aligned."""
  yaw_zero = _make_controller_or_skip()
  yaw_quarter_turn = _make_controller_or_skip()
  dof_states, body_zero, command = _nominal_inputs()
  body_quarter_turn = body_zero.copy()
  body_quarter_turn[5] = math.sin(math.pi / 4.0)
  body_quarter_turn[6] = math.cos(math.pi / 4.0)
  command[:] = (0.2, 0.1, 0.2)
  offsets = np.array(
    [[0.04, 0.01], [-0.02, 0.03], [0.01, -0.04], [-0.03, -0.02]],
    dtype=np.float32,
  )

  # Include multiple MPC/gait updates and both swing pairs. Absolute world yaw
  # must not change body-frame torque output for otherwise identical states.
  for _ in range(60):
    torque_zero = yaw_zero.step(dof_states, body_zero, command, offsets)
    torque_quarter_turn = yaw_quarter_turn.step(
      dof_states, body_quarter_turn, command, offsets
    )
    # OSQP solves the two rotated sparse systems independently at a 1e-3
    # residual tolerance. The body-frame outputs should agree to a few mNm,
    # rather than bit-level equality in world coordinates.
    np.testing.assert_allclose(torque_quarter_turn, torque_zero, atol=5e-3, rtol=2e-3)

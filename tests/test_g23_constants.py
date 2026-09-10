"""Tests for the SyncAI G23 asset."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.syncai_g23 import g23_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr

_JOINT_NAMES = (
  "FL_HipX_joint",
  "FL_HipY_joint",
  "FL_Knee_joint",
  "FR_HipX_joint",
  "FR_HipY_joint",
  "FR_Knee_joint",
  "HL_HipX_joint",
  "HL_HipY_joint",
  "HL_Knee_joint",
  "HR_HipX_joint",
  "HR_HipY_joint",
  "HR_Knee_joint",
)


@pytest.fixture(scope="module")
def g23_entity() -> Entity:
  return Entity(g23_constants.get_g23_robot_cfg())


@pytest.fixture(scope="module")
def g23_model(g23_entity: Entity) -> mujoco.MjModel:
  return g23_entity.spec.compile()


@pytest.fixture(scope="module")
def g23_rl_mpc_entity() -> Entity:
  return Entity(g23_constants.get_g23_rl_mpc_robot_cfg())


@pytest.fixture(scope="module")
def g23_rl_mpc_model(g23_rl_mpc_entity: Entity) -> mujoco.MjModel:
  return g23_rl_mpc_entity.spec.compile()


def test_g23_entity_contract(g23_entity: Entity, g23_model: mujoco.MjModel) -> None:
  assert g23_entity.joint_names == _JOINT_NAMES
  assert g23_entity.actuator_names == _JOINT_NAMES
  assert g23_model.nq == 19
  assert g23_model.nv == 18
  assert g23_model.nu == 12

  sensor_names = {g23_model.sensor(i).name for i in range(g23_model.nsensor)}
  assert {"imu_ang_vel", "imu_lin_vel", "imu_lin_acc"} <= sensor_names

  for site_name in g23_constants.G23_FOOT_SITE_NAMES:
    assert site_name in g23_entity.site_names
    assert g23_model.site(site_name).id >= 0


def test_g23_joint_dynamics_match_source_urdf(g23_model: mujoco.MjModel) -> None:
  for joint_name in _JOINT_NAMES:
    dof_idx = g23_model.joint(joint_name).dofadr[0]
    assert g23_model.dof_damping[dof_idx] == pytest.approx(0.0)
    assert g23_model.dof_armature[dof_idx] == pytest.approx(0.0)


def test_g23_keyframe_joint_positions(
  g23_entity: Entity, g23_model: mujoco.MjModel
) -> None:
  key = g23_model.key("init_state")
  expected_joint_pos = g23_constants.HOME_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, g23_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    g23_entity.joint_names, expected_values, strict=True
  ):
    joint = g23_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_g23_foot_collision_geoms(g23_model: mujoco.MjModel) -> None:
  foot_pattern = r"^(FL|FR|HL|HR)_foot_collision$"
  for i in range(g23_model.ngeom):
    geom = g23_model.geom(i)
    if re.match(foot_pattern, geom.name):
      assert geom.condim == 6
      assert geom.priority == 1
      assert geom.friction[0] == 1.0


def test_g23_all_collision_geoms_enabled(g23_model: mujoco.MjModel) -> None:
  collision_geoms = [
    g23_model.geom(i)
    for i in range(g23_model.ngeom)
    if "_collision" in g23_model.geom(i).name
  ]
  assert len(collision_geoms) == 22
  for geom in collision_geoms:
    assert geom.contype == 1, f"{geom.name} lost its contype"
    assert geom.conaffinity == 1, f"{geom.name} lost its conaffinity"


def test_g23_rl_mpc_uses_torque_actuators(
  g23_rl_mpc_entity: Entity, g23_rl_mpc_model: mujoco.MjModel
) -> None:
  assert g23_rl_mpc_entity.joint_names == _JOINT_NAMES
  assert g23_rl_mpc_entity.actuator_names == _JOINT_NAMES
  assert g23_rl_mpc_model.nu == 12

  for joint_name in _JOINT_NAMES:
    actuator = g23_rl_mpc_model.actuator(joint_name)
    assert actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT
    expected_limit = 36.0 if "Knee" in joint_name else 24.0
    np.testing.assert_allclose(
      actuator.ctrlrange,
      (-expected_limit, expected_limit),
      rtol=0.0,
      atol=1e-6,
    )


def test_g23_rl_mpc_initial_stance(g23_rl_mpc_model: mujoco.MjModel) -> None:
  key = g23_rl_mpc_model.key("init_state")
  for joint_name in _JOINT_NAMES:
    expected = 0.0
    if "HipY" in joint_name:
      expected = -0.8
    elif "Knee" in joint_name:
      expected = 1.6
    assert key.qpos[g23_rl_mpc_model.joint(joint_name).qposadr[0]] == pytest.approx(
      expected
    )

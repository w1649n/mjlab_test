"""Kinematic action terms for the navigation task.

Both the robot and the pedestrians move by writing joint positions directly, so
the physics step only has to refresh kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.navigation import assets
from mjlab.tasks.navigation.mdp.hsfm import HsfmParams, hsfm_step, wrap_angle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class UnicycleActionCfg(ActionTermCfg):
  """Drive a planar (x, y, yaw) base with linear and angular velocity commands.

  The raw action is ``(linear, angular)``. The linear part maps ``[-1, 1]`` to
  ``[0, max_lin_vel]`` and the angular part maps ``[-1, 1]`` to
  ``[-max_ang_vel, max_ang_vel]``; both are clipped to those ranges.
  """

  max_lin_vel: float = 1.0
  max_ang_vel: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> UnicycleAction:
    return UnicycleAction(self, env)


class UnicycleAction(ActionTerm):
  cfg: UnicycleActionCfg
  _entity: Entity

  def __init__(self, cfg: UnicycleActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._raw_action = torch.zeros(self.num_envs, 2, device=self.device)
    self.lin_vel = torch.zeros(self.num_envs, device=self.device)
    self.ang_vel = torch.zeros(self.num_envs, device=self.device)
    self.prev_ang_vel = torch.zeros(self.num_envs, device=self.device)
    self._joint_vel = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def action_dim(self) -> int:
    return 2

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_action[:] = actions
    self.prev_ang_vel[:] = self.ang_vel
    self.lin_vel[:] = (0.5 * (actions[:, 0] + 1.0)).clamp(0.0, 1.0)
    self.lin_vel *= self.cfg.max_lin_vel
    self.ang_vel[:] = actions[:, 1].clamp(-1.0, 1.0) * self.cfg.max_ang_vel

  def apply_actions(self) -> None:
    dt = self._env.physics_dt
    pose = self._entity.data.joint_pos
    x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    new_yaw = yaw + self.ang_vel * dt
    # Exact arc for a turning base, straight line otherwise.
    turning = self.ang_vel.abs() > 1e-3
    radius = self.lin_vel / torch.where(turning, self.ang_vel, 1.0)
    dx = torch.where(
      turning,
      radius * (torch.sin(new_yaw) - torch.sin(yaw)),
      self.lin_vel * torch.cos(yaw) * dt,
    )
    dy = torch.where(
      turning,
      radius * (torch.cos(yaw) - torch.cos(new_yaw)),
      self.lin_vel * torch.sin(yaw) * dt,
    )
    new_pose = torch.stack([x + dx, y + dy, wrap_angle(new_yaw)], dim=-1)
    self._entity.write_joint_state_to_sim(new_pose, self._joint_vel)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self.lin_vel[env_ids] = 0.0
    self.ang_vel[env_ids] = 0.0
    self.prev_ang_vel[env_ids] = 0.0

  @property
  def world_velocity(self) -> torch.Tensor:
    """[B, 2] commanded velocity of the base in the world frame."""
    yaw = self._entity.data.joint_pos[:, 2]
    return self.lin_vel[:, None] * torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


@dataclass(kw_only=True)
class PedestrianActionCfg(ActionTermCfg):
  """Move pedestrians with the Headed Social Force Model.

  This term consumes no policy output. It lives in the action manager so that
  the pedestrians advance together with the robot on every physics step.
  """

  hsfm: HsfmParams = field(default_factory=HsfmParams)
  hsfm_dt: float = 0.025
  """Integration step of the pedestrian model in seconds."""
  goal_tolerance: float = 0.3
  """Distance at which a pedestrian turns around towards its other goal."""
  react_to_robot: bool = False
  """Whether pedestrians avoid the robot. The reference environment trains with
  pedestrians that ignore the robot and evaluates with pedestrians that avoid
  it."""
  robot_name: str = "robot"
  robot_action_name: str = "base_velocity"
  robot_radius: float = assets.ROBOT_RADIUS

  def build(self, env: ManagerBasedRlEnv) -> PedestrianAction:
    return PedestrianAction(self, env)


class PedestrianAction(ActionTerm):
  cfg: PedestrianActionCfg
  _entity: Entity

  def __init__(self, cfg: PedestrianActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    num_humans = assets.MAX_HUMANS
    self._raw_action = torch.zeros(self.num_envs, 0, device=self.device)
    self._boxes = torch.tensor(
      assets.load_hospital_map().boxes[:, :4], dtype=torch.float, device=self.device
    )
    self._parking = torch.tensor(
      assets.parking_positions(), dtype=torch.float, device=self.device
    )
    self.state = torch.zeros(self.num_envs, num_humans, 6, device=self.device)
    self.state[..., 0:2] = self._parking
    self.goals = self._parking[None, :, None, :].repeat(self.num_envs, 1, 2, 1)
    self.goal_index = torch.zeros(
      self.num_envs, num_humans, dtype=torch.long, device=self.device
    )
    self.active = torch.zeros(
      self.num_envs, num_humans, dtype=torch.bool, device=self.device
    )
    self._joint_vel = torch.zeros(self.num_envs, num_humans * 3, device=self.device)
    self._substeps = max(1, round(env.physics_dt / cfg.hsfm_dt))

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions

  def set_scenario(
    self,
    env_ids: torch.Tensor,
    pos: torch.Tensor,
    yaw: torch.Tensor,
    goal: torch.Tensor,
    active: torch.Tensor,
  ) -> None:
    """Place pedestrians at the start of an episode.

    Args:
      env_ids: Environments to update.
      pos: [N, H, 2] start positions.
      yaw: [N, H] start headings.
      goal: [N, H, 2] goals. Pedestrians walk back and forth between their
        start position and their goal.
      active: [N, H] pedestrians used by the scenario. The rest are parked.
    """
    pos = torch.where(active[..., None], pos, self._parking)
    state = torch.zeros(len(env_ids), assets.MAX_HUMANS, 6, device=self.device)
    state[..., 0:2] = pos
    state[..., 4] = yaw
    self.state[env_ids] = state
    self.goals[env_ids] = torch.stack([pos, goal], dim=2)
    # Head for the far goal first.
    self.goal_index[env_ids] = 1
    self.active[env_ids] = active
    self._write_to_sim(env_ids)

  def apply_actions(self) -> None:
    robot_pos = robot_vel = None
    if self.cfg.react_to_robot:
      robot: Entity = self._env.scene[self.cfg.robot_name]
      robot_action = self._env.action_manager.get_term(self.cfg.robot_action_name)
      assert isinstance(robot_action, UnicycleAction)
      robot_pos = robot.data.joint_pos[:, 0:2]
      robot_vel = robot_action.world_velocity
    dt = self._env.physics_dt / self._substeps
    for _ in range(self._substeps):
      goal = self._current_goal()
      reached = torch.linalg.norm(goal - self.state[..., 0:2], dim=-1)
      reached = (reached < self.cfg.goal_tolerance) & self.active
      self.goal_index[reached] = 1 - self.goal_index[reached]
      self.state = hsfm_step(
        self.state,
        self._current_goal(),
        self.active,
        self._boxes,
        self.cfg.hsfm,
        dt,
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_radius=self.cfg.robot_radius,
      )
    self._write_to_sim(slice(None))

  def _current_goal(self) -> torch.Tensor:
    index = self.goal_index[..., None, None].expand(-1, -1, 1, 2)
    return self.goals.gather(2, index).squeeze(2)

  def _write_to_sim(self, env_ids: torch.Tensor | slice) -> None:
    state = self.state[env_ids]
    joint_pos = torch.cat([state[..., 0:2], state[..., 4:5]], dim=-1).flatten(1)
    self._entity.write_joint_state_to_sim(
      joint_pos, self._joint_vel[env_ids], env_ids=env_ids
    )

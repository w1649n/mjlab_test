"""Goal command for the navigation task.

Each episode replays one layout from a bank of pre-sampled scenarios: a robot
start pose, a goal and the start poses and goals of the pedestrians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.navigation import assets
from mjlab.tasks.navigation.mdp.actions import PedestrianAction
from mjlab.tasks.navigation.mdp.hsfm import wrap_angle

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_NEVER = 1.0e9


@dataclass(kw_only=True)
class NavigationCommandCfg(CommandTermCfg):
  """Sample a scenario on reset and expose the goal in the robot frame."""

  entity_name: str = "robot"
  scenarios: tuple[str, ...]
  """Names of the scenarios in the bank that episodes are drawn from."""
  pedestrian_action_name: str | None = None
  """Action term that owns the pedestrians, or None for an empty hospital."""
  adaptive_sampling: bool = True
  """Sample scenarios in proportion to their failure rate, as the reference
  environment does while training. When False, sample uniformly."""
  success_term_name: str = "goal_reached"
  """Termination term that marks a successful episode."""
  resampling_time_range: tuple[float, float] = (_NEVER, _NEVER)
  goal_radius: float = 0.2
  goal_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.8)

  def build(self, env: ManagerBasedRlEnv) -> NavigationCommand:
    return NavigationCommand(self, env)


class NavigationCommand(CommandTerm):
  cfg: NavigationCommandCfg

  def __init__(self, cfg: NavigationCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    bank = np.load(assets.DATA_DIR / "scenarios.npz")
    names = [str(n) for n in bank["names"]]
    unknown = sorted(set(cfg.scenarios) - set(names))
    if unknown:
      raise ValueError(f"Unknown scenarios {unknown}. Available: {names}")
    rows = [names.index(n) for n in cfg.scenarios]

    def load(key: str) -> torch.Tensor:
      return torch.as_tensor(bank[key][rows], device=self.device)

    self._robot_start = load("robot_start")  # [S, K, 3]
    self._goal = load("goal")  # [S, K, 2]
    self._human_pos = load("human_pos")  # [S, K, H, 2]
    self._human_yaw = load("human_yaw")  # [S, K, H]
    self._human_goal = load("human_goal")  # [S, K, H, 2]
    self._human_active = load("human_active")  # [S, K, H]
    self._num_scenarios, self._num_samples = self._goal.shape[:2]

    self.goal = torch.zeros(self.num_envs, 2, device=self.device)
    self.prev_distance = torch.zeros(self.num_envs, device=self.device)
    self.scenario_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._command = torch.zeros(self.num_envs, 2, device=self.device)
    self._joint_vel = torch.zeros(self.num_envs, 3, device=self.device)
    # Start at one attempt so the failure rate is defined from the first reset.
    self._attempts = torch.ones(self._num_scenarios, device=self.device)
    self._successes = torch.zeros(self._num_scenarios, device=self.device)
    self._has_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """[B, 2] distance to the goal and its bearing in the robot frame."""
    return self._command

  def goal_polar(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Distance and robot-frame bearing of the goal from the current pose."""
    pose = self.robot.data.joint_pos
    delta = self.goal - pose[:, 0:2]
    distance = torch.linalg.norm(delta, dim=-1)
    bearing = wrap_angle(torch.atan2(delta[:, 1], delta[:, 0]) - pose[:, 2])
    return distance, bearing

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    extras: dict[str, float] = {}
    finished = env_ids[self._has_episode[env_ids]]
    if len(finished) > 0:
      success = self._env.termination_manager.get_term(self.cfg.success_term_name)
      success = success[finished].float()
      ids = self.scenario_ids[finished]
      self._attempts.index_add_(0, ids, torch.ones_like(success))
      self._successes.index_add_(0, ids, success)
      extras["success_rate"] = success.mean().item()
    self._has_episode[env_ids] = True
    extras.update(super().reset(env_ids))
    return extras

  def _update_metrics(self) -> None:
    self.metrics["goal_distance"][:] = self.goal_polar()[0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.adaptive_sampling:
      weights = (1.0 - self._successes / self._attempts).clamp(min=0.01)
    else:
      weights = torch.ones(self._num_scenarios, device=self.device)
    scenario = torch.multinomial(weights, len(env_ids), replacement=True)
    sample = torch.randint(self._num_samples, (len(env_ids),), device=self.device)
    self.scenario_ids[env_ids] = scenario

    start = self._robot_start[scenario, sample]
    self.goal[env_ids] = self._goal[scenario, sample]
    self.prev_distance[env_ids] = torch.linalg.norm(
      self.goal[env_ids] - start[:, 0:2], dim=-1
    )
    self.robot.write_joint_state_to_sim(
      start, self._joint_vel[env_ids], env_ids=env_ids
    )

    if self.cfg.pedestrian_action_name is not None:
      pedestrians = self._env.action_manager.get_term(self.cfg.pedestrian_action_name)
      assert isinstance(pedestrians, PedestrianAction)
      pedestrians.set_scenario(
        env_ids,
        pos=self._human_pos[scenario, sample],
        yaw=self._human_yaw[scenario, sample],
        goal=self._human_goal[scenario, sample],
        active=self._human_active[scenario, sample],
      )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    distance, bearing = self.goal_polar()
    self._command[:, 0] = distance
    self._command[:, 1] = bearing
    # Rewards for the step have been computed by now, so this becomes the
    # reference distance for the next step's progress reward.
    self.prev_distance[:] = distance

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for i in visualizer.get_env_indices(self.num_envs):
      goal = self.goal[i].cpu().numpy()
      visualizer.add_sphere(
        center=np.array([goal[0], goal[1], assets.LIDAR_HEIGHT]),
        radius=self.cfg.goal_radius,
        color=self.cfg.goal_color,
        label="goal",
      )

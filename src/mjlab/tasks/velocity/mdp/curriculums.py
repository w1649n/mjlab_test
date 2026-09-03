from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from typing_extensions import NotRequired

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .velocity_command import (
  UniformVelocityCommandCfg,
  _validate_directional_bucket_fractions,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: NotRequired[tuple[float, float] | None]
  lin_vel_y: NotRequired[tuple[float, float] | None]
  ang_vel_z: NotRequired[tuple[float, float] | None]
  rel_backward_envs: NotRequired[float]


# 2026-09-02 stair-training update: every performance stage is a complete
# command snapshot.  Reapplying any stage after checkpoint restore therefore
# does not depend on mutations made by earlier stages.
class PerformanceVelocityStage(TypedDict):
  min_terrain_level: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]
  rel_forward_envs: float
  rel_backward_envs: float


class _TerrainPerformance(TypedDict):
  episode_count: int
  safe_success_count: int
  forward_tracking_sum: float
  forward_tracking_count: int
  backward_tracking_sum: float
  backward_tracking_count: int


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
      if "rel_backward_envs" in stage:
        rel_backward_envs = stage["rel_backward_envs"]
        _validate_directional_bucket_fractions(cfg.rel_forward_envs, rel_backward_envs)
        cfg.rel_backward_envs = rel_backward_envs
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "rel_backward_envs": torch.tensor(cfg.rel_backward_envs),
  }


# 2026-09-02 stair-training update: replace wall-clock-only G23 command
# progression with a checkpointable gate driven by completed stair episodes.
class PerformanceGatedVelocityCurriculum:
  """Advance complete velocity-command stages after sustained stair performance.

  The term must be placed before ``terrain_levels_vel`` in the curriculum mapping:
  it reads the just-finished episode's terrain origin and difficulty before that
  term moves the environment to its next patch.  Directional tracking statistics
  are accumulated by :class:`UniformVelocityCommand` over every command segment.

  ``stages[0]`` is applied at construction.  To enter each later stage, both the
  safe traversal rate and forward/backward tracking scores must pass for every
  configured terrain type for ``required_passes`` independent windows.  At most
  one stage is entered per call.
  """

  _STATE_VERSION = 1
  _WINDOW_FIELDS = (
    "episode_count",
    "safe_success_count",
    "forward_tracking_sum",
    "forward_tracking_count",
    "backward_tracking_sum",
    "backward_tracking_count",
  )

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self._env = env
    self._command_name = str(params["command_name"])
    self._stages = deepcopy(cast(list[PerformanceVelocityStage], params["stages"]))
    self._terrain_names = tuple(str(name) for name in params["terrain_names"])
    self._min_episodes_per_terrain = int(params["min_episodes_per_terrain"])
    self._min_directional_samples_per_terrain = int(
      params["min_directional_samples_per_terrain"]
    )
    self._min_safe_success_rate = float(params["min_safe_success_rate"])
    self._min_directional_tracking_score = float(
      params["min_directional_tracking_score"]
    )
    self._warmup_steps = int(params.get("warmup_steps", 0))
    self._min_stage_steps = int(params.get("min_stage_steps", 0))
    self._required_passes = int(params.get("required_passes", 1))
    self._success_termination_name = str(
      params.get("success_termination_name", "time_out")
    )
    self._asset_cfg = cast(SceneEntityCfg, params.get("asset_cfg", _DEFAULT_SCENE_CFG))

    # 2026-09-02 anti-tripod update: optionally reject completed episodes whose
    # terminal state still has a foot that has been continuously airborne too long.
    contact_sensor_name = params.get("contact_sensor_name")
    max_foot_air_time = params.get("max_foot_air_time")
    if (contact_sensor_name is None) != (max_foot_air_time is None):
      raise ValueError(
        "contact_sensor_name and max_foot_air_time must be configured together."
      )
    self._contact_sensor: ContactSensor | None = None
    self._max_foot_air_time: float | None = None
    if contact_sensor_name is not None:
      if not isinstance(contact_sensor_name, str) or not contact_sensor_name:
        raise TypeError("contact_sensor_name must be a non-empty string.")
      if (
        not isinstance(max_foot_air_time, (int, float))
        or isinstance(max_foot_air_time, bool)
        or not math.isfinite(max_foot_air_time)
        or max_foot_air_time <= 0.0
      ):
        raise ValueError("max_foot_air_time must be a positive finite number.")
      contact_sensor = env.scene[contact_sensor_name]
      if not isinstance(contact_sensor, ContactSensor):
        raise TypeError(
          f"Sensor '{contact_sensor_name}' must be a ContactSensor for foot health."
        )
      if contact_sensor.data.current_air_time is None:
        raise ValueError(
          f"Sensor '{contact_sensor_name}' must enable track_air_time for foot health."
        )
      self._contact_sensor = contact_sensor
      self._max_foot_air_time = float(max_foot_air_time)

    command_term = env.command_manager.get_term(self._command_name)
    if command_term is None or not isinstance(
      command_term.cfg, UniformVelocityCommandCfg
    ):
      raise TypeError(
        f"Command '{self._command_name}' must use UniformVelocityCommandCfg."
      )
    directional_stats = getattr(command_term, "get_directional_tracking_stats", None)
    if not callable(directional_stats):
      raise TypeError(
        f"Command '{self._command_name}' does not expose directional tracking stats."
      )
    self._directional_stats = cast(
      Callable[[torch.Tensor], dict[str, torch.Tensor]], directional_stats
    )
    refresh_command = getattr(command_term, "resample_after_curriculum_restore", None)
    if not callable(refresh_command):
      raise TypeError(
        f"Command '{self._command_name}' cannot refresh commands after resume."
      )
    self._refresh_command = cast(Callable[[], None], refresh_command)
    self._command_cfg = command_term.cfg

    self._asset: Entity = env.scene[self._asset_cfg.name]
    terrain = env.scene.terrain
    if terrain is None:
      raise ValueError(
        "Performance-gated velocity curriculum requires initialized generated terrain."
      )
    self._terrain = terrain
    terrain_origins = getattr(self._terrain, "terrain_origins", None)
    terrain_levels = getattr(self._terrain, "terrain_levels", None)
    terrain_types = getattr(self._terrain, "terrain_types", None)
    terrain_env_origins = getattr(self._terrain, "env_origins", None)
    if not all(
      isinstance(value, torch.Tensor)
      for value in (
        terrain_origins,
        terrain_levels,
        terrain_types,
        terrain_env_origins,
      )
    ):
      raise ValueError(
        "Performance-gated velocity curriculum requires initialized generated terrain."
      )
    self._terrain_origins = cast(torch.Tensor, terrain_origins)
    self._terrain_levels = cast(torch.Tensor, terrain_levels)
    self._terrain_types = cast(torch.Tensor, terrain_types)
    self._terrain_env_origins = cast(torch.Tensor, terrain_env_origins)
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
      raise ValueError(
        "Performance-gated velocity curriculum requires a terrain generator."
      )
    self._terrain_generator = terrain_generator
    terrain_type_names = tuple(terrain_generator.sub_terrains)
    if self._terrain_origins.shape[1] != len(terrain_type_names):
      raise ValueError(
        "Performance-gated velocity curriculum requires one terrain column per type."
      )
    self._terrain_type_ids: dict[str, int] = {}
    for terrain_name in self._terrain_names:
      if terrain_name not in terrain_type_names:
        raise ValueError(f"Unknown curriculum terrain name '{terrain_name}'.")
      self._terrain_type_ids[terrain_name] = terrain_type_names.index(terrain_name)

    self._validate_configuration()
    # Fail at environment construction instead of midway through training when
    # the configured successful termination term is misspelled or is not a timeout.
    env.termination_manager.get_term(self._success_termination_name)
    success_cfg = env.termination_manager.get_term_cfg(self._success_termination_name)
    if not success_cfg.time_out:
      raise ValueError(
        f"Success termination '{self._success_termination_name}' must be a timeout."
      )
    self._competing_termination_names = tuple(
      name
      for name in env.termination_manager.active_terms
      if name != self._success_termination_name
    )

    self._stage_index = 0
    self._stage_enter_step = int(env.common_step_counter)
    self._pass_streak = 0
    self._window = self._empty_window()
    self._last_window = self._empty_window()
    self._last_window_passed = False
    self._last_evaluation_step = -1
    # 2026-09-02 stair-training update: only episodes that started in the
    # current stage may contribute evidence for that stage.
    self._episode_stage_index = torch.zeros_like(self._terrain_levels)
    self._apply_stage(self._stage_index)

  @property
  def stage_index(self) -> int:
    return self._stage_index

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    **_: Any,
  ) -> dict[str, torch.Tensor]:
    if env is not self._env:
      raise ValueError("Curriculum term called with a different environment instance.")
    if env_ids.numel() == 0 or self._stage_index >= len(self._stages) - 1:
      return self._logging_state(env.common_step_counter)
    if env.common_step_counter < self._warmup_steps:
      return self._logging_state(env.common_step_counter)

    valid = env.episode_length_buf[env_ids] > 0
    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor):
      valid &= reset_buf[env_ids]
    completed_ids = env_ids[valid]
    if completed_ids.numel() == 0:
      return self._logging_state(env.common_step_counter)

    stage_completed_ids = completed_ids[
      self._episode_stage_index[completed_ids] == self._stage_index
    ]
    if stage_completed_ids.numel() > 0:
      self._accumulate_completed_episodes(env, stage_completed_ids)

    stage_age = env.common_step_counter - self._stage_enter_step
    if stage_age >= self._min_stage_steps and self._window_ready():
      passed = self._window_passes()
      self._last_window = deepcopy(self._window)
      self._last_window_passed = passed
      self._last_evaluation_step = int(env.common_step_counter)
      self._window = self._empty_window()
      self._pass_streak = self._pass_streak + 1 if passed else 0

      if self._pass_streak >= self._required_passes:
        self._stage_index += 1
        self._stage_enter_step = int(env.common_step_counter)
        self._pass_streak = 0
        self._apply_stage(self._stage_index)

    # Curriculum runs before command reset.  Label the episodes that are about
    # to start with the stage whose command configuration they will sample.
    self._episode_stage_index[completed_ids] = self._stage_index

    return self._logging_state(env.common_step_counter)

  def state_dict(self) -> dict[str, Any]:
    """Return portable state containing only completed-episode statistics."""
    return {
      "version": self._STATE_VERSION,
      "stage_count": len(self._stages),
      "terrain_names": self._terrain_names,
      "stage_index": self._stage_index,
      "stage_enter_step": self._stage_enter_step,
      "pass_streak": self._pass_streak,
      "window": deepcopy(self._window),
      "last_window": deepcopy(self._last_window),
      "last_window_passed": self._last_window_passed,
      "last_evaluation_step": self._last_evaluation_step,
    }

  def load_state_dict(self, state: Mapping[str, Any]) -> None:
    """Atomically restore gate progress and reapply the complete command stage."""
    if not isinstance(state, Mapping):
      raise TypeError("Performance curriculum state must be a mapping.")
    if state.get("version") != self._STATE_VERSION:
      raise ValueError("Performance curriculum state version does not match.")
    if state.get("stage_count") != len(self._stages):
      raise ValueError("Performance curriculum stage count does not match.")
    saved_terrain_names = state.get("terrain_names")
    if (
      not isinstance(saved_terrain_names, (tuple, list))
      or tuple(saved_terrain_names) != self._terrain_names
    ):
      raise ValueError("Performance curriculum terrain names do not match.")

    stage_index = self._checked_int(state, "stage_index", minimum=0)
    if stage_index >= len(self._stages):
      raise ValueError("Performance curriculum stage index is out of range.")
    stage_enter_step = self._checked_int(state, "stage_enter_step", minimum=0)
    pass_streak = self._checked_int(state, "pass_streak", minimum=0)
    if pass_streak >= self._required_passes:
      raise ValueError("Performance curriculum pass streak is out of range.")
    last_evaluation_step = self._checked_int(state, "last_evaluation_step", minimum=-1)
    last_window_passed = state.get("last_window_passed")
    if not isinstance(last_window_passed, bool):
      raise TypeError("Performance curriculum last_window_passed must be bool.")
    window = self._validated_window(state.get("window"), name="window")
    last_window = self._validated_window(state.get("last_window"), name="last_window")

    # Commit only after every field has validated.
    self._stage_index = stage_index
    self._stage_enter_step = stage_enter_step
    self._pass_streak = pass_streak
    self._window = window
    self._last_window = last_window
    self._last_window_passed = last_window_passed
    self._last_evaluation_step = last_evaluation_step
    self._apply_stage(self._stage_index)
    # 2026-09-02 stair-training update: the wrapper reset happened at stage 0;
    # refresh live commands so the first resumed rollout uses the restored stage.
    self._episode_stage_index.fill_(self._stage_index)
    self._refresh_command()

  def _validate_configuration(self) -> None:
    if not self._stages:
      raise ValueError("Performance velocity curriculum requires at least one stage.")
    if not self._terrain_names or len(set(self._terrain_names)) != len(
      self._terrain_names
    ):
      raise ValueError("terrain_names must be non-empty and unique.")
    if self._min_episodes_per_terrain <= 0:
      raise ValueError("min_episodes_per_terrain must be positive.")
    if self._min_directional_samples_per_terrain <= 0:
      raise ValueError("min_directional_samples_per_terrain must be positive.")
    for name, value in (
      ("min_safe_success_rate", self._min_safe_success_rate),
      ("min_directional_tracking_score", self._min_directional_tracking_score),
    ):
      if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1].")
    if self._warmup_steps < 0:
      raise ValueError("warmup_steps must be non-negative.")
    if self._min_stage_steps < 0:
      raise ValueError("min_stage_steps must be non-negative.")
    if self._required_passes <= 0:
      raise ValueError("required_passes must be positive.")

    max_terrain_level = int(self._terrain_origins.shape[0]) - 1
    previous_level = -1
    required_fields = set(PerformanceVelocityStage.__required_keys__)
    for index, stage in enumerate(self._stages):
      missing = required_fields - stage.keys()
      if missing:
        raise KeyError(
          f"Performance stage {index} is missing fields {sorted(missing)}."
        )
      level = stage["min_terrain_level"]
      if not isinstance(level, int) or isinstance(level, bool):
        raise TypeError(f"Performance stage {index} min_terrain_level must be int.")
      if not 0 <= level <= max_terrain_level:
        raise ValueError(
          f"Performance stage {index} min_terrain_level must be within "
          f"[0, {max_terrain_level}]."
        )
      if level < previous_level:
        raise ValueError("Performance stage terrain levels must be nondecreasing.")
      previous_level = level
      for range_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        value = stage[range_name]
        if (
          len(value) != 2
          or not all(math.isfinite(float(bound)) for bound in value)
          or value[0] > value[1]
        ):
          raise ValueError(f"Performance stage {index} has invalid {range_name}.")
      _validate_directional_bucket_fractions(
        stage["rel_forward_envs"], stage["rel_backward_envs"]
      )

  def _apply_stage(self, stage_index: int) -> None:
    stage = self._stages[stage_index]
    self._command_cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
    self._command_cfg.ranges.lin_vel_y = tuple(stage["lin_vel_y"])
    self._command_cfg.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
    self._command_cfg.rel_forward_envs = float(stage["rel_forward_envs"])
    self._command_cfg.rel_backward_envs = float(stage["rel_backward_envs"])

  def _accumulate_completed_episodes(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> None:
    next_level = self._stages[self._stage_index + 1]["min_terrain_level"]
    terrain_types = self._terrain_types[env_ids]
    terrain_levels = self._terrain_levels[env_ids]
    distance = torch.norm(
      self._asset.data.root_link_pos_w[env_ids, :2]
      - self._terrain_env_origins[env_ids, :2],
      dim=1,
    )
    made_progress = distance > self._terrain_generator.size[0] / 2
    completed_safely = env.termination_manager.get_term(self._success_termination_name)[
      env_ids
    ].clone()
    # 2026-09-02 stair-training update: the designated horizon timeout must
    # fire alone; simultaneous falls, illegal contacts, terrain exits, or any
    # other termination/truncation remain failures.
    for term_name in self._competing_termination_names:
      completed_safely &= ~env.termination_manager.get_term(term_name)[env_ids]
    if self._contact_sensor is not None:
      foot_air_time = self._contact_sensor.data.current_air_time
      assert foot_air_time is not None
      assert self._max_foot_air_time is not None
      healthy_foot_usage = torch.all(
        foot_air_time[env_ids].reshape(len(env_ids), -1) <= self._max_foot_air_time,
        dim=1,
      )
      completed_safely &= healthy_foot_usage
    safe_success = made_progress & completed_safely
    directional = self._directional_stats(env_ids)

    for terrain_name, terrain_type_id in self._terrain_type_ids.items():
      eligible = (terrain_types == terrain_type_id) & (terrain_levels >= next_level)
      if not eligible.any():
        continue
      bucket = self._window[terrain_name]
      bucket["episode_count"] += int(eligible.sum().item())
      bucket["safe_success_count"] += int(safe_success[eligible].sum().item())
      bucket["forward_tracking_sum"] += float(
        directional["forward_tracking_sum"][eligible].sum().item()
      )
      bucket["forward_tracking_count"] += int(
        directional["forward_tracking_count"][eligible].sum().item()
      )
      bucket["backward_tracking_sum"] += float(
        directional["backward_tracking_sum"][eligible].sum().item()
      )
      bucket["backward_tracking_count"] += int(
        directional["backward_tracking_count"][eligible].sum().item()
      )

  def _window_ready(self) -> bool:
    for bucket in self._window.values():
      if bucket["episode_count"] < self._min_episodes_per_terrain:
        return False
      if (
        bucket["forward_tracking_count"] < self._min_directional_samples_per_terrain
        or bucket["backward_tracking_count"] < self._min_directional_samples_per_terrain
      ):
        return False
    return True

  def _window_passes(self) -> bool:
    for bucket in self._window.values():
      success_rate = bucket["safe_success_count"] / bucket["episode_count"]
      forward_score = bucket["forward_tracking_sum"] / bucket["forward_tracking_count"]
      backward_score = (
        bucket["backward_tracking_sum"] / bucket["backward_tracking_count"]
      )
      if success_rate < self._min_safe_success_rate:
        return False
      if min(forward_score, backward_score) < self._min_directional_tracking_score:
        return False
    return True

  def _empty_window(self) -> dict[str, _TerrainPerformance]:
    return {
      name: {
        "episode_count": 0,
        "safe_success_count": 0,
        "forward_tracking_sum": 0.0,
        "forward_tracking_count": 0,
        "backward_tracking_sum": 0.0,
        "backward_tracking_count": 0,
      }
      for name in self._terrain_names
    }

  def _logging_state(self, step: int) -> dict[str, torch.Tensor]:
    state = {
      "stage": torch.tensor(self._stage_index),
      "stage_steps": torch.tensor(max(0, step - self._stage_enter_step)),
      "pass_streak": torch.tensor(self._pass_streak),
      "last_window_passed": torch.tensor(float(self._last_window_passed)),
    }
    if self._stage_index < len(self._stages) - 1:
      state["next_min_terrain_level"] = torch.tensor(
        self._stages[self._stage_index + 1]["min_terrain_level"]
      )
    for terrain_name, bucket in self._window.items():
      episodes = bucket["episode_count"]
      forward_count = bucket["forward_tracking_count"]
      backward_count = bucket["backward_tracking_count"]
      state[f"{terrain_name}/episodes"] = torch.tensor(episodes)
      state[f"{terrain_name}/safe_success_rate"] = torch.tensor(
        bucket["safe_success_count"] / episodes if episodes else 0.0
      )
      state[f"{terrain_name}/forward_tracking_score"] = torch.tensor(
        bucket["forward_tracking_sum"] / forward_count if forward_count else 0.0
      )
      state[f"{terrain_name}/backward_tracking_score"] = torch.tensor(
        bucket["backward_tracking_sum"] / backward_count if backward_count else 0.0
      )
    # 2026-09-02 stair-training update: an evaluated window is cleared before
    # logging, so retain its exact gate evidence under stable ``last/`` keys.
    for terrain_name, bucket in self._last_window.items():
      episodes = bucket["episode_count"]
      forward_count = bucket["forward_tracking_count"]
      backward_count = bucket["backward_tracking_count"]
      state[f"last/{terrain_name}/episodes"] = torch.tensor(episodes)
      state[f"last/{terrain_name}/safe_success_rate"] = torch.tensor(
        bucket["safe_success_count"] / episodes if episodes else 0.0
      )
      state[f"last/{terrain_name}/forward_tracking_score"] = torch.tensor(
        bucket["forward_tracking_sum"] / forward_count if forward_count else 0.0
      )
      state[f"last/{terrain_name}/backward_tracking_score"] = torch.tensor(
        bucket["backward_tracking_sum"] / backward_count if backward_count else 0.0
      )
    return state

  @staticmethod
  def _checked_int(state: Mapping[str, Any], key: str, minimum: int) -> int:
    value = state.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
      raise ValueError(f"Performance curriculum {key} must be an int >= {minimum}.")
    return value

  def _validated_window(
    self, value: Any, *, name: str
  ) -> dict[str, _TerrainPerformance]:
    if not isinstance(value, Mapping) or set(value) != set(self._terrain_names):
      raise ValueError(f"Performance curriculum {name} terrain buckets do not match.")
    result = self._empty_window()
    integer_fields = {
      "episode_count",
      "safe_success_count",
      "forward_tracking_count",
      "backward_tracking_count",
    }
    for terrain_name in self._terrain_names:
      raw_bucket = value[terrain_name]
      if not isinstance(raw_bucket, Mapping) or set(raw_bucket) != set(
        self._WINDOW_FIELDS
      ):
        raise ValueError(f"Performance curriculum {name}/{terrain_name} is invalid.")
      bucket = result[terrain_name]
      for field in self._WINDOW_FIELDS:
        raw = raw_bucket[field]
        if field in integer_fields:
          if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
            raise ValueError(
              f"Performance curriculum {name}/{terrain_name}/{field} is invalid."
            )
          bucket[field] = raw
        else:
          if not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 0:
            raise ValueError(
              f"Performance curriculum {name}/{terrain_name}/{field} is invalid."
            )
          bucket[field] = float(raw)
      if bucket["safe_success_count"] > bucket["episode_count"]:
        raise ValueError(
          f"Performance curriculum {name}/{terrain_name} has excess successes."
        )
      if bucket["forward_tracking_sum"] > bucket["forward_tracking_count"] + 1e-6:
        raise ValueError(
          f"Performance curriculum {name}/{terrain_name} has invalid forward score."
        )
      if bucket["backward_tracking_sum"] > bucket["backward_tracking_count"] + 1e-6:
        raise ValueError(
          f"Performance curriculum {name}/{terrain_name} has invalid backward score."
        )
    return result

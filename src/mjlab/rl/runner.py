import os
import warnings
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from mjlab.entity import Entity
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # Strip None-valued optional configs so MLPModel doesn't receive them.
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export policy to ONNX format using legacy export path.

    Overrides the base implementation to set dynamo=False, avoiding warnings about
    dynamic_axes being deprecated with the new TorchDynamo export path
    (torch>=2.9 default).
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint.

    Extends the base implementation to persist the environment's curriculum
    progress and to respect the ``upload_model`` config flag.
    """
    env_state: dict[str, Any] = {
      "common_step_counter": self.env.unwrapped.common_step_counter
    }
    # 2026-09-02 stair-training update: checkpoint stateful curriculum terms.
    env_state["curriculum"] = self.env.unwrapped.curriculum_manager.state_dict()

    terrain = self.env.unwrapped.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
      # fix terrain levels problem: keep curriculum placement when training resumes.
      generator_cfg = terrain.cfg.terrain_generator
      env_state["terrain"] = {
        # Store environment state on CPU so checkpoints are portable across devices.
        "levels": terrain.terrain_levels.detach().cpu().clone(),
        "types": terrain.terrain_types.detach().cpu().clone(),
        "grid_shape": tuple(int(size) for size in terrain.terrain_origins.shape[:2]),
        "terrain_names": (
          tuple(generator_cfg.sub_terrains) if generator_cfg is not None else None
        ),
      }
    infos = {**(infos or {}), "env_state": env_state}
    # Inline base OnPolicyRunner.save() to conditionally gate W&B upload.
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoint.

    Extends the base implementation to:
    1. Restore the step counter and compatible terrain curriculum placement.
    2. Migrate legacy checkpoints (actor.* -> mlp.*, actor_obs_normalizer.*
      -> obs_normalizer.*) to the current format (rsl-rl>=4.0).
    """
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    if "model_state_dict" in loaded_dict:
      print(f"Detected legacy checkpoint at {path}. Migrating to new format...")
      model_state_dict = loaded_dict.pop("model_state_dict")
      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():
        # Migrate actor keys.
        if key.startswith("actor."):
          new_key = key.replace("actor.", "mlp.")
          actor_state_dict[new_key] = value
        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace("actor_obs_normalizer.", "obs_normalizer.")
          actor_state_dict[new_key] = value
        elif key in ["std", "log_std"]:
          actor_state_dict[key] = value

        # Migrate critic keys.
        if key.startswith("critic."):
          new_key = key.replace("critic.", "mlp.")
          critic_state_dict[new_key] = value
        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace("critic_obs_normalizer.", "obs_normalizer.")
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = actor_state_dict
      loaded_dict["critic_state_dict"] = critic_state_dict

    # Migrate rsl-rl 4.x actor keys to 5.x distribution keys.
    actor_sd = loaded_dict.get("actor_state_dict", {})
    if "std" in actor_sd:
      actor_sd["distribution.std_param"] = actor_sd.pop("std")
    if "log_std" in actor_sd:
      actor_sd["distribution.log_std_param"] = actor_sd.pop("log_std")

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict.get("infos")
    # Partial loads are used by play/hot-swap and must not teleport the live env.
    if load_iteration and infos and "env_state" in infos:
      self._restore_env_state(infos["env_state"])
    return infos

  def _restore_env_state(self, env_state: object) -> None:
    """Restore checkpointed environment state for a full training resume."""
    if not isinstance(env_state, dict):
      warnings.warn(
        "Checkpoint env_state is not a dictionary; environment state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return
    env_state = cast(dict[str, Any], env_state)

    common_step_counter = env_state.get("common_step_counter")
    if isinstance(common_step_counter, int):
      self.env.unwrapped.common_step_counter = common_step_counter
    elif common_step_counter is not None:
      warnings.warn(
        "Checkpoint common_step_counter is invalid; keeping the current value.",
        RuntimeWarning,
        stacklevel=2,
      )

    terrain_state = env_state.get("terrain")
    if terrain_state is not None:
      self._restore_terrain_state(terrain_state)

    # 2026-09-02 stair-training update: old checkpoints omit this key.
    curriculum_state = env_state.get("curriculum")
    if curriculum_state is not None:
      self.env.unwrapped.curriculum_manager.load_state_dict(curriculum_state)

  def _restore_terrain_state(self, terrain_state: object) -> None:
    """Restore terrain buckets and translate the live scene to their new origins."""
    env = self.env.unwrapped
    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      warnings.warn(
        "Checkpoint contains terrain curriculum state, but the current environment "
        "has no curriculum terrain; terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return
    if not isinstance(terrain_state, dict):
      warnings.warn(
        "Checkpoint terrain state is invalid; terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return
    terrain_state = cast(dict[str, Any], terrain_state)

    saved_grid_shape = terrain_state.get("grid_shape")
    current_grid_shape = tuple(int(size) for size in terrain.terrain_origins.shape[:2])
    if (
      not isinstance(saved_grid_shape, (tuple, list))
      or tuple(saved_grid_shape) != current_grid_shape
    ):
      warnings.warn(
        "Checkpoint terrain grid does not match the current configuration; "
        "terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return

    generator_cfg = terrain.cfg.terrain_generator
    current_terrain_names = (
      tuple(generator_cfg.sub_terrains) if generator_cfg is not None else None
    )
    saved_terrain_names = terrain_state.get("terrain_names")
    if saved_terrain_names is not None and (
      not isinstance(saved_terrain_names, (tuple, list))
      or tuple(saved_terrain_names) != current_terrain_names
    ):
      warnings.warn(
        "Checkpoint terrain types do not match the current configuration; "
        "terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return

    levels = self._validated_terrain_indices(
      terrain_state.get("levels"),
      name="levels",
      current=terrain.terrain_levels,
      upper_bound=current_grid_shape[0],
    )
    types = self._validated_terrain_indices(
      terrain_state.get("types"),
      name="types",
      current=terrain.terrain_types,
      upper_bound=current_grid_shape[1],
    )
    if levels is None or types is None:
      return

    if not isinstance(terrain.env_origins, torch.Tensor):
      warnings.warn(
        "Current terrain origins are invalid; terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return

    old_origins = terrain.env_origins.clone()
    new_origins = terrain.terrain_origins[levels, types]
    if old_origins.shape != new_origins.shape:
      warnings.warn(
        "Checkpoint terrain origins do not match the current environment count; "
        "terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return

    origin_delta = new_origins - old_origins
    terrain.terrain_levels.copy_(levels)
    terrain.terrain_types.copy_(types)
    terrain.env_origins.copy_(new_origins)

    # The wrapper has already reset entities at its initially sampled origins.
    # Translate them by the same delta instead of resetting again, which would run
    # the curriculum and could immediately alter the restored terrain levels.
    for entity in env.scene.entities.values():
      if not isinstance(entity, Entity):
        continue
      if entity.is_fixed_base and entity.is_mocap:
        mocap_id = entity.data.indexing.mocap_id
        assert mocap_id is not None
        mocap_pose = torch.cat(
          [
            entity.data.data.mocap_pos[:, mocap_id],
            entity.data.data.mocap_quat[:, mocap_id],
          ],
          dim=-1,
        ).clone()
        mocap_pose[:, :3] += origin_delta
        entity.write_mocap_pose_to_sim(mocap_pose)
      elif not entity.is_fixed_base:
        root_pose = entity.data.root_link_pose_w.clone()
        root_pose[:, :3] += origin_delta
        entity.write_root_link_pose_to_sim(root_pose)

    # Refresh all derived state and force the next observation query to rebuild.
    env.sim.forward()
    env.sim.sense()
    env.observation_manager.invalidate_cache()

  @staticmethod
  def _validated_terrain_indices(
    value: object,
    *,
    name: str,
    current: torch.Tensor,
    upper_bound: int,
  ) -> torch.Tensor | None:
    """Validate and move saved terrain index vectors to the environment device."""
    if not isinstance(value, torch.Tensor) or value.shape != current.shape:
      warnings.warn(
        f"Checkpoint terrain {name} do not match the current environment count; "
        "terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return None
    if value.is_complex():
      warnings.warn(
        f"Checkpoint terrain {name} contain complex values; terrain state was not "
        "restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return None
    if value.is_floating_point() and (
      not torch.isfinite(value).all() or not torch.equal(value, value.round())
    ):
      warnings.warn(
        f"Checkpoint terrain {name} contain non-integer values; terrain state "
        "was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return None

    indices = value.to(device=current.device, dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= upper_bound):
      warnings.warn(
        f"Checkpoint terrain {name} are outside the current terrain grid; "
        "terrain state was not restored.",
        RuntimeWarning,
        stacklevel=2,
      )
      return None
    return indices

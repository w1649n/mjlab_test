"""Tests for MjlabOnPolicyRunner."""

import ast
import tempfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mujoco
import onnx
import pytest
import torch
from conftest import get_test_device
from rsl_rl.models import MLPModel
from rsl_rl.utils import WandbLogWriter
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

import mjlab.scripts.train as train_mod
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking.rl.runner import _OnnxMotionModel
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.spaces import Box


@pytest.fixture(scope="module")
def device():
  return get_test_device()


@pytest.fixture(scope="module")
def env(device):
  robot_xml = """
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <freejoint name="free_joint"/>
        <geom name="base_geom" type="box" size="0.2 0.2 0.1" mass="1.0"/>
        <body name="link1" pos="0 0 0">
          <joint name="joint1" type="hinge" axis="0 0 1" range="-1.57 1.57"/>
          <geom name="link1_geom" type="box" size="0.1 0.1 0.1" mass="0.1"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor name="actuator1" joint="joint1" gear="1.0"/>
    </actuator>
  </mujoco>
  """
  robot_cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(robot_xml),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)
    ),
  )

  env_cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=2,
      extent=1.0,
      entities={"robot": robot_cfg},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
      "critic": ObservationGroupCfg(
        terms={
          "joint_pos": ObservationTermCfg(
            func=lambda env: env.scene["robot"].data.joint_pos
          ),
        },
      ),
    },
    actions={
      "joint_pos": mdp.JointPositionActionCfg(
        entity_name="robot", actuator_names=(".*",), scale=1.0
      )
    },
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, iterations=1)),
    decimation=1,
    episode_length_s=1.0,
  )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  yield env
  env.close()


def test_vecenv_wrapper_clips_raw_actions_before_environment_step(env, device):
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=5.0)
  raw_actions = torch.tensor([[9.0], [-2.0]], device=device)

  _, _, _, extras = wrapped_env.step(raw_actions)

  expected = torch.tensor([[5.0], [-2.0]], device=device)
  torch.testing.assert_close(env.action_manager.action, expected)
  action_term = env.action_manager.get_term("joint_pos")
  torch.testing.assert_close(action_term.raw_action, expected)
  assert extras["log"]["Metrics/raw_action_step_abs_max"].item() == pytest.approx(9.0)
  assert extras["log"]["Metrics/raw_action_clip_fraction"].item() == pytest.approx(0.5)
  assert isinstance(wrapped_env.action_space, Box)
  assert wrapped_env.action_space.low == -5.0
  assert wrapped_env.action_space.high == 5.0


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_vecenv_wrapper_rejects_nonfinite_actions_before_environment_step(
  env, device, nonfinite
):
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=5.0)
  action_before = env.action_manager.action.clone()
  raw_actions = torch.tensor([[nonfinite], [0.0]], device=device)

  with pytest.raises(FloatingPointError, match="non-finite raw action"):
    wrapped_env.step(raw_actions)

  torch.testing.assert_close(env.action_manager.action, action_before)


@pytest.mark.parametrize("clip_actions", [0.0, -1.0, float("nan"), float("inf")])
def test_vecenv_wrapper_rejects_invalid_action_clip(clip_actions):
  with pytest.raises(ValueError, match="positive finite"):
    RslRlVecEnvWrapper(MagicMock(), clip_actions=clip_actions)


# 2026-09-02 stair-training update: simultaneous failure and timeout remains
# terminal for PPO, while a timeout without failure is still bootstrapped.
def test_vecenv_wrapper_excludes_failures_from_timeout_bootstrap(
  env, device, monkeypatch
):
  wrapped_env = RslRlVecEnvWrapper(env)
  observations = env.get_observations()
  rewards = torch.zeros(2, device=device)
  terminated = torch.tensor([False, True], device=device)
  truncated = torch.tensor([True, True], device=device)
  monkeypatch.setattr(
    env,
    "step",
    lambda _actions: (observations, rewards, terminated, truncated, {}),
  )

  _, _, dones, extras = wrapped_env.step(torch.zeros(2, 1, device=device))

  assert dones.tolist() == [1, 1]
  assert extras["time_outs"].tolist() == [True, False]


@pytest.fixture
def curriculum_terrain(env, device):
  """Temporarily turn the plane fixture into an isolated curriculum terrain."""
  terrain = env.scene.terrain
  assert terrain is not None

  had_levels = hasattr(terrain, "terrain_levels")
  had_types = hasattr(terrain, "terrain_types")
  old_levels = getattr(terrain, "terrain_levels", None)
  old_types = getattr(terrain, "terrain_types", None)
  old_terrain_origins = terrain.terrain_origins
  old_env_origins = terrain.env_origins
  old_generator_cfg = terrain.cfg.terrain_generator
  old_common_step_counter = env.common_step_counter

  terrain_origins = torch.tensor(
    [
      [[0.0, 0.0, 0.0], [0.0, 20.0, 1.0]],
      [[10.0, 0.0, 2.0], [10.0, 20.0, 3.0]],
      [[20.0, 0.0, 4.0], [20.0, 20.0, 5.0]],
    ],
    device=device,
  )
  terrain.cfg.terrain_generator = SimpleNamespace(
    sub_terrains={"flat": object(), "stairs": object()}
  )
  terrain.terrain_origins = terrain_origins
  terrain.terrain_levels = torch.tensor([0, 1], device=device, dtype=torch.long)
  terrain.terrain_types = torch.tensor([0, 1], device=device, dtype=torch.long)
  terrain.env_origins = terrain_origins[
    terrain.terrain_levels, terrain.terrain_types
  ].clone()

  try:
    yield terrain, terrain_origins
  finally:
    terrain.cfg.terrain_generator = old_generator_cfg
    terrain.terrain_origins = old_terrain_origins
    terrain.env_origins = old_env_origins
    if had_levels:
      terrain.terrain_levels = old_levels
    else:
      del terrain.terrain_levels
    if had_types:
      terrain.terrain_types = old_types
    else:
      del terrain.terrain_types
    # Restore both the Python origins and live MuJoCo state for the module fixture.
    env.common_step_counter = old_common_step_counter
    env.reset()


def test_runner_persists_common_step_counter(env, device, monkeypatch):
  """MjlabOnPolicyRunner should save and restore common_step_counter."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)

    wrapped_env.unwrapped.common_step_counter = 12345
    checkpoint_path = str(Path(tmpdir) / "test_checkpoint.pt")
    runner.save(checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 0
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 12345


# 2026-09-02 stair-training update: cover manager state through a full save/load.
def test_runner_persists_curriculum_manager_state(env, device, monkeypatch):
  wrapped_env = RslRlVecEnvWrapper(env)
  curriculum_manager = wrapped_env.unwrapped.curriculum_manager
  curriculum_state = {"command_velocity": {"stage": 2, "score": 0.75}}
  state_dict = MagicMock(return_value=curriculum_state)
  load_state_dict = MagicMock()
  monkeypatch.setattr(curriculum_manager, "state_dict", state_dict)
  monkeypatch.setattr(curriculum_manager, "load_state_dict", load_state_dict)

  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )
  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)
    checkpoint_path = str(Path(tmpdir) / "curriculum_checkpoint.pt")

    runner.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["infos"]["env_state"]["curriculum"] == curriculum_state
    state_dict.assert_called_once_with()

    runner.load(checkpoint_path)
    load_state_dict.assert_called_once_with(curriculum_state)


def test_runner_handles_old_checkpoints_without_env_state(env, device):
  """Old checkpoints without env_state should load without crashing."""

  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )

    checkpoint_path = str(Path(tmpdir) / "old_checkpoint.pt")
    old_checkpoint = {
      "actor_state_dict": runner.alg.actor.state_dict(),
      "critic_state_dict": runner.alg.critic.state_dict(),
      "optimizer_state_dict": runner.alg.optimizer.state_dict(),
      "iter": 100,
      "infos": None,
    }
    torch.save(old_checkpoint, checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 999
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 999


# 2026-09-02 stair-training update: prior env_state payloads lack curriculum.
def test_runner_handles_old_env_state_without_curriculum(env, device, monkeypatch):
  wrapped_env = RslRlVecEnvWrapper(env)
  curriculum_manager = wrapped_env.unwrapped.curriculum_manager
  load_state_dict = MagicMock()
  monkeypatch.setattr(curriculum_manager, "load_state_dict", load_state_dict)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    checkpoint = runner.alg.save()
    checkpoint.update(
      {
        "iter": 100,
        "infos": {"env_state": {"common_step_counter": 2468}},
      }
    )
    checkpoint_path = str(Path(tmpdir) / "old_env_state_checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 0
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 2468
    load_state_dict.assert_not_called()


def test_runner_persists_terrain_state_and_aligns_entities(
  env, device, monkeypatch, curriculum_terrain
):
  """Full resume restores terrain buckets without leaving robots at old origins."""
  terrain, terrain_origins = curriculum_terrain
  terrain.terrain_levels.copy_(torch.tensor([2, 1], device=device))
  terrain.terrain_types.copy_(torch.tensor([1, 0], device=device))
  terrain.env_origins.copy_(
    terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  )
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)

    wrapped_env.unwrapped.common_step_counter = 12345
    checkpoint_path = str(Path(tmpdir) / "terrain_checkpoint.pt")
    runner.save(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_terrain = checkpoint["infos"]["env_state"]["terrain"]
    assert saved_terrain["levels"].device.type == "cpu"
    assert saved_terrain["types"].device.type == "cpu"
    assert saved_terrain["grid_shape"] == (3, 2)
    assert saved_terrain["terrain_names"] == ("flat", "stairs")

    terrain.terrain_levels.copy_(torch.tensor([0, 0], device=device))
    terrain.terrain_types.copy_(torch.tensor([0, 1], device=device))
    terrain.env_origins.copy_(
      terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    )
    wrapped_env.reset()
    robot = wrapped_env.unwrapped.scene["robot"]
    relative_pose_before = (
      robot.data.root_link_pose_w[:, :3] - terrain.env_origins
    ).clone()

    wrapped_env.unwrapped.common_step_counter = 0
    runner.load(checkpoint_path)

    assert wrapped_env.unwrapped.common_step_counter == 12345
    assert terrain.terrain_levels.tolist() == [2, 1]
    assert terrain.terrain_types.tolist() == [1, 0]
    torch.testing.assert_close(
      terrain.env_origins,
      terrain_origins[terrain.terrain_levels, terrain.terrain_types],
    )
    torch.testing.assert_close(
      robot.data.root_link_pose_w[:, :3] - terrain.env_origins,
      relative_pose_before,
    )


def test_runner_rejects_invalid_terrain_state_atomically(
  env, device, curriculum_terrain
):
  """Invalid saved indices must not partially mutate terrain or entity placement."""
  terrain, _ = curriculum_terrain
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    levels_before = terrain.terrain_levels.clone()
    types_before = terrain.terrain_types.clone()
    origins_before = terrain.env_origins.clone()
    robot_pose_before = env.scene["robot"].data.root_link_pose_w.clone()

    invalid_state = {
      "levels": torch.tensor([1, 0]),
      "types": torch.tensor([0, 2]),
      "grid_shape": (3, 2),
      "terrain_names": ("flat", "stairs"),
    }
    with pytest.warns(RuntimeWarning, match="outside the current terrain grid"):
      runner._restore_terrain_state(invalid_state)

    assert torch.equal(terrain.terrain_levels, levels_before)
    assert torch.equal(terrain.terrain_types, types_before)
    assert torch.equal(terrain.env_origins, origins_before)
    assert torch.equal(env.scene["robot"].data.root_link_pose_w, robot_pose_before)


@pytest.mark.parametrize("mismatch", ["grid_shape", "terrain_names", "num_envs"])
def test_runner_rejects_mismatched_terrain_configuration_atomically(
  env, device, curriculum_terrain, mismatch
):
  """A checkpoint from an incompatible terrain layout must be ignored as a unit."""
  terrain, _ = curriculum_terrain
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    levels_before = terrain.terrain_levels.clone()
    types_before = terrain.terrain_types.clone()
    origins_before = terrain.env_origins.clone()

    terrain_state = {
      "levels": torch.tensor([1, 0]),
      "types": torch.tensor([1, 0]),
      "grid_shape": (3, 2),
      "terrain_names": ("flat", "stairs"),
    }
    if mismatch == "grid_shape":
      terrain_state["grid_shape"] = (2, 2)
      warning = "terrain grid"
    elif mismatch == "terrain_names":
      terrain_state["terrain_names"] = ("stairs", "flat")
      warning = "terrain types"
    else:
      terrain_state["levels"] = torch.tensor([1])
      warning = "environment count"

    with pytest.warns(RuntimeWarning, match=warning):
      runner._restore_terrain_state(terrain_state)

    assert torch.equal(terrain.terrain_levels, levels_before)
    assert torch.equal(terrain.terrain_types, types_before)
    assert torch.equal(terrain.env_origins, origins_before)


def test_runner_handles_environment_without_terrain(env, device, monkeypatch):
  """Saving is valid and saved curriculum state is ignored when terrain is absent."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapped_env.unwrapped.scene, "_terrain", None)

    checkpoint_path = str(Path(tmpdir) / "no_terrain_checkpoint.pt")
    runner.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert "terrain" not in checkpoint["infos"]["env_state"]
    with pytest.warns(RuntimeWarning, match="has no curriculum terrain"):
      runner._restore_terrain_state(
        {
          "levels": torch.tensor([0, 0]),
          "types": torch.tensor([0, 0]),
          "grid_shape": (1, 1),
        }
      )


def test_runner_partial_load_does_not_restore_environment(env, device, monkeypatch):
  """Actor-only loads used by play/hot-swap must not alter the live environment."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    monkeypatch.setattr(runner.logger, "save_model", lambda *args, **kwargs: None)
    wrapped_env.unwrapped.common_step_counter = 12345
    checkpoint_path = str(Path(tmpdir) / "partial_checkpoint.pt")
    runner.save(checkpoint_path)

    wrapped_env.unwrapped.common_step_counter = 77
    with patch.object(runner, "_restore_env_state") as restore_env_state:
      runner.load(checkpoint_path, load_cfg={"actor": True})

    restore_env_state.assert_not_called()
    assert wrapped_env.unwrapped.common_step_counter == 77


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_policy_to_onnx(env, device):
  """runner.export_policy_to_onnx() produces a valid ONNX file."""
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=4, max_iterations=10, save_interval=5
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    runner = MjlabOnPolicyRunner(
      wrapped_env, asdict(agent_cfg), log_dir=tmpdir, device=device
    )
    runner.export_policy_to_onnx(tmpdir, "test_policy.onnx")
    onnx_path = Path(tmpdir) / "test_policy.onnx"
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))


def _make_actor(obs_dim=8, output_dim=4, obs_normalization=True):
  obs = TensorDict({"actor": torch.zeros(1, obs_dim)})
  obs_groups = {"actor": ["actor"]}
  return MLPModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=output_dim,
    hidden_dims=[32, 32],
    activation="elu",
    obs_normalization=obs_normalization,
  )


def _train_normalizer(actor, n_batches=50, batch_size=64):
  actor.train()
  for _ in range(n_batches):
    obs = TensorDict({"actor": torch.randn(batch_size, actor.obs_dim) * 5 + 3})
    actor.update_normalization(obs)
  actor.eval()


def _model_output(actor, x_flat):
  obs = TensorDict({"actor": x_flat})
  with torch.no_grad():
    return actor(obs)


def test_onnx_export_matches_actor():
  """as_onnx() model produces the same output as the full actor with normalization."""
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()
  x = torch.randn(4, actor.obs_dim)
  model_out = _model_output(actor, x)
  with torch.no_grad():
    onnx_out = onnx_model(x)
  torch.testing.assert_close(model_out, onnx_out, atol=1e-6, rtol=0)


def test_onnx_export_without_normalization():
  """as_onnx() works when normalization is disabled."""
  actor = _make_actor(obs_normalization=False)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()
  x = torch.randn(4, actor.obs_dim)
  with torch.no_grad():
    out = onnx_model(x)
  assert out.shape == (4, 4)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_onnx_runtime_roundtrip_matches_pytorch():
  """Exported .onnx file produces the same outputs as PyTorch via onnxruntime."""
  ort = pytest.importorskip("onnxruntime")
  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()

  x = torch.randn(4, actor.obs_dim)
  expected = _model_output(actor, x)

  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "policy.onnx"
    torch.onnx.export(
      onnx_model,
      (x,),
      str(path),
      input_names=onnx_model.input_names,  # pyright: ignore[reportArgumentType]
      output_names=onnx_model.output_names,  # pyright: ignore[reportArgumentType]
      opset_version=18,
      dynamo=False,
    )
    sess = ort.InferenceSession(str(path))
    [actual] = sess.run(None, {"obs": x.numpy()})

  torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=0)


# CNN (spatial-softmax) ONNX export tests.

_IMG_H, _IMG_W, _IMG_C = 16, 16, 3
_OBS_DIM_1D = 8
_OUTPUT_DIM = 4


def _make_cnn_actor(obs_normalization=True):
  obs = TensorDict(
    {
      "actor": torch.zeros(1, _OBS_DIM_1D),
      "camera": torch.zeros(1, _IMG_C, _IMG_H, _IMG_W),
    }
  )
  obs_groups = {"actor": ["actor", "camera"]}
  cnn_cfg = {
    "output_channels": [8],
    "kernel_size": [3],
    "stride": [1],
    "spatial_softmax_temperature": 1.0,
  }
  return SpatialSoftmaxCNNModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=_OUTPUT_DIM,
    cnn_cfg=cnn_cfg,
    hidden_dims=[32, 32],
    activation="elu",
    obs_normalization=obs_normalization,
  )


def _train_cnn_normalizer(actor, n_batches=50, batch_size=64):
  actor.train()
  for _ in range(n_batches):
    obs = TensorDict(
      {
        "actor": torch.randn(batch_size, _OBS_DIM_1D) * 5 + 3,
        "camera": torch.randn(batch_size, _IMG_C, _IMG_H, _IMG_W),
      }
    )
    actor.update_normalization(obs)
  actor.eval()


def _cnn_model_output(actor, x_1d, x_2d):
  obs = TensorDict({"actor": x_1d, "camera": x_2d})
  with torch.no_grad():
    return actor(obs)


def test_cnn_onnx_export_matches_actor():
  """as_onnx() with SpatialSoftmaxCNNModel matches the original model."""
  actor = _make_cnn_actor(obs_normalization=True)
  _train_cnn_normalizer(actor)

  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()

  x_1d = torch.randn(4, _OBS_DIM_1D)
  x_2d = torch.randn(4, _IMG_C, _IMG_H, _IMG_W)

  expected = _cnn_model_output(actor, x_1d, x_2d)
  with torch.no_grad():
    actual = onnx_model(x_1d, x_2d)
  torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cnn_onnx_export_to_file():
  """SpatialSoftmaxCNNModel exports to a valid ONNX file."""
  actor = _make_cnn_actor(obs_normalization=False)
  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()

  with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = Path(tmpdir) / "cnn_policy.onnx"
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # pyright: ignore[reportCallIssue]
      str(onnx_path),
      export_params=True,
      opset_version=18,
      input_names=onnx_model.input_names,  # pyright: ignore[reportArgumentType]
      output_names=onnx_model.output_names,  # pyright: ignore[reportArgumentType]
      dynamic_axes={},
      dynamo=False,
    )
    assert onnx_path.exists()
    onnx.checker.check_model(str(onnx_path))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cnn_onnx_runtime_roundtrip_matches_pytorch():
  """Exported CNN .onnx file produces the same outputs as PyTorch via onnxruntime."""
  ort = pytest.importorskip("onnxruntime")
  actor = _make_cnn_actor(obs_normalization=True)
  _train_cnn_normalizer(actor)

  onnx_model = actor.as_onnx(verbose=False)
  onnx_model.eval()

  x_1d = torch.randn(4, _OBS_DIM_1D)
  x_2d = torch.randn(4, _IMG_C, _IMG_H, _IMG_W)
  expected = _cnn_model_output(actor, x_1d, x_2d)

  with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "cnn_policy.onnx"
    torch.onnx.export(
      onnx_model,
      (x_1d, x_2d),
      str(path),
      input_names=onnx_model.input_names,  # pyright: ignore[reportArgumentType]
      output_names=onnx_model.output_names,  # pyright: ignore[reportArgumentType]
      opset_version=18,
      dynamo=False,
    )
    sess = ort.InferenceSession(str(path))
    [actual] = sess.run(None, {"obs": x_1d.numpy(), "camera": x_2d.numpy()})

  torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-5, rtol=0)


def test_get_export_paths():
  """_get_export_paths resolves the correct dir, filename, and full path."""
  # Normal case: "model" only appears in the checkpoint filename.
  export_dir, filename, onnx_path = MjlabOnPolicyRunner._get_export_paths(
    "/logs/2026-03-30_12-00-00/model_10.pt"
  )
  assert export_dir == Path("/logs/2026-03-30_12-00-00")
  assert filename == "2026-03-30_12-00-00.onnx"
  assert onnx_path == Path("/logs/2026-03-30_12-00-00/2026-03-30_12-00-00.onnx")

  # Bug case: "model" also appears in a parent directory name — the old
  # path.split("model")[0] would have truncated to "/tmp/my_".
  export_dir, filename, onnx_path = MjlabOnPolicyRunner._get_export_paths(
    "/tmp/my_model_experiment/2026-03-30/model_10.pt"
  )
  assert export_dir == Path("/tmp/my_model_experiment/2026-03-30")
  assert filename == "2026-03-30.onnx"
  assert onnx_path == Path("/tmp/my_model_experiment/2026-03-30/2026-03-30.onnx")


def test_agent_cfg_serializable_after_runner_creation(env, device):
  """dump_yaml must be called before runner creation.

  The runner mutates agent_cfg in-place (e.g. resolve_symmetry_config injects
  non-serializable objects). Verify that the train script writes config files before
  constructing the runner.

  Regression test for https://github.com/mjlab-org/mjlab/issues/764.
  """
  wrapped_env = RslRlVecEnvWrapper(env)
  agent_cfg = asdict(
    RslRlOnPolicyRunnerCfg(num_steps_per_env=4, max_iterations=10, save_interval=5)
  )

  # Dump should succeed before runner creation.
  with tempfile.TemporaryDirectory() as tmpdir:
    dump_yaml(Path(tmpdir) / "agent.yaml", agent_cfg)

  # Create runner (mutates agent_cfg via resolve_symmetry_config).
  with tempfile.TemporaryDirectory() as tmpdir:
    MjlabOnPolicyRunner(wrapped_env, agent_cfg, log_dir=tmpdir, device=device)

  # Confirm that the runner added non-serializable keys to agent_cfg.
  sym_cfg = agent_cfg.get("algorithm", {}).get("symmetry_cfg")
  runner_mutated = sym_cfg is not None or "multi_gpu" in agent_cfg
  assert runner_mutated, "Expected runner to mutate agent_cfg"

  # Verify the train script calls dump_yaml before runner_cls().
  source = Path(train_mod.__file__).read_text()
  tree = ast.parse(source)

  dump_yaml_line = None
  runner_cls_line = None
  for node in ast.walk(tree):
    if isinstance(node, ast.Call):
      func = node.func
      # Look for dump_yaml(..., agent_cfg)
      if isinstance(func, ast.Name) and func.id == "dump_yaml":
        for arg in node.args:
          if isinstance(arg, ast.Name) and arg.id == "agent_cfg":
            dump_yaml_line = node.lineno
      # Look for runner_cls(...)
      if isinstance(func, ast.Name) and func.id == "runner_cls":
        runner_cls_line = node.lineno

  assert dump_yaml_line is not None, "dump_yaml(agent_cfg) not found"
  assert runner_cls_line is not None, "runner_cls() not found"
  assert dump_yaml_line < runner_cls_line, (
    f"dump_yaml (line {dump_yaml_line}) must be called before "
    f"runner_cls (line {runner_cls_line})"
  )


class _MockMotion:
  """Minimal mock of a motion object with tensor attributes."""

  def __init__(self, num_steps, num_joints=12, num_bodies=5):
    self.joint_pos = torch.randn(num_steps, num_joints)
    self.joint_vel = torch.randn(num_steps, num_joints)
    self.body_pos_w = torch.randn(num_steps, num_bodies, 3)
    self.body_quat_w = torch.randn(num_steps, num_bodies, 4)
    self.body_lin_vel_w = torch.randn(num_steps, num_bodies, 3)
    self.body_ang_vel_w = torch.randn(num_steps, num_bodies, 3)


def test_onnx_motion_model_policy_matches_actor():
  """_OnnxMotionModel actions output matches calling the actor directly."""

  actor = _make_actor(obs_normalization=True)
  _train_normalizer(actor)
  motion = _MockMotion(num_steps=50)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(4, actor.obs_dim)
  time_step = torch.tensor([[5]], dtype=torch.float32)

  with torch.no_grad():
    actions, *_ = model(x, time_step)
  expected = _model_output(actor, x)
  torch.testing.assert_close(actions, expected, atol=1e-6, rtol=0)


def test_onnx_motion_model_returns_correct_motion_frame():
  """_OnnxMotionModel returns the motion data at the requested time step."""

  actor = _make_actor(obs_normalization=False)
  motion = _MockMotion(num_steps=50)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(1, actor.obs_dim)
  t = 17
  time_step = torch.tensor([[t]], dtype=torch.float32)

  with torch.no_grad():
    _, joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = model(
      x, time_step
    )

  torch.testing.assert_close(joint_pos, motion.joint_pos[t : t + 1])
  torch.testing.assert_close(joint_vel, motion.joint_vel[t : t + 1])
  torch.testing.assert_close(body_pos, motion.body_pos_w[t : t + 1])
  torch.testing.assert_close(body_quat, motion.body_quat_w[t : t + 1])
  torch.testing.assert_close(body_lin_vel, motion.body_lin_vel_w[t : t + 1])
  torch.testing.assert_close(body_ang_vel, motion.body_ang_vel_w[t : t + 1])


def test_onnx_motion_model_clamps_out_of_bounds_time_step():
  """_OnnxMotionModel clamps time_step beyond motion length to last frame."""

  num_steps = 20
  actor = _make_actor(obs_normalization=False)
  motion = _MockMotion(num_steps=num_steps)

  model = _OnnxMotionModel(actor, motion)
  model.eval()

  x = torch.randn(1, actor.obs_dim)
  time_step = torch.tensor([[999]], dtype=torch.float32)

  with torch.no_grad():
    _, joint_pos, *_ = model(x, time_step)

  torch.testing.assert_close(joint_pos, motion.joint_pos[num_steps - 1 : num_steps])


def _make_logger_mock(is_wandb: bool) -> MagicMock:
  """Mock an rsl-rl Logger, specced off a real (cheap, inert) instance.

  Speccing off an instance rather than the class keeps the instance attributes,
  so an attribute rsl-rl renames or drops fails the test instead of silently
  resolving to a truthy mock.
  """
  logger = Logger(
    log_dir=None,
    cfg={"algorithm": {}},
    env_cfg={},
    num_envs=1,
    is_distributed=False,
    gpu_world_size=1,
    gpu_global_rank=0,
    device="cpu",
  )
  mock = MagicMock(spec=logger)
  mock.writer = MagicMock(spec=WandbLogWriter) if is_wandb else MagicMock()
  return mock


@pytest.mark.parametrize(
  ("runner_module", "runner_class"),
  [
    ("mjlab.tasks.velocity.rl.runner", "VelocityOnPolicyRunner"),
    ("mjlab.tasks.manipulation.rl.runner", "ManipulationOnPolicyRunner"),
  ],
)
def test_task_runner_uploads_onnx_for_wandb_logger(
  runner_module, runner_class, monkeypatch, tmp_path
):
  """ONNX is uploaded to W&B when the logger's writer is a WandbLogWriter."""
  import importlib

  runner_mod = importlib.import_module(runner_module)
  runner_cls = getattr(runner_mod, runner_class)
  runner = runner_cls.__new__(runner_cls)
  runner.cfg = {"upload_model": True}
  runner.logger = _make_logger_mock(is_wandb=True)
  runner.env = MagicMock()
  runner.export_policy_to_onnx = MagicMock()

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", lambda *a, **kw: None)
  monkeypatch.setattr(runner_mod, "get_base_metadata", lambda *a: {})
  monkeypatch.setattr(runner_mod, "attach_metadata_to_onnx", lambda *a: None)

  checkpoint = tmp_path / "run-dir" / "model_100.pt"
  checkpoint.parent.mkdir()
  checkpoint.touch()
  onnx_path = checkpoint.parent / f"{checkpoint.parent.name}.onnx"

  mock_run = MagicMock()
  mock_run.name = "test-run"
  with patch.object(runner_mod, "wandb") as mock_wandb:
    mock_wandb.run = mock_run
    runner.save(str(checkpoint))

  mock_wandb.save.assert_called_once_with(
    str(onnx_path), base_path=str(checkpoint.parent)
  )


def _make_tracking_runner_shell(registry_name, is_wandb, upload_model=True):
  """Build a MotionTrackingOnPolicyRunner with all heavy parts mocked out."""
  from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

  runner = MotionTrackingOnPolicyRunner.__new__(MotionTrackingOnPolicyRunner)
  runner.registry_name = registry_name
  runner.cfg = {"upload_model": upload_model}
  runner.logger = _make_logger_mock(is_wandb)

  mock_motion_term = MagicMock()
  mock_motion_term.cfg.anchor_body_name = "pelvis"
  mock_motion_term.cfg.body_names = ["body1"]
  runner.env = MagicMock()
  runner.env.unwrapped.command_manager.get_term.return_value = mock_motion_term
  return runner


def test_tracking_runner_registers_artifact_for_wandb_logger(monkeypatch, tmp_path):
  """use_artifact is called when the logger's writer is a WandbLogWriter."""
  from mjlab.rl.runner import MjlabOnPolicyRunner
  from mjlab.tasks.tracking.rl import runner as runner_mod

  runner = _make_tracking_runner_shell("org/motions/motion:latest", is_wandb=True)

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", lambda *a, **kw: None)
  monkeypatch.setattr(runner_mod, "get_base_metadata", lambda *a: {})
  monkeypatch.setattr(runner_mod, "attach_metadata_to_onnx", lambda *a: None)
  monkeypatch.setattr(
    runner.env.unwrapped.__class__,
    "export_policy_to_onnx",
    lambda *a, **kw: None,
    raising=False,
  )

  checkpoint = tmp_path / "run-dir" / "model_100.pt"
  checkpoint.parent.mkdir()
  checkpoint.touch()

  mock_run = MagicMock()
  mock_run.name = "test-run"

  with patch.object(runner_mod, "wandb") as mock_wandb:
    mock_wandb.run = mock_run
    runner.export_policy_to_onnx = MagicMock()
    runner.save(str(checkpoint))

  mock_run.use_artifact.assert_called_once_with("org/motions/motion:latest")


def test_tracking_runner_does_not_register_artifact_for_tensorboard(
  monkeypatch, tmp_path
):
  """use_artifact is NOT called when using the tensorboard logger."""
  from mjlab.rl.runner import MjlabOnPolicyRunner
  from mjlab.tasks.tracking.rl import runner as runner_mod

  runner = _make_tracking_runner_shell("org/motions/motion:latest", is_wandb=False)

  monkeypatch.setattr(MjlabOnPolicyRunner, "save", lambda *a, **kw: None)
  monkeypatch.setattr(runner_mod, "get_base_metadata", lambda *a: {})
  monkeypatch.setattr(runner_mod, "attach_metadata_to_onnx", lambda *a: None)

  checkpoint = tmp_path / "run-dir" / "model_100.pt"
  checkpoint.parent.mkdir()
  checkpoint.touch()

  with patch.object(runner_mod, "wandb") as mock_wandb:
    runner.export_policy_to_onnx = MagicMock()
    runner.save(str(checkpoint))

  mock_wandb.run.use_artifact.assert_not_called()

"""Export a G23 foot-placement RLMPC checkpoint for native deployment.

The exporter intentionally reconstructs only the actor.  It does not create a
MuJoCo environment, import the Python MPC extension, or use the generic joint
position metadata path.  The resulting ONNX graph includes the learned
empirical observation normalizer.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

OBS_DIM = 375
ACTION_DIM = 8
HIDDEN_DIMS = (512, 256, 128)

OBSERVATION_LAYOUT = (
  "5x75_newest_first(rpy[3],linear_velocity_world[3],angular_velocity_world[3],"
  "joint_pos[12],joint_vel[12],mpc_nominal_foothold_xy[8],measured_contact[4],"
  "mpc_contact[4],mpc_foot_force_div12[12],command_world[3],"
  "previous_offsets_m[8],gait_mode_one_hot[3])"
)

ACTION_LAYOUT = "FL_dx,FL_dy,FR_dx,FR_dy,HL_dx,HL_dy,HR_dx,HR_dy"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export the 375D -> 8D G23 RLMPC actor to ONNX."
  )
  parser.add_argument("checkpoint", type=Path, help="RSL-RL model_*.pt checkpoint")
  parser.add_argument("output", type=Path, help="Destination .onnx file")
  parser.add_argument(
    "--force", action="store_true", help="Replace an existing output file"
  )
  return parser.parse_args()


def _require_tensor_shape(
  state_dict: dict[str, Any], key: str, expected: tuple[int, ...]
) -> None:
  value = state_dict.get(key)
  if not isinstance(value, torch.Tensor):
    raise RuntimeError(f"Checkpoint is missing actor tensor {key!r}")
  if tuple(value.shape) != expected:
    raise RuntimeError(
      f"Actor tensor {key!r} has shape {tuple(value.shape)}, expected {expected}"
    )


def load_actor(checkpoint_path: Path) -> tuple[MLPModel, int | None]:
  """Reconstruct and strictly load the trained deterministic actor."""
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  if not isinstance(checkpoint, dict):
    raise RuntimeError("Checkpoint root must be a mapping")
  state_dict = checkpoint.get("actor_state_dict")
  if not isinstance(state_dict, dict):
    raise RuntimeError("Checkpoint does not contain actor_state_dict")

  # Fail before constructing/exporting if a different task checkpoint is used.
  _require_tensor_shape(state_dict, "obs_normalizer._mean", (1, OBS_DIM))
  _require_tensor_shape(state_dict, "mlp.0.weight", (HIDDEN_DIMS[0], OBS_DIM))
  _require_tensor_shape(state_dict, "mlp.6.weight", (ACTION_DIM, HIDDEN_DIMS[-1]))

  obs = TensorDict({"actor": torch.zeros(1, OBS_DIM)})
  actor = MLPModel(
    obs=obs,
    obs_groups={"actor": ["actor"]},
    obs_set="actor",
    output_dim=ACTION_DIM,
    hidden_dims=HIDDEN_DIMS,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.25,
      "std_range": (0.05, 0.50),
      "std_type": "scalar",
    },
  )
  actor.load_state_dict(state_dict, strict=True)
  actor.eval()

  iteration = checkpoint.get("iter")
  return actor, iteration if isinstance(iteration, int) else None


def checkpoint_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def attach_deployment_metadata(
  model_path: Path, checkpoint_path: Path, iteration: int | None
) -> None:
  model = onnx.load(model_path)
  metadata = {
    "contract": "mjlab_foot_state_history_v3",
    "checkpoint": checkpoint_path.name,
    "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
    "checkpoint_iteration": "unknown" if iteration is None else str(iteration),
    "observation_dim": str(OBS_DIM),
    "observation_layout": OBSERVATION_LAYOUT,
    "observation_normalization": "empirical_embedded_epsilon_0.01",
    "terrain_scan": "none",
    "observation_history": "5,newest_first,repeat_initial_frame",
    "frame_dim": "75",
    "previous_action": "mapped_foot_offsets_m",
    "joint_order": "FL_FR_HL_HR_each_HipX_HipY_Knee",
    "mpc_snapshot": "nominal_xy_com_relative,contact_binary,foot_force_body_div12",
    "policy_rate_hz": "50",
    "controller_rate_hz": "100",
    "mpc_rate_hz": "50",
    "gait_period_s": "0.5",
    "action_dim": str(ACTION_DIM),
    "action_layout": ACTION_LAYOUT,
    "action_clip": "[-1,1]",
    "action_scale_xy_m": "0.08,0.03",
    # The completed checkpoint was trained through the external controller's
    # direct ``Pf[x/y] += offset`` path.  Despite the mapper's historical
    # yaw-aligned wording, that path never rotates the action by base yaw.
    # Record the executed contract so deployment cannot silently add a frame
    # transform that was absent during training.
    "action_frame": "controller_xy_unrotated",
    "action_latching": "per_leg_stance_to_swing_entry",
  }
  onnx.helper.set_model_props(model, metadata)
  onnx.checker.check_model(model)
  onnx.save(model, model_path)


def verify_runtime_parity(actor: MLPModel, onnx_path: Path) -> float:
  """Compare deterministic PyTorch and ONNX Runtime outputs."""
  generator = torch.Generator(device="cpu").manual_seed(42)
  samples = torch.randn((8, OBS_DIM), generator=generator)
  samples[0].zero_()
  onnx_actor = cast(Any, actor.as_onnx(verbose=False)).eval()
  with torch.no_grad():
    expected = onnx_actor(samples).cpu().numpy()

  session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  inputs = session.get_inputs()
  outputs = session.get_outputs()
  if (
    len(inputs) != 1
    or inputs[0].name != "obs"
    or inputs[0].shape != [1, OBS_DIM]
    or inputs[0].type != "tensor(float)"
  ):
    raise RuntimeError(
      "Unexpected ONNX input contract: "
      f"{[(item.name, item.shape, item.type) for item in inputs]}"
    )
  if (
    len(outputs) != 1
    or outputs[0].name != "actions"
    or outputs[0].shape != [1, ACTION_DIM]
    or outputs[0].type != "tensor(float)"
  ):
    raise RuntimeError(
      "Unexpected ONNX output contract: "
      f"{[(item.name, item.shape, item.type) for item in outputs]}"
    )

  # The deployment graph has a fixed batch of one, matching the Rust runner.
  actual = np.concatenate(
    [session.run(["actions"], {"obs": row[None].numpy()})[0] for row in samples]
  )
  max_error = float(np.max(np.abs(expected - actual)))
  np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
  return max_error


def export_policy(checkpoint_path: Path, output_path: Path, force: bool) -> None:
  checkpoint_path = checkpoint_path.resolve(strict=True)
  if checkpoint_path.suffix != ".pt":
    raise RuntimeError(f"Checkpoint must be a .pt file: {checkpoint_path}")
  if output_path.suffix != ".onnx":
    raise RuntimeError(f"Output must use the .onnx suffix: {output_path}")
  if output_path.exists() and not force:
    raise FileExistsError(f"Output exists; pass --force to replace it: {output_path}")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  actor, iteration = load_actor(checkpoint_path)
  exportable = cast(Any, actor.as_onnx(verbose=False)).cpu().eval()

  fd, temporary_name = tempfile.mkstemp(
    prefix=f".{output_path.stem}.", suffix=".onnx", dir=output_path.parent
  )
  os.close(fd)
  temporary_path = Path(temporary_name)
  try:
    torch.onnx.export(
      exportable,
      exportable.get_dummy_inputs(),
      temporary_path,
      export_params=True,
      opset_version=18,
      input_names=exportable.input_names,
      output_names=exportable.output_names,
      dynamic_axes={},
      dynamo=False,
    )
    attach_deployment_metadata(temporary_path, checkpoint_path, iteration)
    max_error = verify_runtime_parity(actor, temporary_path)
    temporary_path.chmod(0o644)
    os.replace(temporary_path, output_path)
  finally:
    temporary_path.unlink(missing_ok=True)

  print(f"[OK] Exported {checkpoint_path.name} -> {output_path}")
  print(f"[OK] Contract: obs[1,{OBS_DIM}] -> actions[1,{ACTION_DIM}]")
  print(f"[OK] PyTorch/ONNX max absolute error: {max_error:.3e}")


def main() -> None:
  args = parse_args()
  export_policy(args.checkpoint, args.output, args.force)


if __name__ == "__main__":
  main()

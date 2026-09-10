"""Stop half a cycle apart and check alternating home steps in simulation."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_rl_mpc import _set_exact_manual_command  # noqa: E402

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.rl_mpc.config.g23.env_cfgs import (
  syncai_g23_pure_mpc_validation_env_cfg,
)

torch.set_num_threads(1)
cfg = syncai_g23_pure_mpc_validation_env_cfg()
cfg.scene.num_envs = 2
cfg.auto_reset = False
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
try:
  env.reset(seed=42)
  term = env.action_manager.get_term("foot_placement")
  command = env.command_manager.get_term("twist")
  controllers = [backend._runner.cMPC for backend in term._controllers]
  last_swing = [None, None]
  launches = [[], []]
  remaining = [None, None]
  for tick in range(300):
    t = tick * env.step_dt
    for i, stop_time in enumerate((3.0, 3.25)):
      vx = 0.2 if 2.0 <= t < stop_time else 0.0
      _set_exact_manual_command(command, i, (vx, 0.0, 0.0))
    _, _, done, timeout, _ = env.step(torch.zeros((2, 8)))
    assert not bool(done.any() or timeout.any()), f"Terminated at {t}s"
    for i, controller in enumerate(controllers):
      if int(controller.gait_mode) == 0:
        swing = controller.last_gait_inputs["swing_states"].reshape(4) > 0
        if swing.any():
          last_swing[i] = np.flatnonzero(swing).tolist()
      if (
        controller._stop_step_active
        and remaining[i] != controller._stop_steps_remaining
      ):
        launches[i].append(np.flatnonzero(controller._stopping_swing_legs).tolist())
        remaining[i] = controller._stop_steps_remaining
  result = []
  for i, controller in enumerate(controllers):
    assert last_swing[i] is not None
    opposite = [leg for leg in range(4) if leg not in last_swing[i]]
    assert launches[i] == [opposite, last_swing[i]], (last_swing[i], launches[i])
    assert int(controller.gait_mode) == 2
    assert bool(np.all(controller._foot_contacts))
    robot = env.scene["robot"].data
    rmse = (robot.joint_pos[i] - robot.default_joint_pos[i]).square().mean().sqrt()
    result.append(
      {
        "last_walking_swing": last_swing[i],
        "home_steps": launches[i],
        "final_mode": "STAND",
        "pose_rmse_rad": float(rmse),
      }
    )
  assert launches[0][0] != launches[1][0]
  (Path(__file__).parent / "stop_order_validation.json").write_text(
    json.dumps(result, indent=2)
  )
  print(json.dumps(result), flush=True)
finally:
  env.close()

"""Register the SyncAI G23 RL-MPC task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import syncai_g23_rl_mpc_env_cfg
from .rl_cfg import syncai_g23_rl_mpc_ppo_cfg

register_mjlab_task(
  task_id="Mjlab-RLMPC-Flat-SyncAI-G23",
  env_cfg=syncai_g23_rl_mpc_env_cfg(),
  play_env_cfg=syncai_g23_rl_mpc_env_cfg(play=True),
  rl_cfg=syncai_g23_rl_mpc_ppo_cfg(),
)

"""Goal-reaching navigation in a hospital populated by pedestrians."""

from mjlab.tasks.registry import register_mjlab_task

from .navigation_env_cfg import (
  hospital_navigation_env_cfg,
  navigation_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Navigation-Hospital",
  env_cfg=hospital_navigation_env_cfg(),
  play_env_cfg=hospital_navigation_env_cfg(play=True),
  rl_cfg=navigation_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Navigation-Hospital-Humans",
  env_cfg=hospital_navigation_env_cfg(with_humans=True),
  play_env_cfg=hospital_navigation_env_cfg(with_humans=True, play=True),
  # Same experiment as the empty hospital so that a policy trained there can be
  # resumed here, which is how the reference setup trains among pedestrians.
  rl_cfg=navigation_ppo_runner_cfg(),
)

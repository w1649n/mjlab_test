"""PPO configuration for G23 residual foot-placement learning."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def syncai_g23_rl_mpc_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Start close to the validated pure-MPC controller. With the 8 cm dx
        # and 3 cm dy scales, std=0.25 corresponds to 2 cm / 0.75 cm residuals.
        "init_std": 0.25,
        # Keep exploration bounded if a long resumed run pushes the learned
        # scalar standard deviation upward.  The previous run reached 0.44
        # with roughly one third of raw actions outside the action limit.
        "std_range": (0.05, 0.50),
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Foot-placement exploration was already increasing after locomotion
      # performance plateaued; a smaller bonus prevents it from dominating
      # the offset and action-rate penalties during fine-tuning.
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g23_rl_mpc_foot_state_history_v3",
    clip_actions=1.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )

目前最新 G23 RLMPC 訓練參數，整理日期 2026-09-16。

以 2026-09-10_11-11-39 run 保存設定為準；包括所有環境與 agent 原始欄位，保留 Python YAML tags，未執行其構造器。

**agent.yaml（完整保存設定）**

```yaml
seed: 42
num_steps_per_env: 24
max_iterations: 5000
obs_groups:
  actor: !!python/tuple
  - actor
  critic: !!python/tuple
  - critic
save_interval: 100
experiment_name: g23_rl_mpc_foot_state_history_v3
run_name: ''
logger: wandb
wandb_project: mjlab
wandb_tags: !!python/tuple []
resume: true
load_run: \A2026\-09\-10_10\-51\-22\Z
load_checkpoint: \Amodel_400\.pt\Z
clip_actions: 1.0
upload_model: true
class_name: OnPolicyRunner
actor:
  hidden_dims: !!python/tuple
  - 512
  - 256
  - 128
  activation: elu
  obs_normalization: true
  cnn_cfg: null
  distribution_cfg:
    class_name: GaussianDistribution
    init_std: 0.25
    std_range: !!python/tuple
    - 0.05
    - 0.5
    std_type: scalar
  rnn_type: null
  rnn_hidden_dim: 256
  rnn_num_layers: 1
  class_name: MLPModel
critic:
  hidden_dims: !!python/tuple
  - 512
  - 256
  - 128
  activation: elu
  obs_normalization: true
  cnn_cfg: null
  distribution_cfg: null
  rnn_type: null
  rnn_hidden_dim: 256
  rnn_num_layers: 1
  class_name: MLPModel
algorithm:
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 0.001
  schedule: adaptive
  gamma: 0.99
  lam: 0.95
  entropy_coef: 0.001
  desired_kl: 0.01
  max_grad_norm: 1.0
  value_loss_coef: 1.0
  use_clipped_value_loss: true
  clip_param: 0.2
  normalize_advantage_per_mini_batch: false
  optimizer: adam
  share_cnn_encoders: false
  class_name: PPO
```

**env.yaml（完整保存設定）**

```yaml
decimation: 4
scene:
  num_envs: 32
  env_spacing: 2.0
  terrain:
    init_state:
      pos: !!python/tuple
      - 0.0
      - 0.0
      - 0.0
      rot: !!python/tuple
      - 1.0
      - 0.0
      - 0.0
      - 0.0
      lin_vel: !!python/tuple
      - 0.0
      - 0.0
      - 0.0
      ang_vel: !!python/tuple
      - 0.0
      - 0.0
      - 0.0
      joint_pos:
        .*: 0.0
      joint_vel:
        .*: 0.0
    spec_fn: !!python/name:mjlab.entity.entity.%3Clambda%3E ''
    articulation: null
    sort_actuators: false
    lights: !!python/tuple
    - name: sun
      body: world
      mode: fixed
      target: null
      type: directional
      castshadow: true
      pos: !!python/tuple
      - 0.0
      - 0.0
      - 1.5
      dir: !!python/tuple
      - 0.0
      - 0.0
      - -1.0
      cutoff: 45.0
      exponent: 10.0
      diffuse: null
      specular: null
      ambient: null
      active: true
      attenuation: !!python/tuple
      - 1.0
      - 0.0
      - 0.0
    cameras: !!python/tuple []
    textures: !!python/tuple
    - name: groundplane
      type: 2d
      builtin: checker
      rgb1: !!python/tuple
      - 0.2
      - 0.3
      - 0.4
      rgb2: !!python/tuple
      - 0.1
      - 0.2
      - 0.3
      width: 300
      height: 300
      mark: edge
      markrgb: !!python/tuple
      - 0.8
      - 0.8
      - 0.8
      random: 0.01
      file: null
      cubefiles: null
      gridsize: null
      gridlayout: null
      nchannel: 3
      hflip: false
      vflip: false
    materials: !!python/tuple
    - name: groundplane
      rgba: !!python/tuple
      - 1.0
      - 1.0
      - 1.0
      - 1.0
      texuniform: true
      texrepeat: !!python/tuple
      - 4.0
      - 4.0
      reflectance: 0.2
      texture: groundplane
      geom_names_expr: !!python/tuple
      - terrain$
    meshes: !!python/tuple []
    geoms: !!python/tuple []
    collisions: !!python/tuple []
    terrain_type: generator
    terrain_generator:
      seed: 42
      curriculum: true
      size: !!python/tuple
      - 8.0
      - 8.0
      border_width: 10.0
      border_height: 1.0
      num_rows: 6
      num_cols: 3
      color_scheme: height
      sub_terrains:
        flat:
          proportion: 0.4
          size: !!python/tuple
          - 8.0
          - 8.0
          flat_patch_sampling: null
        random_rough:
          proportion: 0.4
          size: !!python/tuple
          - 8.0
          - 8.0
          flat_patch_sampling: null
          noise_range: !!python/tuple
          - 0.0
          - 0.08
          noise_step: 0.01
          downsampled_scale: 0.4
          horizontal_scale: 0.1
          vertical_scale: 0.005
          base_thickness_ratio: 1.0
          border_width: 0.5
          scale_with_difficulty: true
        low_wave:
          proportion: 0.2
          size: !!python/tuple
          - 8.0
          - 8.0
          flat_patch_sampling: null
          amplitude_range: !!python/tuple
          - 0.01
          - 0.04
          num_waves: 3
          horizontal_scale: 0.1
          vertical_scale: 0.005
          base_thickness_ratio: 0.25
          border_width: 0.5
      difficulty_range: !!python/tuple
      - 0.0
      - 1.0
      add_lights: true
    env_spacing: 2.0
    max_init_terrain_level: 0
    num_envs: 32
    debug_vis: false
  entities:
    robot:
      init_state:
        pos: !!python/tuple
        - 0.0
        - 0.0
        - 0.32
        rot: !!python/tuple
        - 1.0
        - 0.0
        - 0.0
        - 0.0
        lin_vel: !!python/tuple
        - 0.0
        - 0.0
        - 0.0
        ang_vel: !!python/tuple
        - 0.0
        - 0.0
        - 0.0
        joint_pos:
          .*_HipX_joint: 0.0
          .*_HipY_joint: -0.8
          .*_Knee_joint: 1.6
        joint_vel:
          .*: 0.0
      spec_fn: !!python/name:mjlab.asset_zoo.robots.syncai_g23.g23_constants.get_spec ''
      articulation:
        actuators: !!python/tuple
        - target_names_expr: !!python/tuple
          - .*_Hip[XY]_joint
          transmission_type: !!python/object/apply:mjlab.actuator.actuator.TransmissionType
          - joint
          armature: 0.0
          frictionloss: null
          viscous_damping: null
          delay_min_lag: 0
          delay_max_lag: 0
          delay_hold_prob: 0.0
          delay_update_period: 0
          delay_per_env_phase: true
          effort_limit: 24.0
          gear: 1.0
        - target_names_expr: !!python/tuple
          - .*_Knee_joint
          transmission_type: !!python/object/apply:mjlab.actuator.actuator.TransmissionType
          - joint
          armature: 0.0
          frictionloss: null
          viscous_damping: null
          delay_min_lag: 0
          delay_max_lag: 0
          delay_hold_prob: 0.0
          delay_update_period: 0
          delay_per_env_phase: true
          effort_limit: 36.0
          gear: 1.0
        soft_joint_pos_limit_factor: 0.99
      sort_actuators: true
      lights: !!python/tuple []
      cameras: !!python/tuple []
      textures: !!python/tuple []
      materials: !!python/tuple []
      meshes: !!python/tuple []
      geoms: !!python/tuple []
      collisions: !!python/tuple
      - geom_names_expr: !!python/tuple
        - .*_collision\d*
        contype: 1
        conaffinity: 1
        condim:
          ^(FL|FR|HL|HR)_foot_collision$: 6
          .*_collision\d*: 1
        priority:
          ^(FL|FR|HL|HR)_foot_collision$: 1
          .*_collision\d*: 0
        friction:
          ^(FL|FR|HL|HR)_foot_collision$: !!python/tuple
          - 1.0
          - 0.005
          - 0.0005
        solref: !!python/tuple
        - 0.01
        - 1
        solimp: null
        margin: null
        gap: null
        solmix: null
        disable_other_geoms: true
  sensors: !!python/tuple
  - name: feet_ground_contact
    primary:
      mode: geom
      pattern: !!python/tuple
      - FL_foot_collision
      - FR_foot_collision
      - HL_foot_collision
      - HR_foot_collision
      entity: robot
      exclude: !!python/tuple []
    secondary:
      mode: body
      pattern: terrain
      entity: null
      exclude: !!python/tuple []
    fields: !!python/tuple
    - found
    - force
    reduce: netforce
    num_slots: 1
    secondary_policy: first
    track_air_time: true
    global_frame: false
    history_length: 0
    debug: false
  - name: self_collision
    primary:
      mode: subtree
      pattern: TORSO
      entity: robot
      exclude: !!python/tuple []
    secondary:
      mode: subtree
      pattern: TORSO
      entity: robot
      exclude: !!python/tuple []
    fields: !!python/tuple
    - found
    - force
    reduce: none
    num_slots: 1
    secondary_policy: first
    track_air_time: false
    global_frame: false
    history_length: 4
    debug: false
  - name: thigh_ground_touch
    primary:
      mode: geom
      pattern: !!python/tuple
      - FL_thigh_collision
      - FR_thigh_collision
      - HL_thigh_collision
      - HR_thigh_collision
      entity: robot
      exclude: !!python/tuple []
    secondary:
      mode: body
      pattern: terrain
      entity: null
      exclude: !!python/tuple []
    fields: !!python/tuple
    - found
    - force
    reduce: maxforce
    num_slots: 1
    secondary_policy: first
    track_air_time: false
    global_frame: false
    history_length: 4
    debug: false
  - name: shank_ground_touch
    primary:
      mode: geom
      pattern: !!python/tuple
      - FL_shank_collision
      - FL_shank_collision1
      - FR_shank_collision
      - FR_shank_collision1
      - HL_shank_collision
      - HL_shank_collision1
      - HR_shank_collision
      - HR_shank_collision1
      entity: robot
      exclude: !!python/tuple []
    secondary:
      mode: body
      pattern: terrain
      entity: null
      exclude: !!python/tuple []
    fields: !!python/tuple
    - found
    - force
    reduce: maxforce
    num_slots: 1
    secondary_policy: first
    track_air_time: false
    global_frame: false
    history_length: 4
    debug: false
  - name: torso_ground_touch
    primary:
      mode: geom
      pattern: !!python/tuple
      - torso_collision
      - torso_collision1
      entity: robot
      exclude: !!python/tuple []
    secondary:
      mode: body
      pattern: terrain
      entity: null
      exclude: !!python/tuple []
    fields: !!python/tuple
    - found
    - force
    reduce: maxforce
    num_slots: 1
    secondary_policy: first
    track_air_time: false
    global_frame: false
    history_length: 4
    debug: false
  extent: 2.0
  spec_fn: null
observations:
  actor:
    terms:
      state_history:
        func: &id001 !!python/name:mjlab.tasks.rl_mpc.mdp.observations.mpc_state_history ''
        params: {}
        noise: null
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        history_length: 0
        flatten_history_dim: true
    concatenate_terms: true
    concatenate_dim: -1
    enable_corruption: false
    history_length: null
    flatten_history_dim: true
    nan_policy: disabled
    nan_check_per_term: true
  critic:
    terms:
      state_history:
        func: *id001
        params: {}
        noise: null
        clip: null
        scale: null
        delay_min_lag: 0
        delay_max_lag: 0
        delay_per_env: true
        delay_hold_prob: 0.0
        delay_update_period: 0
        delay_per_env_phase: true
        history_length: 0
        flatten_history_dim: true
    concatenate_terms: true
    concatenate_dim: -1
    enable_corruption: false
    history_length: null
    flatten_history_dim: true
    nan_policy: disabled
    nan_check_per_term: true
actions:
  foot_placement:
    entity_name: robot
    clip: null
    command_name: twist
    controller_root: null
    formulation: syncai
    controller_dt: 0.01
    iterations_between_mpc: 2
    gait_period: 0.5
    flat_ground: false
    body_mass: 15.7
    foot_landing_height: 0.019
    contact_sensor_name: feet_ground_contact
    terrain_sensor_name: null
    terrain_scan_miss_policy: error
    terrain_scan_min_valid_fraction: 0.9
    enable_stand_mode: true
    stand_enter_linear_speed: 0.03
    stand_enter_yaw_rate: 0.05
    stand_exit_linear_speed: 0.08
    stand_exit_yaw_rate: 0.1
    stand_command_hold_time: 0.2
    stand_contact_hold_time: 0.05
    stand_contact_loss_time: 0.05
    stand_arm_timeout: 0.2
    stand_max_linear_speed: 0.1
    stand_max_vertical_speed: 0.1
    stand_max_yaw_rate: 0.2
    stand_max_roll_pitch_rate: 0.2
    stand_foot_search_rate: 0.05
    stand_foot_search_depth: 0.08
    stand_mpc_xy_position_weight: 5.0
    stand_mpc_vxy_weight: 3.0
    stand_kp_cartesian: !!python/tuple
    - 80.0
    - 80.0
    - 30.0
    stand_kd_cartesian: !!python/tuple
    - 8.0
    - 8.0
    - 8.0
    stand_hipx_kp: 12.0
    stand_hipx_kd: 1.0
    symmetric_residuals: true
    stop_reposition_steps: 2
    foot_offset_scale_xy: !!python/tuple
    - 0.08
    - 0.03
    joint_names: !!python/tuple
    - FL_HipX_joint
    - FL_HipY_joint
    - FL_Knee_joint
    - FR_HipX_joint
    - FR_HipY_joint
    - FR_Knee_joint
    - HL_HipX_joint
    - HL_HipY_joint
    - HL_Knee_joint
    - HR_HipX_joint
    - HR_HipY_joint
    - HR_Knee_joint
    torque_limits: !!python/tuple
    - 24.0
    - 24.0
    - 36.0
    - 24.0
    - 24.0
    - 36.0
    - 24.0
    - 24.0
    - 36.0
    - 24.0
    - 24.0
    - 36.0
    fail_on_controller_error: true
    controller_factory: null
events:
  reset_base:
    func: !!python/name:mjlab.envs.mdp.events.reset_root_state_uniform ''
    params:
      pose_range:
        x: !!python/tuple
        - -0.5
        - 0.5
        y: !!python/tuple
        - -0.5
        - 0.5
        z: !!python/tuple
        - 0.01
        - 0.05
        yaw: !!python/tuple
        - -3.14
        - 3.14
      velocity_range: {}
    mode: reset
    interval_range_s: null
    is_global_time: false
    min_step_count_between_reset: 0
  reset_robot_joints:
    func: !!python/name:mjlab.envs.mdp.events.reset_joints_by_offset ''
    params:
      position_range: !!python/tuple
      - 0.0
      - 0.0
      velocity_range: !!python/tuple
      - 0.0
      - 0.0
      asset_cfg:
        name: robot
        joint_names: !!python/tuple
        - .*
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    mode: reset
    interval_range_s: null
    is_global_time: false
    min_step_count_between_reset: 0
seed: 42
sim:
  nconmax: 256
  njmax: 1500
  contact_sensor_maxmatch: 500
  broadphase: null
  broadphase_filter: null
  ls_parallel: null
  mujoco:
    timestep: 0.005
    integrator: implicitfast
    impratio: 10
    cone: elliptic
    jacobian: auto
    solver: newton
    iterations: 10
    tolerance: 1.0e-08
    ls_iterations: 20
    ls_tolerance: 0.01
    ccd_iterations: 16
    gravity: !!python/tuple
    - 0.0
    - 0.0
    - -9.81
    disableflags: !!python/tuple []
    enableflags: !!python/tuple []
  nan_guard:
    enabled: false
    buffer_size: 100
    output_dir: /tmp/mjlab/nan_dumps
    max_envs_to_dump: 5
viewer:
  origin_type: !!python/object/apply:mjlab.viewer.viewer_config.OriginType
  - 4
  height: 240
  width: 320
  enable_reflections: true
  enable_shadows: true
  geom_group: !!python/tuple
  - 1
  - 1
  - 1
  - 0
  - 0
  - 0
  site_group: !!python/tuple
  - 1
  - 1
  - 1
  - 0
  - 0
  - 0
  env_idx: 0
  max_extra_envs: 2
  reward_bar_max_terms: 20
  distance: 1.5
  elevation: -10.0
  azimuth: 90.0
  fovy: null
  lookat: !!python/tuple
  - 0.0
  - 0.0
  - 0.0
  entity_name: robot
  body_name: TORSO
episode_length_s: 20.0
rewards:
  track_linear_velocity:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.track_linear_velocity_xy ''
    params:
      command_name: twist
      std: 0.5
    weight: 2.0
  track_angular_velocity:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.track_angular_velocity_yaw ''
    params:
      command_name: twist
      std: 0.7071067811865476
    weight: 2.0
  upright:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.upright ''
    params:
      std: 0.4472135954999579
      asset_cfg:
        name: robot
        joint_names: null
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: !!python/tuple
        - TORSO
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: 1.0
  pose:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.variable_posture ''
    params:
      asset_cfg:
        name: robot
        joint_names: !!python/tuple
        - .*
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
      command_name: twist
      std_standing:
        .*_HipX_joint: 0.05
        .*_HipY_joint: 0.05
        .*_Knee_joint: 0.1
      std_walking:
        .*_HipX_joint: 0.2
        .*_HipY_joint: 0.3
        .*_Knee_joint: 0.6
      std_running:
        .*_HipX_joint: 0.25
        .*_HipY_joint: 0.4
        .*_Knee_joint: 0.8
      walking_threshold: 0.05
      running_threshold: 1.5
    weight: 1.0
  body_ang_vel:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.body_angular_velocity_penalty ''
    params:
      asset_cfg:
        name: robot
        joint_names: null
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: !!python/tuple
        - TORSO
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: -0.02
  angular_momentum:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.angular_momentum_penalty ''
    params:
      sensor_name: robot/root_angmom
    weight: 0.0
  dof_pos_limits:
    func: !!python/name:mjlab.envs.mdp.rewards.joint_pos_limits ''
    params: {}
    weight: -1.0
  action_rate_l2:
    func: !!python/name:mjlab.envs.mdp.rewards.action_rate_l2 ''
    params: {}
    weight: -0.02
  air_time:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.feet_air_time ''
    params:
      sensor_name: feet_ground_contact
      threshold_min: 0.05
      threshold_max: 0.5
      command_name: twist
      command_threshold: 0.5
    weight: 0.0
  foot_slip:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.feet_slip ''
    params:
      sensor_name: feet_ground_contact
      command_name: twist
      command_threshold: 0.05
      asset_cfg:
        name: robot
        joint_names: null
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: !!python/tuple
        - FL
        - FR
        - HL
        - HR
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: -0.1
  soft_landing:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.soft_landing ''
    params:
      sensor_name: feet_ground_contact
      command_name: twist
      command_threshold: 0.05
    weight: -5.0e-05
  excessive_foot_air_time:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.feet_excessive_air_time ''
    params:
      sensor_name: feet_ground_contact
      max_air_time: 0.55
      max_excess_time: 0.5
    weight: -4.0
  standing_missing_foot_contacts:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.feet_contact_count_standing ''
    params:
      sensor_name: feet_ground_contact
      command_name: twist
      required_contacts: 4
      command_threshold: 0.05
    weight: -0.5
  vertical_velocity_l2:
    func: !!python/name:mjlab.tasks.velocity.mdp.rewards.vertical_velocity_l2 ''
    params: {}
    weight: -0.5
  electrical_power:
    func: !!python/name:mjlab.envs.mdp.rewards.electrical_power_cost ''
    params:
      asset_cfg:
        name: robot
        joint_names: !!python/tuple
        - .*
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: -2.0e-05
  joint_acc_l2:
    func: !!python/name:mjlab.envs.mdp.rewards.joint_acc_l2 ''
    params: {}
    weight: -2.5e-07
  action_acc_l2:
    func: !!python/name:mjlab.envs.mdp.rewards.action_acc_l2 ''
    params: {}
    weight: -0.005
  self_collisions:
    func: &id002 !!python/name:mjlab.tasks.velocity.mdp.rewards.self_collision_cost ''
    params:
      sensor_name: self_collision
    weight: -0.1
  shank_collision:
    func: *id002
    params:
      sensor_name: shank_ground_touch
      force_threshold: 10.0
    weight: -0.25
  torso_collision:
    func: *id002
    params:
      sensor_name: torso_ground_touch
    weight: -0.1
  joint_torques_l2:
    func: !!python/name:mjlab.envs.mdp.rewards.joint_torques_l2 ''
    params:
      asset_cfg:
        name: robot
        joint_names: null
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names:
        - .*
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: -2.5e-05
  foot_placement_offset_l2:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.rewards.foot_placement_offset_l2 ''
    params: {}
    weight: -0.02
  foot_placement_lateral_offset_l2:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.rewards.foot_placement_lateral_offset_l2 ''
    params: {}
    weight: -0.05
  straight_foot_placement_symmetry:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.rewards.straight_foot_placement_symmetry ''
    params: {}
    weight: -0.1
  stopped_joint_pose_l2:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.rewards.stopped_joint_pose_l2 ''
    params:
      asset_cfg:
        name: robot
        joint_names: !!python/tuple
        - .*
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    weight: -1.0
terminations:
  time_out:
    func: !!python/name:mjlab.envs.mdp.terminations.time_out ''
    params: {}
    time_out: true
  out_of_terrain_bounds:
    func: !!python/name:mjlab.tasks.velocity.mdp.terminations.out_of_terrain_bounds ''
    params: {}
    time_out: true
  illegal_contact:
    func: &id003 !!python/name:mjlab.tasks.velocity.mdp.terminations.illegal_contact ''
    params:
      sensor_name: thigh_ground_touch
      force_threshold: 60.0
      history_count_threshold: 2
    time_out: false
  shank_illegal_contact:
    func: *id003
    params:
      sensor_name: shank_ground_touch
      force_threshold: 180.0
      history_count_threshold: 4
    time_out: false
  torso_illegal_contact:
    func: *id003
    params:
      sensor_name: torso_ground_touch
      force_threshold: 80.0
      history_count_threshold: 3
    time_out: false
  prolonged_foot_air_time:
    func: !!python/name:mjlab.tasks.velocity.mdp.terminations.prolonged_foot_air_time ''
    params:
      sensor_name: feet_ground_contact
      max_air_time: 2.0
    time_out: false
commands:
  twist:
    resampling_time_range: !!python/tuple
    - 3.0
    - 8.0
    debug_vis: true
    entity_name: robot
    heading_command: true
    heading_control_stiffness: 0.5
    rel_standing_envs: 0.25
    rel_heading_envs: 0.1
    rel_world_envs: 0.0
    rel_forward_envs: 0.4
    rel_backward_envs: 0.1
    init_velocity_prob: 0.0
    sample_single_axis_commands: true
    ranges:
      lin_vel_x: !!python/tuple
      - -0.2
      - 0.5
      lin_vel_y: !!python/tuple
      - -0.15
      - 0.15
      ang_vel_z: !!python/tuple
      - -0.5
      - 0.5
      heading: !!python/tuple
      - -3.141592653589793
      - 3.141592653589793
    viz:
      z_offset: 0.45
      scale: 0.5
curriculum:
  terrain_levels:
    func: !!python/name:mjlab.tasks.velocity.mdp.curriculums.terrain_levels_vel ''
    params:
      command_name: twist
metrics:
  mean_action_acc:
    func: !!python/name:mjlab.envs.mdp.metrics.mean_action_acc ''
    params: {}
    per_substep: false
    reduce: mean
  mean_abs_lateral_foot_placement_action:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.metrics.mean_abs_lateral_foot_placement_action ''
    params: {}
    per_substep: false
    reduce: mean
  max_abs_hip_x:
    func: !!python/name:mjlab.tasks.rl_mpc.mdp.metrics.max_abs_hip_x ''
    params:
      asset_cfg:
        name: robot
        joint_names: !!python/tuple
        - .*_HipX_joint
        joint_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        body_names: null
        body_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        geom_names: null
        geom_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        site_names: null
        site_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        actuator_names: null
        actuator_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        tendon_names: null
        tendon_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        camera_names: null
        camera_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        light_names: null
        light_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        material_names: null
        material_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        texture_names: null
        texture_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        pair_names: null
        pair_ids: !!python/object/apply:builtins.slice
        - null
        - null
        - null
        preserve_order: false
    per_substep: false
    reduce: max
recorders: {}
is_finite_horizon: false
auto_reset: true
scale_rewards_by_dt: true
```

**與目前程式預設值的差異**

```json
[
  {
    "parameter": "env.scene.terrain.terrain_generator.sub_terrains.flat.size",
    "saved_run": [
      "8.0",
      "8.0"
    ],
    "current_default": [
      "10.0",
      "10.0"
    ]
  },
  {
    "parameter": "env.scene.terrain.terrain_generator.sub_terrains.low_wave.size",
    "saved_run": [
      "8.0",
      "8.0"
    ],
    "current_default": [
      "10.0",
      "10.0"
    ]
  },
  {
    "parameter": "env.scene.terrain.terrain_generator.sub_terrains.random_rough.size",
    "saved_run": [
      "8.0",
      "8.0"
    ],
    "current_default": [
      "10.0",
      "10.0"
    ]
  },
  {
    "parameter": "env.scene.terrain.num_envs",
    "saved_run": "32",
    "current_default": "1"
  },
  {
    "parameter": "env.scene.num_envs",
    "saved_run": "32",
    "current_default": "1"
  },
  {
    "parameter": "env.seed",
    "saved_run": "42",
    "current_default": "null"
  },
  {
    "parameter": "agent.load_checkpoint",
    "saved_run": "\\Amodel_400\\.pt\\Z",
    "current_default": "model_.*.pt"
  },
  {
    "parameter": "agent.resume",
    "saved_run": "true",
    "current_default": "false"
  },
  {
    "parameter": "agent.load_run",
    "saved_run": "\\A2026\\-09\\-10_10\\-51\\-22\\Z",
    "current_default": ".*"
  }
]
```

補充說明：上方 current_default 是建構環境前的設定。TerrainGenerator 會將 sub_terrains.size 覆寫成父層 size，所以子地形的 10×10 預設與保存的 8×8 差異不代表有效訓練地形改大；實際使用父層 8×8。env.scene.num_envs=32、seed=42 和 resume 條件來自最新 run 的啟動設定。

目前 MPC 原始碼補充（這些並非完整保存於 run YAML 的 controller 快照）：

- Controller：vendored rl-mpc-locomotion，SyncAI formulation、qpOASES；horizon 10 × 0.02 s = 0.2 s；alpha=1e-5。
- MPC target height 0.28 m，body mass 15.7 kg（adapter 覆寫），QP friction coefficient 0.4；模擬足端 friction=(1.0,0.005,0.0005)。
- TROT weights，state order roll,pitch,yaw,x,y,z,wx,wy,wz,vx,vy,vz,gravity：
  [100,100,70,0,0,25,0.2,0.2,1,0.3,0.3,0.5,0]。
- Swing Cartesian Kp=(700,700,150)，Kd=(7,7,7)；TROT stance Cartesian Kp=0、Kd=(7,7,7)，joint Kp=0、Kd=0.2。
- 一般行走與停止 home-step 的 swing height 都由每次更新的 body height/3 決定，約 0.0933 m。setHeight(0.05) 僅為 firstRun 初始化，隨後會被覆寫，並非獨立的停止抬腳高度。
- 直行判定 |vy|≤0.03 m/s、|yaw rate|≤0.05 rad/s；同對左右腳 X 取平均、Y 鏡射，每完整 gait cycle 共用 residual sample。
- 橫移／轉向 residual cap：每腳 X ±0.015 m、Y ±0.0075 m；paired X difference≤0.005 m、Y sum≤0.005 m。
- stopped_joint_pose_l2 的函式預設門檻：線速度≤0.03 m/s、yaw rate≤0.05 rad/s 且 STOPPING/STAND。

來源：[backend](../../src/mjlab/tasks/rl_mpc/controller/backend.py)、[controller](../../thirdparty/rl-mpc-locomotion/MPC_Controller/convex_MPC/ConvexMPCLocomotion.py)、[Quadruped](../../thirdparty/rl-mpc-locomotion/MPC_Controller/common/Quadruped.py)、[reward functions](../../src/mjlab/tasks/rl_mpc/mdp/rewards.py)。

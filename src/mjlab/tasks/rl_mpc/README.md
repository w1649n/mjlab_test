# SyncAI G23 RL-MPC

`Mjlab-RLMPC-Flat-SyncAI-G23` trains an eight-dimensional residual
foot-placement policy around the existing G23 convex MPC controller.

## Control contract

The policy action is leg-major and uses the external controller's unrotated XY
coordinates:

```text
[FL_dx, FL_dy, FR_dx, FR_dy, HL_dx, HL_dy, HR_dx, HR_dy]
```

Normalized actions are clipped to `[-1, 1]`, then scaled to `+/-0.08 m` in the
fore-aft direction and `+/-0.03 m` laterally. Each leg consumes the latest
residual when it enters swing and keeps that residual until touchdown.
Joint-angle residuals are intentionally not part of this task.

The action is added directly through the controller's ``Pf[x/y] += offset``
path.  Do not rotate it by base yaw again during export or deployment.

The baseline swing planner reads the existing yaw-aligned terrain scan and
interpolates the surface height at each planned foothold. This changes only the
touchdown Z used by pure MPC; the policy still outputs planar XY offsets only.
On episode reset, a non-zero command waits until four-foot support is confirmed
before the controller releases the first diagonal pair into swing.

The simulation and controller both run at `100 Hz` (`physics_dt=0.005`,
`decimation=2`). The MPC and gait segment advance every three controller ticks:
the ten-segment trot is `0.30 s` (`3.33 Hz`), with `0.15 s` stance and swing.

At zero command the hybrid controller holds all four feet in `STAND`. A command
above the exit threshold starts `TROT`; returning to zero enters `STOPPING`,
finishes only the feet already in swing, and commits to `STAND` after continuous
four-foot contact. The policy observes this mode as a three-value one-hot vector.

Stage-one random commands sample one axis at a time from `vx=[-0.2, 0.5] m/s`,
`vy=+/-0.15 m/s`, or `wz=+/-0.5 rad/s`. These are the zero-residual limits that
pass the maximum-terrain gate. Mixed-command fine-tuning should be enabled only
after the residual policy has learned a stable terrain-aware foothold baseline.

## Terrain contract

The curriculum starts at level zero, where all three terrain columns are real
flat planes. It then progresses through six levels to:

- ordinary flat ground;
- smooth random uneven ground; and
- low continuous waves.

The generated surface span is capped at `0.08 m` peak-to-peak. Stair terrain is
rejected by the terrain-contract validator.

Training and play tolerate isolated ray misses at heightfield-patch seams by
keeping the missed ray's query XY and copying only Z from the nearest valid ray
in the same environment. At least 90% of an environment's rays must remain
valid; otherwise that environment receives zero torque for the step and resets
through `terrain_scan_failed`. The maximum miss fraction and cumulative action
term counters make these events visible. A severe first scan, or simultaneous
severe scans in multiple environments, still stops the run because it indicates
a systemic sensor or terrain configuration failure. Pure-MPC validation
deliberately uses the strict `error` policy, so even one missed ray fails the
pre-training gate.

## Native solver for the mjlab interpreter

The controller is bundled in `thirdparty/rl-mpc-locomotion` at the mjlab
repository root. It uses `RobotRunnerMin` / `ConvexMPCLocomotion`, with the
`syncai` formulation and qpOASES solver through the `mpc_osqp` C++ extension.
Build the extension for mjlab's active Python (currently CPython 3.13):

```bash
uv run python scripts/build_rl_mpc_solver.py
```

The helper uses a modern installed `pybind11`, or the headers vendored by
PyTorch. Runtime and build commands default to the bundled source; no sibling
checkout is required. To use a different source, set
`MJLAB_RLMPC_CONTROLLER_ROOT` for both commands, or pass `--controller-root`
to the build helper and set `LegacyMpcBackendCfg.controller_root` at runtime.
Rebuild after changing native sources or Python versions. This bundled source
is for repository-based use; it is not included in the mjlab wheel.

## Required pre-training gate

```bash
uv run python scripts/validate_rl_mpc.py --levels 0 5 --seconds 5 --device cpu
```

With no `--command` argument, the gate checks the complete one-axis first-stage
command envelope: `vx=-0.2/+0.5 m/s`, `vy=+/-0.15 m/s`, and
`wz=+/-0.5 rad/s`. It runs zero residual actions on the flat and maximum-height
levels with automatic reset disabled, then checks `STAND -> TROT -> STOPPING ->
STAND` in one episode. Any termination, base height below 0.20 m, excessive
standing/moving HipX, solver failure, incomplete final stand, commanded-axis
tracking below the configured ratio, or excessive tracking MAE fails the gate.
The first 0.5 s is excluded from tracking statistics. A custom `--command`
outside the training envelope is rejected instead of silently validating its
clamped replacement. Default tracking limits are ratio `>=0.50` for linear
commands and `>=0.65` for yaw, with MAE `<=0.15 m/s` and `<=0.18 rad/s`.

The SyncAI solver integrates the requested yaw rate into the horizon's desired
yaw trajectory. This prevents the high yaw-position cost from opposing the
angular-rate command; the zero-residual baseline tracks the tested yaw limits
at roughly 70--78% before residual training.

For a quick single-command diagnostic:

```bash
uv run python scripts/validate_rl_mpc.py \
  --levels 0 5 --seconds 5 --command 0.2 0 0 --device cpu
```

## Training

Run the gate first, then start with a modest environment count:

```bash
uv run train Mjlab-RLMPC-Flat-SyncAI-G23 \
  --env.scene.num-envs 32 --gpu-ids "[0]"
```

The MuJoCo simulation can be parallelized on a GPU, but this integration still
owns one stateful CPU MPC instance per environment and invokes those instances
sequentially. Benchmark 32 environments before increasing the count.

The gait-mode observation changes the actor input from 231 to 234 values and
the critic input from 255 to 258 values. Start a new run; checkpoints created
before this STAND integration are not shape-compatible. For a later mixed-axis
fine-tune, set `twist.sample_single_axis_commands = False`, keep the same action
contract, and first rerun the gate with representative combined commands.

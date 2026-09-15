# Third-party MPC controller

`rl-mpc-locomotion/` is a source snapshot of the workspace controller from
`../rl-mpc-locomotion`, based on upstream
https://github.com/silvery107/rl-mpc-locomotion at commit
`df944efacdd403c05600907a3866dfe400f6d067`, copied on 2026-09-11.
It includes the local G23 / SyncAI, gait, stand, residual and solver fixes;
it is not an unmodified upstream release. `SOURCE_STATUS.txt` records the
source checkout's changes when copied.

Included: `MPC_Controller`, native `setup.py`, Eigen, OSQP, qpOASES, pybind11,
and task YAML files read by `Quadruped`. Training environments, models, logs,
Git metadata and prebuilt binaries are omitted. Original licenses and source
notices are retained; dependency licenses remain in their `extern` folders.

From the mjlab repository root:

```bash
uv run python scripts/build_rl_mpc_solver.py
```

The helper selects modern pybind11 headers from the active environment or
PyTorch for current Python compatibility. Runtime prefers this copy; explicit
`controller_root` and `MJLAB_RLMPC_CONTROLLER_ROOT` can override it. The native
module is named `mpc_osqp`, but the controller selects qpOASES and defaults to
the SyncAI formulation. Rebuild the extension after native-source changes.

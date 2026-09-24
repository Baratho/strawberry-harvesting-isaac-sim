# Strawberry Harvesting Experiments

This project separates the shared harvesting pipeline from the ten scene
configurations. Each Isaac Sim experiment runs in its own process, and the
batch launcher waits for that process to exit before starting the next one.

## Directory structure

```text
strawberry_harvest_experiments/
├── harvest_pipeline.py          # Shared planning, grasping, placing and logging
├── trial_configs.py             # Scene parameters for trials 01 through 10
├── run_trials_sequential.py     # Blocking one-at-a-time batch launcher
├── README.md                    # Setup and usage
└── results/                     # Created and updated during execution
    ├── trial_summary.csv        # One summary row per trial
    ├── batch_status.json        # Process exit status from the sequential launcher
    └── trial_XX/
        ├── candidate_paths.csv  # Every feasible angle and seed path
        ├── summary.json         # Scene configuration and final result
        └── rgbd/                # Initial, pre-grasp and final RGB-D evidence
```

## Files

### harvest_pipeline.py

The common pipeline performs the following steps:

1. Load one scene from `trial_configs.py`.
2. Create the Franka, RGB-D camera, target fruit, leaves, stem and neighbor.
3. Test every candidate angle with every RRT seed.
4. Estimate the grasp time of every feasible pre-grasp path.
5. Select and execute the fastest path.
6. Approach and close the gripper.
7. Extract the strawberry to the pre-grasp point.
8. Plan to the drop zone while checking the carried fruit.
9. Place the fruit on the ground.
10. Save the path comparison and trial outcome.

The estimated grasp time is

```text
RRT action count × RRT hold frames × physics timestep
+ straight-approach waypoint time
+ gripper-closing time
```

### trial_configs.py

Only scene parameters change between experiments:

- target strawberry position;
- leaf positions and sizes;
- neighboring-fruit position;
- descriptive difficulty and experimental purpose.

The robot home pose, RRT seeds, candidate angles, controller timing, camera
pose, drop location and collision margin remain constant for a fair comparison.

### run_trials_sequential.py

This launcher uses blocking `subprocess.run()`. Trial 02 cannot start until the
entire Trial 01 Isaac Sim process has closed. It also inserts a configurable
cooldown between processes. It does not use multiprocessing, threads or a
parallel process pool.

## Choosing the Isaac Sim Python command

Use the same Isaac Sim Python command that already runs the standalone
harvesting script successfully on your server. In the examples below,
`<ISAAC_PYTHON>` means that exact command or command prefix.

Examples of commands used by different Isaac Sim installations include an
Isaac-provided Python launcher or an Isaac Sim launcher with a Python-script
option. Do not use a normal system Python unless it can import `isaacsim`.

## Run one trial with the UI

```bash
cd strawberry_harvest_experiments
<ISAAC_PYTHON> harvest_pipeline.py --trial-id 2
```

The final pose remains displayed until the Isaac Sim window is closed.

This mode is appropriate for debugging and recording the demonstration video.

## Run one trial headlessly and exit automatically

```bash
cd strawberry_harvest_experiments
<ISAAC_PYTHON> harvest_pipeline.py \
  --trial-id 2 \
  --headless \
  --auto-close
```

Motion rendering is disabled in headless mode to reduce load. RGB-D capture
still performs short offscreen rendering periods so the required images and
depth arrays are produced.

## Run all ten trials sequentially

```bash
cd strawberry_harvest_experiments
<ISAAC_PYTHON> run_trials_sequential.py
```

Default batch behavior:

- trials run in the order 1 through 10;
- only one Isaac Sim process exists at a time;
- every trial receives `--headless --auto-close`;
- the launcher waits three seconds after each process exits;
- the batch stops if an Isaac Sim process crashes.

Run selected trials only:

```bash
<ISAAC_PYTHON> run_trials_sequential.py --trials 1 2 7 10
```

Increase the cooldown:

```bash
<ISAAC_PYTHON> run_trials_sequential.py --cooldown-seconds 10
```

Continue after a process crash:

```bash
<ISAAC_PYTHON> run_trials_sequential.py --continue-on-error
```

Show the UI while still running sequentially:

```bash
<ISAAC_PYTHON> run_trials_sequential.py --show-ui
```

For the full ten-trial evaluation, headless mode is recommended. Use UI mode
only for selected successful and failure trials needed in the video.

## Results

Each trial creates `results/trial_XX/candidate_paths.csv`. It contains one row
for every feasible candidate-angle and seed combination:

```text
trial_id,angle_deg,seed,actions,joint_length,estimated_grasp_time_s,planning_wall_time_s,selected
```

`results/trial_XX/summary.json` stores both the complete scene configuration
and the outcome. `results/trial_summary.csv` contains one row per trial with:

- feasible-path count;
- selected angle and seed;
- selected path length and estimated grasp time;
- planning, approach, gripper and drop status;
- detected collision status;
- complete harvest success;
- failure stage and reason;
- actual grasp and total wall-clock time.

Rerunning a trial replaces its row in `trial_summary.csv` rather than adding a
duplicate. Its per-trial CSV, JSON and RGB-D evidence are also overwritten.

## Editing experiment parameters

Modify only the corresponding entry in `trial_configs.py`. For example:

```python
2: {
    "description": "target left and slightly lower",
    "difficulty": "moderate",
    "purpose": "Preserve the validated development scene.",
    "fruit_position": [0.520, 0.030, 0.480],
    "leaf_0_position": [0.485, 0.025, 0.525],
    "leaf_1_position": [0.595, -0.020, 0.530],
    "neighbor_position": [0.555, -0.105, 0.490],
},
```

Do not change planner or controller parameters for only one trial unless the
experiment is explicitly intended to study that planner parameter.

## CUDA message

The message below means the installed Lula build is using its CPU search tree:

```text
[Lula] CUDA not enabled in build. Creating 'BasicTree' instead of 'CudaTree'
```

The launcher cannot enable CUDA in a Lula binary that was built without it.
Sequential processes, headless motion and cooldown periods are therefore used
to reduce load and avoid the severe slowdown caused by running multiple Isaac
Sim instances simultaneously.

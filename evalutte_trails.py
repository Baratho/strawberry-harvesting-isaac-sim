from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, FixedSphere, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    PathPlannerVisualizer,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.sensors.camera import Camera


# =============================================================================
# Experiment configuration
# =============================================================================

ROBOT_BASE_POSITION = np.array([-0.15, 0.0, 0.0], dtype=float)
HOME_JOINTS = np.array(
    [0.00, -1.00, 0.00, -2.20, 0.00, 2.40, 0.80, 0.04, 0.04],
    dtype=float,
)

GRIPPER_OPEN = np.array([0.04, 0.04], dtype=float)
GRIPPER_CLOSED = np.array([0.025, 0.025], dtype=float)

FRUIT_RADIUS = 0.030
NEAR_GRASP_DISTANCE = 0.080
GRASP_POSITION_TOLERANCE = 0.025
GRASP_Z_OFFSET = 0.010

# The arm controller cannot physically reach a new IK waypoint in only one or
# two simulation frames.  The old evaluator measured the pose immediately
# after the last command, so controller lag was incorrectly counted as an
# approach failure.  These limits let the arm settle while keeping a hard
# timeout for every trial.
PREGRASP_POSITION_TOLERANCE = 0.030
PREGRASP_SETTLE_MAX_FRAMES = 90
APPROACH_SETTLE_MAX_FRAMES = 180
SETTLE_REQUIRED_FRAMES = 5

CANDIDATE_ANGLES = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]
RRT_SEEDS = [123456, 2026]
RRT_MAX_ITERATIONS = 8000

# Evaluation execution is deliberately faster than the demonstration script.
RRT_HOLD_FRAMES = 1
IK_HOLD_FRAMES = 4
APPROACH_WAYPOINTS = 14
HOME_HOLD_FRAMES = 60
GRIPPER_CLOSE_FRAMES = 30

# Collision geometry used for the straight final approach.
LEAF_SCALE = np.array([0.17, 0.050, 0.025], dtype=float)
STEM_POSITION = np.array([0.55, 0.0, 0.29], dtype=float)
STEM_SCALE = np.array([0.025, 0.025, 0.36], dtype=float)
NEIGHBOR_RADIUS = 0.030
LEAF_CLEARANCE = 0.006
STEM_CLEARANCE = 0.005
NEIGHBOR_CLEARANCE = 0.012

# The camera is on the opposite side requested after the single-trial test.
CAMERA_POSITION = np.array([1.20, 1.10, 0.95], dtype=float)
CAMERA_LOOK_AT = np.array([0.38, 0.00, 0.40], dtype=float)
CAMERA_RESOLUTION = (640, 480)
CAMERA_WARMUP_FRAMES = 12

OUTPUT_DIR = Path.cwd() / "evaluation_output"
TRIAL_CSV = OUTPUT_DIR / "trial_results.csv"
CANDIDATE_CSV = OUTPUT_DIR / "candidate_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.txt"

GREEN = np.array([0.10, 0.75, 0.25])
UNRIPE_GREEN = np.array([0.25, 0.75, 0.15])
RED = np.array([0.90, 0.08, 0.08])
SOIL = np.array([0.48, 0.30, 0.12])


# Each leaf/neighbor position is expressed relative to the target fruit.
# Trial 10 is intentionally outside the Franka workspace and supplies the
# required representative failure case.
TRIALS = [
    {
        "id": 1,
        "description": "baseline moderate occlusion",
        "occlusion": "moderate",
        "target": [0.55, 0.00, 0.50],
        "leaf0_offset": [-0.065, 0.025, 0.025],
        "leaf1_offset": [0.045, -0.020, 0.030],
        "neighbor_offset": [0.005, -0.105, -0.010],
    },
    {
        "id": 2,
        "description": "target left and slightly lower",
        "occlusion": "light",
        "target": [0.52, 0.03, 0.48],
        "leaf0_offset": [-0.080, 0.035, 0.040],
        "leaf1_offset": [0.060, -0.030, 0.045],
        "neighbor_offset": [0.020, -0.120, 0.000],
    },
    {
        "id": 3,
        "description": "target right with front leaf",
        "occlusion": "moderate",
        "target": [0.58, -0.04, 0.52],
        "leaf0_offset": [-0.045, 0.005, 0.020],
        "leaf1_offset": [0.070, -0.025, 0.035],
        "neighbor_offset": [0.000, -0.095, -0.015],
    },
    {
        "id": 4,
        "description": "negative-y target with clear left approach",
        "occlusion": "light",
        "target": [0.54, -0.07, 0.50],
        "leaf0_offset": [-0.080, 0.060, 0.045],
        "leaf1_offset": [0.070, 0.020, 0.050],
        "neighbor_offset": [0.015, -0.125, -0.010],
    },
    {
        "id": 5,
        "description": "positive-y target with neighboring fruit",
        "occlusion": "moderate",
        "target": [0.56, 0.08, 0.51],
        "leaf0_offset": [-0.060, 0.010, 0.025],
        "leaf1_offset": [0.055, -0.055, 0.030],
        "neighbor_offset": [-0.015, -0.080, -0.005],
    },
    {
        "id": 6,
        "description": "high target",
        "occlusion": "moderate",
        "target": [0.55, 0.00, 0.56],
        "leaf0_offset": [-0.055, 0.030, 0.015],
        "leaf1_offset": [0.060, -0.030, 0.020],
        "neighbor_offset": [0.010, -0.105, -0.030],
    },
    {
        "id": 7,
        "description": "low target near plant",
        "occlusion": "heavy",
        "target": [0.53, 0.02, 0.45],
        "leaf0_offset": [-0.040, 0.015, 0.020],
        "leaf1_offset": [0.035, -0.020, 0.025],
        "neighbor_offset": [0.005, -0.080, 0.000],
    },
    {
        "id": 8,
        "description": "crowded neighbor on direct approach",
        "occlusion": "heavy",
        "target": [0.59, 0.02, 0.50],
        "leaf0_offset": [-0.055, 0.025, 0.025],
        "leaf1_offset": [0.045, -0.015, 0.030],
        "neighbor_offset": [-0.060, -0.010, 0.000],
    },
    {
        "id": 9,
        "description": "two leaves close to fruit",
        "occlusion": "heavy",
        "target": [0.57, -0.02, 0.49],
        "leaf0_offset": [-0.035, 0.010, 0.015],
        "leaf1_offset": [0.030, -0.010, 0.018],
        "neighbor_offset": [0.000, -0.075, -0.005],
    },
    {
        "id": 10,
        "description": "intentionally unreachable target",
        "occlusion": "failure_case",
        "target": [1.05, 0.00, 0.58],
        "leaf0_offset": [-0.060, 0.025, 0.025],
        "leaf1_offset": [0.050, -0.020, 0.030],
        "neighbor_offset": [0.000, -0.100, -0.010],
    },
]


TRIAL_FIELDS = [
    "trial_id",
    "description",
    "occlusion",
    "target_x",
    "target_y",
    "target_z",
    "candidate_count",
    "geometric_clear_count",
    "ik_reachable_count",
    "blocked_candidate_count",
    "rrt_attempt_count",
    "selected_angle_deg",
    "planning_success",
    "approach_success",
    "harvest_success",
    "collision_count",
    "planning_time_s",
    "execution_time_s",
    "pregrasp_position_error_m",
    "final_position_error_m",
    "failure_reason",
]

CANDIDATE_FIELDS = [
    "trial_id",
    "candidate_index",
    "angle_deg",
    "near_x",
    "near_y",
    "near_z",
    "geometric_clear",
    "blocker",
    "ik_success",
    "rrt_attempted",
    "rrt_success",
    "status",
]


# =============================================================================
# Math and geometry
# =============================================================================

def normalize(vector):
    length = np.linalg.norm(vector)
    if length < 1.0e-8:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def rotation_matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to scalar-first [w, x, y, z]."""
    m = rotation
    trace = np.trace(m)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quaternion = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ]
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quaternion = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ]
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quaternion = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ]
        )

    return normalize(quaternion.astype(float))


def make_grasp_orientation(approach_direction):
    z_axis = normalize(approach_direction)
    world_up = np.array([0.0, 0.0, 1.0])
    y_axis = normalize(np.cross(z_axis, world_up))
    x_axis = normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix_to_quaternion(rotation)


def make_camera_orientation(camera_position, target_position):
    forward = normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0])
    camera_up = normalize(world_up - np.dot(world_up, forward) * forward)
    camera_left = normalize(np.cross(camera_up, forward))
    rotation = np.column_stack((forward, camera_left, camera_up))
    return rotation_matrix_to_quaternion(rotation)


def make_candidate(target_position, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
    # Aim slightly above the geometric center of the strawberry.  This keeps
    # the fingers away from the lower tip while still enclosing the fruit.
    grasp_position = target_position + np.array(
        [0.0, 0.0, GRASP_Z_OFFSET], dtype=float
    )
    near_position = grasp_position - NEAR_GRASP_DISTANCE * direction
    return {
        "angle": angle_degrees,
        "direction": direction,
        "grasp_position": grasp_position,
        "near_position": near_position,
        "orientation": make_grasp_orientation(direction),
    }


def segment_intersects_aabb(start, end, center, scale, padding=0.0):
    minimum = center - 0.5 * scale - padding
    maximum = center + 0.5 * scale + padding
    direction = end - start

    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-10:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
        else:
            inverse = 1.0 / direction[axis]
            t0 = (minimum[axis] - start[axis]) * inverse
            t1 = (maximum[axis] - start[axis]) * inverse
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                return False
    return True


def segment_intersects_sphere(start, end, center, radius):
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1.0e-12:
        return np.linalg.norm(start - center) <= radius
    alpha = float(np.dot(center - start, direction) / denominator)
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = start + alpha * direction
    return np.linalg.norm(closest - center) <= radius


def approach_blocker(candidate, target_position, scene_positions):
    # Stop at the near surface of the target instead of testing through its
    # center. This avoids the false-positive leaf collision from the first
    # candidate-filter implementation.
    approach_end = (
        candidate["grasp_position"]
        - FRUIT_RADIUS * candidate["direction"]
    )
    start = candidate["near_position"]

    if segment_intersects_aabb(
        start,
        approach_end,
        scene_positions["leaf0"],
        LEAF_SCALE,
        LEAF_CLEARANCE,
    ):
        return "leaf_0"

    if segment_intersects_aabb(
        start,
        approach_end,
        scene_positions["leaf1"],
        LEAF_SCALE,
        LEAF_CLEARANCE,
    ):
        return "leaf_1"

    if segment_intersects_aabb(
        start,
        approach_end,
        STEM_POSITION,
        STEM_SCALE,
        STEM_CLEARANCE,
    ):
        return "stem"

    if segment_intersects_sphere(
        start,
        approach_end,
        scene_positions["neighbor"],
        NEIGHBOR_RADIUS + NEIGHBOR_CLEARANCE,
    ):
        return "neighbor_fruit"

    return ""


def plan_length(plan):
    if plan is None or len(plan) == 0:
        return np.inf
    positions = [np.asarray(action.joint_positions, dtype=float) for action in plan]
    return float(
        sum(
            np.linalg.norm(positions[index] - positions[index - 1])
            for index in range(1, len(positions))
        )
    )


# =============================================================================
# Scene
# =============================================================================

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 60.0,
)
world.scene.add_default_ground_plane()

sun = UsdLux.DistantLight.Define(world.stage, "/World/Sun")
sun.CreateIntensityAttr(2500.0)
sun.CreateAngleAttr(0.5)

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=ROBOT_BASE_POSITION,
    )
)

plant_base = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Base",
        name="plant_base",
        position=np.array([0.55, 0.0, 0.06]),
        size=1.0,
        scale=np.array([0.28, 0.28, 0.12]),
        color=SOIL,
    )
)

stem = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Stem",
        name="stem",
        position=STEM_POSITION,
        size=1.0,
        scale=STEM_SCALE,
        color=GREEN,
    )
)

target_strawberry = world.scene.add(
    VisualSphere(
        prim_path="/World/Plant/TargetStrawberry",
        name="target_strawberry",
        position=np.asarray(TRIALS[0]["target"], dtype=float),
        radius=FRUIT_RADIUS,
        color=RED,
    )
)

leaf_0 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_0",
        name="leaf_0",
        position=np.array([0.485, 0.025, 0.525]),
        size=1.0,
        scale=LEAF_SCALE,
        color=GREEN,
    )
)

leaf_1 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_1",
        name="leaf_1",
        position=np.array([0.595, -0.020, 0.530]),
        size=1.0,
        scale=LEAF_SCALE,
        color=GREEN,
    )
)

neighbor_fruit = world.scene.add(
    FixedSphere(
        prim_path="/World/Plant/NeighborFruit_0",
        name="neighbor_fruit_0",
        position=np.array([0.555, -0.105, 0.490]),
        radius=NEIGHBOR_RADIUS,
        color=UNRIPE_GREEN,
    )
)

rgbd_camera = Camera(
    prim_path="/World/RGBDCamera",
    name="rgbd_camera",
    frequency=20,
    resolution=CAMERA_RESOLUTION,
)

world.reset()

rgbd_camera.initialize()
rgbd_camera.set_world_pose(
    position=CAMERA_POSITION,
    orientation=make_camera_orientation(CAMERA_POSITION, CAMERA_LOOK_AT),
    camera_axes="world",
)
rgbd_camera.set_clipping_range(near_distance=0.05, far_distance=10.0)
rgbd_camera.add_distance_to_image_plane_to_frame()
rgbd_camera.pause()


# =============================================================================
# Solvers and execution helpers
# =============================================================================

kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
    "Franka"
)
lula_ik = LulaKinematicsSolver(**kinematics_config)
articulation_ik = ArticulationKinematicsSolver(
    franka,
    lula_ik,
    "right_gripper",
)

rrt_config = interface_config_loader.load_supported_path_planner_config(
    "Franka",
    "RRT",
)
rrt = RRT(**rrt_config)
rrt.set_max_iterations(RRT_MAX_ITERATIONS)
path_visualizer = PathPlannerVisualizer(franka, rrt)


def apply_gripper(target):
    franka.gripper.apply_action(
        ArticulationAction(joint_positions=np.asarray(target, dtype=float))
    )


def hold_action(action, gripper_target, frames, carry=False):
    for _ in range(frames):
        if not simulation_app.is_running():
            return
        franka.apply_action(action)
        apply_gripper(gripper_target)
        world.step(render=True)
        if carry:
            update_carried_fruit()


def update_carried_fruit():
    end_effector_position, _ = articulation_ik.compute_end_effector_pose()
    # The commanded grasp point is GRASP_Z_OFFSET above the fruit center.
    # Preserve that offset while the fruit is carried.
    fruit_center = end_effector_position - np.array(
        [0.0, 0.0, GRASP_Z_OFFSET], dtype=float
    )
    target_strawberry.set_world_pose(position=fruit_center)


def reset_robot_home():
    franka.set_joint_positions(HOME_JOINTS)
    franka.set_joint_velocities(np.zeros(9))
    action = ArticulationAction(joint_positions=HOME_JOINTS)
    for _ in range(HOME_HOLD_FRAMES):
        hold_action(action, GRIPPER_OPEN, 1, carry=False)
    return action


def execute_plan(plan, gripper_target, carry=False, reverse=False):
    actions = list(reversed(plan)) if reverse else plan
    for action in actions:
        hold_action(
            action,
            gripper_target,
            RRT_HOLD_FRAMES,
            carry=carry,
        )


def settle_end_effector(
    action,
    target_position,
    gripper_target,
    tolerance,
    max_frames,
    carry=False,
):
    """Hold one action until the EE is stably inside the position tolerance."""
    consecutive_frames = 0
    final_error = np.inf

    for _ in range(max_frames):
        if not simulation_app.is_running():
            break

        hold_action(action, gripper_target, 1, carry=carry)
        end_effector_position, _ = articulation_ik.compute_end_effector_pose()
        final_error = float(
            np.linalg.norm(end_effector_position - target_position)
        )

        if final_error <= tolerance:
            consecutive_frames += 1
            if consecutive_frames >= SETTLE_REQUIRED_FRAMES:
                return True, final_error
        else:
            consecutive_frames = 0

    return False, final_error


def execute_ik_line(start, end, orientation, gripper_target, carry=False):
    actions = []
    positions = np.linspace(start, end, APPROACH_WAYPOINTS)
    for position in positions[1:]:
        action, success = articulation_ik.compute_inverse_kinematics(
            target_position=position,
            target_orientation=orientation,
        )
        if not success:
            return None
        actions.append(action)
        hold_action(
            action,
            gripper_target,
            IK_HOLD_FRAMES,
            carry=carry,
        )
    return actions


def compute_rrt_plan(target_position, target_orientation, trial_id):
    rrt.set_end_effector_target(target_position, target_orientation)
    for seed_offset, seed in enumerate(RRT_SEEDS):
        rrt.set_random_seed(seed + 100 * trial_id + seed_offset)
        rrt.update_world()
        plan = path_visualizer.compute_plan_as_articulation_actions(
            max_cspace_dist=0.015
        )
        if plan is not None and len(plan) > 0:
            # For evaluation, the first valid plan is enough.  The previous
            # version always ran every seed even after success, producing
            # avoidable BasicTree warnings and extra CPU planning time.
            return plan

    return None


def set_trial_scene(trial):
    target = np.asarray(trial["target"], dtype=float)
    positions = {
        "leaf0": target + np.asarray(trial["leaf0_offset"], dtype=float),
        "leaf1": target + np.asarray(trial["leaf1_offset"], dtype=float),
        "neighbor": target + np.asarray(trial["neighbor_offset"], dtype=float),
    }

    target_strawberry.set_world_pose(position=target)
    leaf_0.set_world_pose(position=positions["leaf0"])
    leaf_1.set_world_pose(position=positions["leaf1"])
    neighbor_fruit.set_world_pose(position=positions["neighbor"])

    # Allow USD/Fabric and Lula to observe the new transforms.
    for _ in range(10):
        world.step(render=True)
    rrt.update_world()
    return target, positions


# =============================================================================
# RGB-D capture
# =============================================================================

def save_rgbd_snapshot(trial_id, label):
    trial_directory = OUTPUT_DIR / f"trial_{trial_id:02d}"
    trial_directory.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(rgbd_camera.get_rgba())
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.size > 0 and np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    frame = rgbd_camera.get_current_frame()
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        depth = rgbd_camera.get_depth()
    depth = np.asarray(depth, dtype=np.float32).squeeze()

    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("camera returned no valid depth pixels")

    near = float(np.percentile(depth[valid], 1.0))
    far = float(np.percentile(depth[valid], 99.0))
    if far <= near:
        far = near + 1.0e-6
    clipped = np.clip(depth, near, far)
    preview = 255.0 * (1.0 - (clipped - near) / (far - near))
    preview[~valid] = 0.0

    Image.fromarray(rgb).save(trial_directory / f"{label}_rgb.png")
    Image.fromarray(preview.astype(np.uint8)).save(
        trial_directory / f"{label}_depth_visual.png"
    )
    np.save(trial_directory / f"{label}_depth_meters.npy", depth)


def capture_rgbd(trial_id, label):
    try:
        rgbd_camera.resume()
        for _ in range(CAMERA_WARMUP_FRAMES):
            world.step(render=True)
        save_rgbd_snapshot(trial_id, label)
    except Exception as error:
        print(f"RGB-D warning trial={trial_id}, label={label}: {error}")
    finally:
        rgbd_camera.pause()


# =============================================================================
# CSV and candidate evaluation
# =============================================================================

def write_csv(path, fieldnames, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidates(trial_id, target_position, scene_positions):
    evaluations = []
    for index, angle in enumerate(CANDIDATE_ANGLES):
        candidate = make_candidate(target_position, angle)
        blocker = approach_blocker(candidate, target_position, scene_positions)
        geometric_clear = blocker == ""

        ik_success = False
        if geometric_clear:
            _, ik_success = articulation_ik.compute_inverse_kinematics(
                target_position=candidate["near_position"],
                target_orientation=candidate["orientation"],
            )

        if not geometric_clear:
            status = "BLOCKED"
        elif not ik_success:
            status = "IK_FAILED"
        else:
            status = "READY"

        evaluations.append(
            {
                "candidate": candidate,
                "csv": {
                    "trial_id": trial_id,
                    "candidate_index": index,
                    "angle_deg": angle,
                    "near_x": round(float(candidate["near_position"][0]), 4),
                    "near_y": round(float(candidate["near_position"][1]), 4),
                    "near_z": round(float(candidate["near_position"][2]), 4),
                    "geometric_clear": int(geometric_clear),
                    "blocker": blocker,
                    "ik_success": int(ik_success),
                    "rrt_attempted": 0,
                    "rrt_success": 0,
                    "status": status,
                },
            }
        )

    return evaluations


def select_rrt_candidate(trial_id, evaluations):
    attempts = 0
    for evaluation in evaluations:
        row = evaluation["csv"]
        if row["status"] != "READY":
            continue

        attempts += 1
        row["rrt_attempted"] = 1
        candidate = evaluation["candidate"]
        plan = compute_rrt_plan(
            candidate["near_position"],
            candidate["orientation"],
            trial_id,
        )
        if plan is not None:
            row["rrt_success"] = 1
            row["status"] = "SELECTED"
            return candidate, plan, attempts
        row["status"] = "RRT_FAILED"

    return None, None, attempts


# =============================================================================
# Register obstacles and run trials
# =============================================================================

home_action = reset_robot_home()
base_position, base_orientation = franka.get_world_pose()
lula_ik.set_robot_base_pose(base_position, base_orientation)
rrt.set_robot_base_pose(base_position, base_orientation)

# Base and stem never move. Leaves and neighboring fruit move between trials,
# so they must not be registered as static obstacles.
rrt.add_obstacle(plant_base, static=True)
rrt.add_obstacle(stem, static=True)
rrt.add_obstacle(leaf_0, static=False)
rrt.add_obstacle(leaf_1, static=False)
rrt.add_obstacle(neighbor_fruit, static=False)
rrt.update_world()

trial_rows = []
candidate_rows = []

print("=" * 88)
print("STARTING 10-TRIAL STRAWBERRY HARVEST EVALUATION")
print("=" * 88)

for trial in TRIALS:
    if not simulation_app.is_running():
        break

    trial_id = trial["id"]
    print("\n" + "=" * 88)
    print(f"TRIAL {trial_id:02d}/10: {trial['description']}")
    print("=" * 88)

    reset_robot_home()
    target_position, scene_positions = set_trial_scene(trial)
    capture_rgbd(trial_id, "initial")

    planning_start = time.perf_counter()
    evaluations = evaluate_candidates(
        trial_id,
        target_position,
        scene_positions,
    )
    selected_candidate, selected_plan, rrt_attempt_count = select_rrt_candidate(
        trial_id,
        evaluations,
    )
    planning_time = time.perf_counter() - planning_start

    geometric_clear_count = sum(
        row["csv"]["geometric_clear"] for row in evaluations
    )
    ik_reachable_count = sum(row["csv"]["ik_success"] for row in evaluations)
    blocked_count = sum(
        row["csv"]["status"] == "BLOCKED" for row in evaluations
    )

    planning_success = selected_plan is not None
    approach_success = False
    harvest_success = False
    collision_count = 0
    execution_time = 0.0
    pregrasp_position_error = np.nan
    final_position_error = np.nan
    failure_reason = "no_collision_free_candidate"
    selected_angle = ""

    if planning_success:
        selected_angle = selected_candidate["angle"]
        execution_start = time.perf_counter()

        execute_plan(selected_plan, GRIPPER_OPEN, carry=False)
        _, pregrasp_position_error = settle_end_effector(
            selected_plan[-1],
            selected_candidate["near_position"],
            GRIPPER_OPEN,
            PREGRASP_POSITION_TOLERANCE,
            PREGRASP_SETTLE_MAX_FRAMES,
            carry=False,
        )

        grasp_position = selected_candidate["grasp_position"]
        approach_actions = execute_ik_line(
            selected_candidate["near_position"],
            grasp_position,
            selected_candidate["orientation"],
            GRIPPER_OPEN,
            carry=False,
        )

        if approach_actions is None:
            failure_reason = "final_approach_ik_failed"
            execute_plan(
                selected_plan,
                GRIPPER_OPEN,
                carry=False,
                reverse=True,
            )
        else:
            approach_success, final_position_error = settle_end_effector(
                approach_actions[-1],
                grasp_position,
                GRIPPER_OPEN,
                GRASP_POSITION_TOLERANCE,
                APPROACH_SETTLE_MAX_FRAMES,
                carry=False,
            )

            print(
                f"Pose check trial={trial_id:02d} | "
                f"pregrasp_error={pregrasp_position_error:.4f} m | "
                f"grasp_error={final_position_error:.4f} m | "
                f"settled={approach_success}"
            )

            if not approach_success:
                failure_reason = (
                    f"grasp_position_error_{final_position_error:.3f}_m"
                )
                for action in reversed(approach_actions):
                    hold_action(action, GRIPPER_OPEN, IK_HOLD_FRAMES)
                execute_plan(
                    selected_plan,
                    GRIPPER_OPEN,
                    carry=False,
                    reverse=True,
                )
            else:
                grasp_action = approach_actions[-1]
                for step in range(GRIPPER_CLOSE_FRAMES):
                    alpha = float(step + 1) / GRIPPER_CLOSE_FRAMES
                    finger_target = (
                        (1.0 - alpha) * GRIPPER_OPEN
                        + alpha * GRIPPER_CLOSED
                    )
                    hold_action(grasp_action, finger_target, 1)

                update_carried_fruit()
                hold_action(
                    grasp_action,
                    GRIPPER_CLOSED,
                    8,
                    carry=True,
                )
                capture_rgbd(trial_id, "grasp_closed")

                for action in reversed(approach_actions):
                    hold_action(
                        action,
                        GRIPPER_CLOSED,
                        IK_HOLD_FRAMES,
                        carry=True,
                    )
                execute_plan(
                    selected_plan,
                    GRIPPER_CLOSED,
                    carry=True,
                    reverse=True,
                )
                harvest_success = True
                failure_reason = "success"

        execution_time = time.perf_counter() - execution_start

    capture_rgbd(
        trial_id,
        "final_success" if harvest_success else "final_failure",
    )

    for evaluation in evaluations:
        candidate_rows.append(evaluation["csv"])

    trial_row = {
        "trial_id": trial_id,
        "description": trial["description"],
        "occlusion": trial["occlusion"],
        "target_x": target_position[0],
        "target_y": target_position[1],
        "target_z": target_position[2],
        "candidate_count": len(CANDIDATE_ANGLES),
        "geometric_clear_count": geometric_clear_count,
        "ik_reachable_count": ik_reachable_count,
        "blocked_candidate_count": blocked_count,
        "rrt_attempt_count": rrt_attempt_count,
        "selected_angle_deg": selected_angle,
        "planning_success": int(planning_success),
        "approach_success": int(approach_success),
        "harvest_success": int(harvest_success),
        "collision_count": collision_count,
        "planning_time_s": round(planning_time, 4),
        "execution_time_s": round(execution_time, 4),
        "pregrasp_position_error_m": (
            round(float(pregrasp_position_error), 5)
            if np.isfinite(pregrasp_position_error)
            else ""
        ),
        "final_position_error_m": (
            round(float(final_position_error), 5)
            if np.isfinite(final_position_error)
            else ""
        ),
        "failure_reason": failure_reason,
    }
    trial_rows.append(trial_row)

    # Write incrementally so completed trials survive an interrupted run.
    write_csv(TRIAL_CSV, TRIAL_FIELDS, trial_rows)
    write_csv(CANDIDATE_CSV, CANDIDATE_FIELDS, candidate_rows)

    print(
        f"RESULT trial={trial_id:02d} | planning={planning_success} | "
        f"approach={approach_success} | harvest={harvest_success} | "
        f"angle={selected_angle} | reason={failure_reason}"
    )


# =============================================================================
# Summary
# =============================================================================

completed_trials = len(trial_rows)
successful_approaches = sum(row["approach_success"] for row in trial_rows)
planning_failures = sum(1 - row["planning_success"] for row in trial_rows)
collisions = sum(row["collision_count"] for row in trial_rows)
harvest_failures = sum(
    row["planning_success"] and not row["harvest_success"]
    for row in trial_rows
)
harvest_successes = sum(row["harvest_success"] for row in trial_rows)

average_planning_time = (
    sum(row["planning_time_s"] for row in trial_rows) / completed_trials
    if completed_trials
    else 0.0
)
executed_rows = [row for row in trial_rows if row["planning_success"]]
average_execution_time = (
    sum(row["execution_time_s"] for row in executed_rows) / len(executed_rows)
    if executed_rows
    else 0.0
)

summary_lines = [
    "Strawberry Harvesting Evaluation Summary",
    "=" * 48,
    f"Completed trials: {completed_trials}",
    f"Successful approaches: {successful_approaches}",
    f"Planning failures: {planning_failures}",
    f"Collision count: {collisions}",
    f"Harvest successes: {harvest_successes}",
    f"Harvest failures after planning: {harvest_failures}",
    f"Average planning time: {average_planning_time:.4f} s",
    f"Average execution time: {average_execution_time:.4f} s",
    "",
    "Collision metric limitation:",
    "The reported collision count is planner-level. The RRT path and final",
    "straight approach are screened before execution, but no physical contact",
    "sensor is used in this version.",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n" + "=" * 88)
for line in summary_lines:
    print(line)
print(f"Trial CSV:     {TRIAL_CSV}")
print(f"Candidate CSV: {CANDIDATE_CSV}")
print(f"Summary:       {SUMMARY_FILE}")
print("=" * 88)

# Keep the final state visible briefly, then exit automatically.
for _ in range(180):
    if not simulation_app.is_running():
        break
    world.step(render=True)

simulation_app.close()


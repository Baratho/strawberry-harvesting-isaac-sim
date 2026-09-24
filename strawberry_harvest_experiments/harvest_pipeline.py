import argparse
import csv
import json
from pathlib import Path

from trial_configs import available_trial_ids, get_trial_config


PROJECT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Run one parameterized Isaac Sim strawberry-harvesting trial."
)
parser.add_argument(
    "--trial-id",
    type=int,
    choices=available_trial_ids(),
    required=True,
    help="Trial configuration ID from trial_configs.py.",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Disable the interactive window; RGB-D snapshots still render offscreen.",
)
parser.add_argument(
    "--auto-close",
    action="store_true",
    help="Close Isaac Sim after saving results instead of waiting in the UI.",
)
parser.add_argument(
    "--results-dir",
    default=str(PROJECT_DIR / "results"),
    help="Directory for CSV, JSON, and RGB-D outputs.",
)
args = parser.parse_args()

TRIAL_CONFIG = get_trial_config(args.trial_id)
RESULTS_ROOT = Path(args.results_dir).expanduser().resolve()
TRIAL_RESULTS_DIR = RESULTS_ROOT / f"trial_{args.trial_id:02d}"
TRIAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import time
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, FixedSphere, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    PathPlannerVisualizer,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT


# -----------------------------------------------------------------------------
# Scene and motion parameters
# -----------------------------------------------------------------------------

TRIAL_ID = args.trial_id
TRIAL_DESCRIPTION = TRIAL_CONFIG["description"]
TRIAL_DIFFICULTY = TRIAL_CONFIG["difficulty"]
TRIAL_PURPOSE = TRIAL_CONFIG["purpose"]
FRUIT_POSITION = np.array(TRIAL_CONFIG["fruit_position"], dtype=float)
FRUIT_RADIUS = float(TRIAL_CONFIG["fruit_radius"])
FRUIT_ORIENTATION = np.array(
    TRIAL_CONFIG["fruit_orientation"], dtype=float
)
NEAR_GRASP_DISTANCE = 0.08

# The carried strawberry is checked as a sphere at the gripper frame.  The
# small positive margin keeps it from merely grazing leaves between waypoints.
PAYLOAD_COLLISION_RADIUS = FRUIT_RADIUS
PAYLOAD_CLEARANCE = 0.001

# Move the base slightly backward so a front-facing home pose is collision-free.
ROBOT_BASE_POSITION = np.array([-0.15, 0.0, 0.0], dtype=float)

# Front-facing home configuration: 7 arm joints + 2 finger joints.
# q1 is zero, so the arm no longer begins by turning away from the plant.
HOME_JOINTS = np.array(
    [0.00, -1.00, 0.00, -2.20, 0.00, 2.40, 0.80, 0.04, 0.04],
    dtype=float,
)

GRIPPER_OPEN = np.array([0.04, 0.04], dtype=float)
GRIPPER_CLOSED = np.array([0.025, 0.025], dtype=float)

# Cheap geometric screening covers a full 360-degree ring at three heights.
# Only the best MAX_RRT_DIRECTIONS proceed to the expensive Lula RRT stage, so
# the total number of RRT calls remains 6 directions x 6 seeds = 36.
CANDIDATE_AZIMUTHS = list(range(-180, 180, 30))
CANDIDATE_ELEVATIONS = [-30, 0, 30]
MAX_RRT_DIRECTIONS = 6
RRT_SEEDS = [123456, 2026, 42, 7, 99, 314159]

APPROACH_PROBE_RADIUS = 0.020
GRASP_CONTACT_EXEMPT_DISTANCE = 0.035
GRASP_POSITION_TOLERANCE = 0.018
GRASP_SETTLE_FRAMES = 90

RRT_HOLD_FRAMES = 3
IK_HOLD_FRAMES = 6
APPROACH_WAYPOINTS = 16
LOWER_WAYPOINTS = 12
GRIPPER_CLOSE_FRAMES = 60
GRIPPER_RELEASE_FRAMES = 60
PHYSICS_DT = 1.0 / 60.0
MOTION_RENDER = not args.headless

# Drop zone is away from the plant.
DROP_HOVER_POSITION = np.array([0.25, -0.35, 0.24], dtype=float)
DROP_RELEASE_POSITION = np.array([0.25, -0.35, 0.09], dtype=float)
# The visual mesh extends 4 cm below its center, so z=0.04 rests on the ground.
DROP_GROUND_POSITION = np.array([0.25, -0.35, 0.040], dtype=float)

# Local +Z of the gripper points down.
DROP_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

# Fixed RGB-D observation camera. It records evidence for the experiment but
# does not replace the assignment's allowed ground-truth fruit position.
CAMERA_POSITION = np.array([1.20, -1.10, 0.95], dtype=float)
CAMERA_LOOK_AT = np.array([0.32, 0.00, 0.38], dtype=float)
CAMERA_RESOLUTION = (640, 480)
CAMERA_WARMUP_FRAMES = 20
RGBD_OUTPUT_DIR = (
    TRIAL_RESULTS_DIR / "rgbd"
)

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.10, 0.35, 1.00])
SOIL = np.array([0.48, 0.30, 0.12])


# Collision geometry is deliberately kept simple for Lula, while separate USD
# meshes below provide a more natural strawberry-and-leaf appearance.
PLANT_BASE_POSITION = np.array(
    TRIAL_CONFIG["plant_base_position"], dtype=float
)
PLANT_BASE_SCALE = np.array(TRIAL_CONFIG["plant_base_scale"], dtype=float)
STEM_POSITION = np.array(TRIAL_CONFIG["stem_position"], dtype=float)
STEM_SCALE = np.array(TRIAL_CONFIG["stem_scale"], dtype=float)
LEAF_0_POSITION = np.array(TRIAL_CONFIG["leaf_0_position"], dtype=float)
LEAF_0_SCALE = np.array(TRIAL_CONFIG["leaf_0_scale"], dtype=float)
LEAF_0_ORIENTATION = np.array(
    TRIAL_CONFIG["leaf_0_orientation"], dtype=float
)
LEAF_1_POSITION = np.array(TRIAL_CONFIG["leaf_1_position"], dtype=float)
LEAF_1_SCALE = np.array(TRIAL_CONFIG["leaf_1_scale"], dtype=float)
LEAF_1_ORIENTATION = np.array(
    TRIAL_CONFIG["leaf_1_orientation"], dtype=float
)
LEAF_2_ENABLED = bool(TRIAL_CONFIG["leaf_2_enabled"])
LEAF_2_POSITION = np.array(TRIAL_CONFIG["leaf_2_position"], dtype=float)
LEAF_2_SCALE = np.array(TRIAL_CONFIG["leaf_2_scale"], dtype=float)
LEAF_2_ORIENTATION = np.array(
    TRIAL_CONFIG["leaf_2_orientation"], dtype=float
)
NEIGHBOR_FRUIT_POSITION = np.array(
    TRIAL_CONFIG["neighbor_position"], dtype=float
)
NEIGHBOR_FRUIT_RADIUS = float(TRIAL_CONFIG["neighbor_radius"])


# -----------------------------------------------------------------------------
# Math utilities
# -----------------------------------------------------------------------------

def normalize(vector):
    length = np.linalg.norm(vector)
    if length < 1.0e-8:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def rotation_matrix_to_quaternion(rotation):
    """Convert a 3x3 rotation matrix to [w, x, y, z]."""
    m = rotation
    trace = np.trace(m)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return normalize(np.array([w, x, y, z], dtype=float))


def make_grasp_orientation(approach_direction):
    z_axis = normalize(approach_direction)
    world_up = np.array([0.0, 0.0, 1.0])
    y_axis = normalize(np.cross(z_axis, world_up))
    x_axis = normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation_matrix_to_quaternion(rotation)


def make_camera_orientation(camera_position, target_position):
    """Camera pose in Isaac Sim world-camera axes: +X forward and +Z up."""
    forward = normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0])
    camera_up = normalize(world_up - np.dot(world_up, forward) * forward)
    camera_left = normalize(np.cross(camera_up, forward))
    rotation = np.column_stack((forward, camera_left, camera_up))
    return rotation_matrix_to_quaternion(rotation)


def make_candidate(azimuth_degrees, elevation_degrees):
    """Create one spherical pre-grasp candidate around the target fruit."""
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=float,
    )
    near_position = FRUIT_POSITION - NEAR_GRASP_DISTANCE * direction
    return {
        "angle": azimuth_degrees,
        "azimuth": azimuth_degrees,
        "elevation": elevation_degrees,
        "direction": direction,
        "near_position": near_position,
        "orientation": make_grasp_orientation(direction),
    }


def plan_length(plan):
    """Joint-space length used to reject unnecessarily winding RRT paths."""
    if plan is None or len(plan) == 0:
        return np.inf

    positions = [np.asarray(action.joint_positions, dtype=float) for action in plan]
    return float(
        sum(
            np.linalg.norm(positions[index] - positions[index - 1])
            for index in range(1, len(positions))
        )
    )


def save_rgbd_snapshot(camera, label):
    """Save RGB, metric depth, and a visible depth preview for one phase."""
    RGBD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(camera.get_rgba())
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.size > 0 and np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    frame = camera.get_current_frame()
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        depth = camera.get_depth()

    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("camera returned no valid depth pixels")

    np.save(RGBD_OUTPUT_DIR / f"{label}_depth_meters.npy", depth)

    near = float(np.percentile(depth[valid], 1.0))
    far = float(np.percentile(depth[valid], 99.0))
    if far <= near:
        far = near + 1.0e-6

    clipped = np.clip(depth, near, far)
    depth_visual = 255.0 * (1.0 - (clipped - near) / (far - near))
    depth_visual[~valid] = 0.0
    depth_visual = depth_visual.astype(np.uint8)

    Image.fromarray(rgb).save(RGBD_OUTPUT_DIR / f"{label}_rgb.png")
    Image.fromarray(depth_visual).save(
        RGBD_OUTPUT_DIR / f"{label}_depth_visual.png"
    )

    camera_position, camera_orientation = camera.get_world_pose(
        camera_axes="world"
    )
    np.savez(
        RGBD_OUTPUT_DIR / "camera_metadata.npz",
        position=np.asarray(camera_position),
        orientation_wxyz=np.asarray(camera_orientation),
        resolution=np.asarray(CAMERA_RESOLUTION),
        depth_unit="meter",
    )

    print(
        f"RGB-D saved: {label} | valid_depth={np.count_nonzero(valid)}/"
        f"{depth.size} | range={near:.3f}-{far:.3f} m"
    )


def capture_rgbd(camera, label):
    """Resume briefly for a fresh frame; camera errors never stop harvesting."""
    try:
        camera.resume()
        for _ in range(CAMERA_WARMUP_FRAMES):
            if not simulation_app.is_running():
                return False
            world.step(render=True)
        save_rgbd_snapshot(camera, label)
        return True
    except Exception as error:
        print(f"RGB-D warning at {label}: {error}")
        return False
    finally:
        camera.pause()


# -----------------------------------------------------------------------------
# Lightweight visual models
# -----------------------------------------------------------------------------

def create_colored_mesh(stage, path, points, faces, color, smooth=False):
    """Create a colored USD mesh from local-space points and polygon faces."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [Gf.Vec3f(float(point[0]), float(point[1]), float(point[2])) for point in points]
    )
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr(
        [index for face in faces for index in face]
    )
    mesh.CreateDisplayColorAttr(
        [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    )
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateSubdivisionSchemeAttr().Set(
        UsdGeom.Tokens.catmullClark if smooth else UsdGeom.Tokens.none
    )
    return mesh


def set_xform_orientation(xform, orientation):
    """Apply an Isaac-order [w, x, y, z] quaternion to a USD xform."""
    quaternion = normalize(np.asarray(orientation, dtype=float))
    xform.AddOrientOp().Set(
        Gf.Quatf(
            float(quaternion[0]),
            Gf.Vec3f(
                float(quaternion[1]),
                float(quaternion[2]),
                float(quaternion[3]),
            ),
        )
    )


def create_strawberry_visual(
    stage,
    root_path,
    position,
    body_color,
    orientation=None,
):
    """Create a tapered strawberry body, calyx, stem, and visible seeds."""
    root = UsdGeom.Xform.Define(stage, root_path)
    translate_op = root.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*[float(value) for value in position]))
    if orientation is not None:
        set_xform_orientation(root, orientation)

    segments = 16
    # (height, radius) produces a rounded shoulder and a pointed lower tip.
    profile = [
        (0.027, 0.006),
        (0.022, 0.020),
        (0.010, 0.029),
        (-0.006, 0.030),
        (-0.021, 0.021),
        (-0.035, 0.004),
    ]
    points = []
    for height, radius in profile:
        for segment in range(segments):
            angle = 2.0 * np.pi * segment / segments
            points.append(
                [radius * np.cos(angle), radius * np.sin(angle), height]
            )

    faces = []
    for ring in range(len(profile) - 1):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            lower = ring * segments
            upper = (ring + 1) * segments
            faces.append(
                [
                    lower + segment,
                    lower + next_segment,
                    upper + next_segment,
                    upper + segment,
                ]
            )

    top_index = len(points)
    points.append([0.0, 0.0, 0.030])
    bottom_index = len(points)
    points.append([0.0, 0.0, -0.040])
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        faces.append([top_index, next_segment, segment])
        last_ring = (len(profile) - 1) * segments
        faces.append(
            [bottom_index, last_ring + segment, last_ring + next_segment]
        )

    create_colored_mesh(
        stage,
        f"{root_path}/Body",
        points,
        faces,
        body_color,
        smooth=True,
    )

    # Five pointed green sepals form the leafy cap.
    calyx_points = [[0.0, 0.0, 0.030]]
    calyx_faces = []
    for leaf_index in range(5):
        angle = 2.0 * np.pi * leaf_index / 5.0
        left_angle = angle - 0.34
        right_angle = angle + 0.34
        base = len(calyx_points)
        calyx_points.extend(
            [
                [0.011 * np.cos(left_angle), 0.011 * np.sin(left_angle), 0.029],
                [0.027 * np.cos(angle), 0.027 * np.sin(angle), 0.025],
                [0.011 * np.cos(right_angle), 0.011 * np.sin(right_angle), 0.029],
            ]
        )
        calyx_faces.append([0, base, base + 1, base + 2])

    create_colored_mesh(
        stage,
        f"{root_path}/Calyx",
        calyx_points,
        calyx_faces,
        np.array([0.08, 0.48, 0.12]),
    )

    fruit_stem = UsdGeom.Cylinder.Define(stage, f"{root_path}/Stem")
    fruit_stem.CreateRadiusAttr(0.0025)
    fruit_stem.CreateHeightAttr(0.018)
    fruit_stem.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.42, 0.08)])
    UsdGeom.Xformable(fruit_stem.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.039)
    )

    # Small pale seed marks around the visible body.
    seed_color = Gf.Vec3f(0.96, 0.78, 0.28)
    for seed_index in range(14):
        angle = 2.0 * np.pi * seed_index / 7.0 + (seed_index // 7) * 0.35
        height = 0.008 if seed_index < 7 else -0.014
        radius = 0.030 if seed_index < 7 else 0.025
        seed = UsdGeom.Sphere.Define(stage, f"{root_path}/Seed_{seed_index:02d}")
        seed.CreateRadiusAttr(0.0014)
        seed.CreateDisplayColorAttr([seed_color])
        UsdGeom.Xformable(seed.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(
                float(radius * np.cos(angle)),
                float(radius * np.sin(angle)),
                float(height),
            )
        )

    return translate_op


def create_leaf_visual(
    stage,
    path,
    position,
    length,
    width,
    color,
    orientation=None,
):
    """Create a pointed, slightly curved leaf with a lighter center vein."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))
    if orientation is not None:
        set_xform_orientation(root, orientation)

    half_length = 0.5 * length
    half_width = 0.5 * width
    bend = 0.006
    points = [
        [-half_length, 0.0, 0.0],
        [-0.28 * length, half_width * 0.78, bend],
        [0.0, half_width, 1.5 * bend],
        [0.28 * length, half_width * 0.78, bend],
        [half_length, 0.0, 0.0],
        [0.28 * length, -half_width * 0.78, bend],
        [0.0, -half_width, 1.5 * bend],
        [-0.28 * length, -half_width * 0.78, bend],
        [0.0, 0.0, 1.7 * bend],
    ]
    faces = [[8, index, (index + 1) % 8] for index in range(8)]
    create_colored_mesh(stage, f"{path}/Blade", points, faces, color)

    vein_width = max(0.0015, width * 0.035)
    vein_points = [
        [-0.46 * length, -vein_width, bend + 0.001],
        [-0.46 * length, vein_width, bend + 0.001],
        [0.46 * length, vein_width, bend + 0.001],
        [0.46 * length, -vein_width, bend + 0.001],
    ]
    create_colored_mesh(
        stage,
        f"{path}/Vein",
        vein_points,
        [[0, 1, 2, 3]],
        np.array([0.35, 0.82, 0.25]),
    )
    return root


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

world = World(
    stage_units_in_meters=1.0,
    physics_dt=PHYSICS_DT,
    rendering_dt=PHYSICS_DT,
)
world.scene.add_default_ground_plane()

# Stable lighting for camera images independent of the interactive viewport.
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
        position=PLANT_BASE_POSITION,
        size=1.0,
        scale=PLANT_BASE_SCALE,
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

# The visible target is a tapered strawberry mesh. It remains a logical payload
# after grasping; its spherical collision envelope is checked explicitly below.
target_strawberry_translate_op = create_strawberry_visual(
    world.stage,
    "/World/Plant/TargetStrawberry",
    FRUIT_POSITION,
    RED,
    orientation=FRUIT_ORIENTATION,
)
target_strawberry_position = FRUIT_POSITION.copy()

# These cuboids are Lula collision proxies. Their rendering is hidden after
# creation and replaced by pointed, curved leaf meshes of matching size.
leaf_0 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_0_Collision",
        name="leaf_0",
        position=LEAF_0_POSITION,
        orientation=LEAF_0_ORIENTATION,
        size=1.0,
        scale=LEAF_0_SCALE,
        color=GREEN,
    )
)

leaf_1 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_1_Collision",
        name="leaf_1",
        position=LEAF_1_POSITION,
        orientation=LEAF_1_ORIENTATION,
        size=1.0,
        scale=LEAF_1_SCALE,
        color=GREEN,
    )
)

UsdGeom.Imageable(leaf_0.prim).MakeInvisible()
UsdGeom.Imageable(leaf_1.prim).MakeInvisible()
create_leaf_visual(
    world.stage,
    "/World/Plant/Leaf_0",
    LEAF_0_POSITION,
    LEAF_0_SCALE[0],
    max(0.060, LEAF_0_SCALE[1]),
    np.array([0.08, 0.62, 0.16]),
    orientation=LEAF_0_ORIENTATION,
)
create_leaf_visual(
    world.stage,
    "/World/Plant/Leaf_1",
    LEAF_1_POSITION,
    LEAF_1_SCALE[0],
    max(0.065, LEAF_1_SCALE[1]),
    np.array([0.12, 0.70, 0.20]),
    orientation=LEAF_1_ORIENTATION,
)

leaf_2 = None
if LEAF_2_ENABLED:
    leaf_2 = world.scene.add(
        FixedCuboid(
            prim_path="/World/Plant/Leaf_2_Collision",
            name="leaf_2",
            position=LEAF_2_POSITION,
            orientation=LEAF_2_ORIENTATION,
            size=1.0,
            scale=LEAF_2_SCALE,
            color=GREEN,
        )
    )
    UsdGeom.Imageable(leaf_2.prim).MakeInvisible()
    create_leaf_visual(
        world.stage,
        "/World/Plant/Leaf_2",
        LEAF_2_POSITION,
        LEAF_2_SCALE[0],
        LEAF_2_SCALE[1],
        np.array([0.10, 0.66, 0.18]),
        orientation=LEAF_2_ORIENTATION,
    )

neighbor_fruit = world.scene.add(
    FixedSphere(
        prim_path="/World/Plant/NeighborFruit_0_Collision",
        name="neighbor_fruit_0",
        position=NEIGHBOR_FRUIT_POSITION,
        radius=NEIGHBOR_FRUIT_RADIUS,
        color=np.array([0.25, 0.75, 0.15]),
    )
)
UsdGeom.Imageable(neighbor_fruit.prim).MakeInvisible()
create_strawberry_visual(
    world.stage,
    "/World/Plant/NeighborFruit_0",
    NEIGHBOR_FRUIT_POSITION,
    np.array([0.15, 0.75, 0.20]),
)

world.scene.add(
    VisualSphere(
        prim_path="/World/DropMarker",
        name="drop_marker",
        position=DROP_GROUND_POSITION,
        radius=0.018,
        color=BLUE,
    )
)

rgbd_camera = Camera(
    prim_path="/World/RGBDCamera",
    name="rgbd_camera",
    frequency=20,
    resolution=CAMERA_RESOLUTION,
)

world.reset()

# Camera sensors must be initialized after the world reset.
rgbd_camera.initialize()
rgbd_camera.set_world_pose(
    position=CAMERA_POSITION,
    orientation=make_camera_orientation(CAMERA_POSITION, CAMERA_LOOK_AT),
    camera_axes="world",
)
rgbd_camera.set_clipping_range(near_distance=0.05, far_distance=10.0)
rgbd_camera.add_distance_to_image_plane_to_frame()
rgbd_camera.pause()


# -----------------------------------------------------------------------------
# Solvers and low-level control helpers
# -----------------------------------------------------------------------------

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
rrt.set_max_iterations(15000)
path_visualizer = PathPlannerVisualizer(franka, rrt)


def apply_gripper(target):
    franka.gripper.apply_action(
        ArticulationAction(joint_positions=np.asarray(target, dtype=float))
    )


def set_target_strawberry_position(position):
    global target_strawberry_position
    target_strawberry_position = np.asarray(position, dtype=float).copy()
    target_strawberry_translate_op.Set(
        Gf.Vec3d(*[float(value) for value in target_strawberry_position])
    )


def update_carried_fruit():
    ee_position, _ = articulation_ik.compute_end_effector_pose()
    set_target_strawberry_position(ee_position)


def hold_action(action, gripper_target, frames, carry=False):
    for _ in range(frames):
        if not simulation_app.is_running():
            return
        franka.apply_action(action)
        apply_gripper(gripper_target)
        world.step(render=MOTION_RENDER)
        if carry:
            update_carried_fruit()


def wait_until_grasp_pose(grasp_action):
    """Confirm the real gripper pose before logically attaching the fruit."""
    final_error = np.inf
    for _ in range(GRASP_SETTLE_FRAMES):
        hold_action(grasp_action, GRIPPER_OPEN, 1, carry=False)
        ee_position, _ = articulation_ik.compute_end_effector_pose()
        final_error = np.linalg.norm(
            np.asarray(ee_position, dtype=float) - FRUIT_POSITION
        )
        if final_error <= GRASP_POSITION_TOLERANCE:
            return True, float(final_error)
    return False, float(final_error)


def execute_plan(plan, gripper_target, carry=False, reverse=False, label="RRT"):
    actions = list(reversed(plan)) if reverse else plan
    for index, action in enumerate(actions):
        hold_action(action, gripper_target, RRT_HOLD_FRAMES, carry=carry)
        if index % 25 == 0 or index == len(actions) - 1:
            print(f"{label}: {index + 1}/{len(actions)}")


def execute_ik_line(start, end, orientation, gripper_target, carry=False):
    actions = []
    positions = np.linspace(start, end, APPROACH_WAYPOINTS)

    for index, position in enumerate(positions[1:]):
        action, success = articulation_ik.compute_inverse_kinematics(
            target_position=position,
            target_orientation=orientation,
        )
        if not success:
            print(f"IK failed at line waypoint {index + 1}")
            return None

        actions.append(action)
        hold_action(action, gripper_target, IK_HOLD_FRAMES, carry=carry)

    return actions


def quaternion_wxyz_to_rotation_matrix(quaternion):
    """Convert a normalized [w, x, y, z] quaternion to a 3x3 matrix."""
    w, x, y, z = normalize(np.asarray(quaternion, dtype=float))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
             2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
             2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
             1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def sphere_intersects_box(
    sphere_center,
    sphere_radius,
    box_center,
    box_scale,
    box_orientation=None,
):
    """Test a sphere against an axis-aligned or quaternion-oriented box."""
    half_extents = 0.5 * np.asarray(box_scale, dtype=float)
    relative_center = (
        np.asarray(sphere_center, dtype=float)
        - np.asarray(box_center, dtype=float)
    )
    if box_orientation is not None:
        rotation = quaternion_wxyz_to_rotation_matrix(box_orientation)
        relative_center = rotation.T @ relative_center

    closest = np.clip(relative_center, -half_extents, half_extents)
    distance = np.linalg.norm(relative_center - closest)
    return distance <= sphere_radius


def point_to_box_distance(
    point,
    box_center,
    box_scale,
    box_orientation=None,
):
    """Return Euclidean distance from a point to an oriented box surface."""
    half_extents = 0.5 * np.asarray(box_scale, dtype=float)
    relative_point = (
        np.asarray(point, dtype=float) - np.asarray(box_center, dtype=float)
    )
    if box_orientation is not None:
        rotation = quaternion_wxyz_to_rotation_matrix(box_orientation)
        relative_point = rotation.T @ relative_point
    outside = np.maximum(np.abs(relative_point) - half_extents, 0.0)
    return float(np.linalg.norm(outside))


def candidate_approach_is_clear(candidate):
    """Screen the complete pre-grasp-to-fruit corridor before running RRT."""
    samples = np.linspace(
        candidate["near_position"],
        FRUIT_POSITION,
        APPROACH_WAYPOINTS,
    )
    always_checked_boxes = [
        ("plant_base", PLANT_BASE_POSITION, PLANT_BASE_SCALE, None),
        (
            "leaf_0",
            LEAF_0_POSITION,
            LEAF_0_SCALE,
            LEAF_0_ORIENTATION,
        ),
        (
            "leaf_1",
            LEAF_1_POSITION,
            LEAF_1_SCALE,
            LEAF_1_ORIENTATION,
        ),
    ]
    if LEAF_2_ENABLED:
        always_checked_boxes.append(
            (
                "leaf_2",
                LEAF_2_POSITION,
                LEAF_2_SCALE,
                LEAF_2_ORIENTATION,
            )
        )

    minimum_clearance = np.inf
    for sample_index, point in enumerate(samples):
        for name, center, scale, orientation in always_checked_boxes:
            clearance = point_to_box_distance(
                point,
                center,
                scale,
                orientation,
            ) - APPROACH_PROBE_RADIUS
            minimum_clearance = min(minimum_clearance, clearance)
            if clearance <= 0.0:
                return (
                    False,
                    f"approach sample {sample_index + 1} intersects {name}",
                    clearance,
                )

        distance_to_target = np.linalg.norm(point - FRUIT_POSITION)
        if distance_to_target >= GRASP_CONTACT_EXEMPT_DISTANCE:
            stem_clearance = point_to_box_distance(
                point,
                STEM_POSITION,
                STEM_SCALE,
            ) - APPROACH_PROBE_RADIUS
            minimum_clearance = min(minimum_clearance, stem_clearance)
            if stem_clearance <= 0.0:
                return (
                    False,
                    f"approach sample {sample_index + 1} intersects stem",
                    stem_clearance,
                )

            neighbor_clearance = (
                np.linalg.norm(point - NEIGHBOR_FRUIT_POSITION)
                - NEIGHBOR_FRUIT_RADIUS
                - APPROACH_PROBE_RADIUS
            )
            minimum_clearance = min(minimum_clearance, neighbor_clearance)
            if neighbor_clearance <= 0.0:
                return (
                    False,
                    "approach sample "
                    f"{sample_index + 1} intersects neighbor_fruit_0",
                    neighbor_clearance,
                )

    return True, "clear", float(minimum_clearance)


def carried_strawberry_collision(fruit_center):
    """Return the first obstacle hit by the carried strawberry, or None."""
    effective_radius = PAYLOAD_COLLISION_RADIUS + PAYLOAD_CLEARANCE
    boxes = [
        ("plant_base", PLANT_BASE_POSITION, PLANT_BASE_SCALE, None),
        ("stem", STEM_POSITION, STEM_SCALE, None),
        (
            "leaf_0",
            LEAF_0_POSITION,
            LEAF_0_SCALE,
            LEAF_0_ORIENTATION,
        ),
        (
            "leaf_1",
            LEAF_1_POSITION,
            LEAF_1_SCALE,
            LEAF_1_ORIENTATION,
        ),
    ]
    if LEAF_2_ENABLED:
        boxes.append(
            (
                "leaf_2",
                LEAF_2_POSITION,
                LEAF_2_SCALE,
                LEAF_2_ORIENTATION,
            )
        )

    for name, center, scale, orientation in boxes:
        if sphere_intersects_box(
            fruit_center,
            effective_radius,
            center,
            scale,
            orientation,
        ):
            # The target begins attached to the stem. Permit that local contact
            # only during the first few centimeters of the detachment motion.
            if (
                name == "stem"
                and np.linalg.norm(fruit_center - FRUIT_POSITION) < 0.045
            ):
                continue
            return name

    neighbor_distance = np.linalg.norm(
        fruit_center - NEIGHBOR_FRUIT_POSITION
    )
    neighbor_clearance = effective_radius + NEIGHBOR_FRUIT_RADIUS
    if neighbor_distance <= neighbor_clearance:
        # Trial 6 begins with the two 3 cm-radius fruits exactly tangent.
        # Permit only that initial contact so the payload can move away; any
        # later motion into the neighboring fruit is still rejected.
        initial_center_distance = np.linalg.norm(
            FRUIT_POSITION - NEIGHBOR_FRUIT_POSITION
        )
        initially_tangent = abs(
            initial_center_distance
            - (FRUIT_RADIUS + NEIGHBOR_FRUIT_RADIUS)
        ) < 1.0e-5
        still_at_initial_position = (
            np.linalg.norm(fruit_center - FRUIT_POSITION) < 0.003
        )
        if not (initially_tangent and still_at_initial_position):
            return "neighbor_fruit_0"
    return None


def carried_strawberry_path_is_clear(plan):
    """Validate every RRT waypoint using FK and the fruit collision sphere."""
    arm_joint_count = len(lula_ik.get_joint_names())
    current_position, _ = articulation_ik.compute_end_effector_pose()
    obstacle_name = carried_strawberry_collision(np.asarray(current_position))
    if obstacle_name is not None:
        return False, f"start pose intersects {obstacle_name}"

    for waypoint_index, action in enumerate(plan):
        joint_positions = np.asarray(action.joint_positions, dtype=float)
        arm_positions = joint_positions[:arm_joint_count]
        fruit_position, _ = lula_ik.compute_forward_kinematics(
            "right_gripper",
            arm_positions,
        )
        obstacle_name = carried_strawberry_collision(
            np.asarray(fruit_position, dtype=float)
        )
        if obstacle_name is not None:
            return (
                False,
                f"waypoint {waypoint_index + 1}/{len(plan)} hits {obstacle_name}",
            )
    return True, "clear"


def estimate_grasp_time(plan):
    """Deterministic simulated time from home to a closed grasp."""
    rrt_time = len(plan) * RRT_HOLD_FRAMES * PHYSICS_DT
    approach_time = (APPROACH_WAYPOINTS - 1) * IK_HOLD_FRAMES * PHYSICS_DT
    close_time = GRIPPER_CLOSE_FRAMES * PHYSICS_DT
    return {
        "rrt_time_s": rrt_time,
        "approach_time_s": approach_time,
        "close_time_s": close_time,
        "total_time_s": rrt_time + approach_time + close_time,
    }


def compute_rrt_seed_plans(
    target_position,
    target_orientation,
    check_carried_strawberry=False,
):
    """Return every valid RRT path, including its seed and planning metrics."""
    rrt.set_end_effector_target(target_position, target_orientation)
    valid_paths = []

    for seed in RRT_SEEDS:
        rrt.set_random_seed(seed)
        rrt.update_world()
        planning_start = time.perf_counter()
        plan = path_visualizer.compute_plan_as_articulation_actions(
            max_cspace_dist=0.01
        )
        planning_time = time.perf_counter() - planning_start
        length = plan_length(plan)

        if plan is not None and len(plan) > 0 and check_carried_strawberry:
            payload_clear, reason = carried_strawberry_path_is_clear(plan)
            if not payload_clear:
                print(
                    f"  seed={seed}, path_length={length:.3f}, "
                    f"planning_time={planning_time:.3f}s, "
                    f"rejected: carried strawberry {reason}"
                )
                continue

        if plan is None or len(plan) == 0:
            print(
                f"  seed={seed}, path_length=inf, "
                f"planning_time={planning_time:.3f}s"
            )
            continue

        print(
            f"  seed={seed}, path_length={length:.3f}, "
            f"actions={len(plan)}, planning_time={planning_time:.3f}s"
        )
        valid_paths.append(
            {
                "seed": seed,
                "plan": plan,
                "joint_length": length,
                "planning_time_s": planning_time,
            }
        )

    return valid_paths


def compute_best_rrt_plan(
    target_position,
    target_orientation,
    check_carried_strawberry=False,
):
    """Run several seeds and keep the shortest valid joint-space path."""
    valid_paths = compute_rrt_seed_plans(
        target_position,
        target_orientation,
        check_carried_strawberry=check_carried_strawberry,
    )
    if not valid_paths:
        return None, np.inf

    best_path = min(
        valid_paths,
        key=lambda item: (item["joint_length"], item["planning_time_s"]),
    )
    return best_path["plan"], best_path["joint_length"]


def print_harvest_path_comparison(
    feasible_paths,
    selected_path,
    actual_grasp_wall_time_s,
):
    """Print all feasible pre-grasp routes after the harvesting attempt."""
    print("=" * 124)
    print("AVAILABLE HARVEST PATH TIME COMPARISON")
    print(
        "Time estimate = RRT execution + straight approach + gripper closing "
        "in simulated seconds."
    )

    if not feasible_paths:
        print("No feasible pre-grasp path was found.")
        print("=" * 124)
        return

    ordered_paths = sorted(
        feasible_paths,
        key=lambda item: (
            item["estimated_grasp_time_s"],
            item["joint_length"],
            item["planning_time_s"],
        ),
    )
    print(
        f"{'Rank':>4} {'Azimuth':>8} {'Elev':>6} {'Seed':>8} {'Actions':>8} "
        f"{'JointLen':>10} {'EstGrasp(s)':>12} {'PlanWall(s)':>12} {'Selected':>9}"
    )
    print("-" * 124)
    for rank, path_record in enumerate(ordered_paths, start=1):
        selected_marker = "YES" if path_record is selected_path else ""
        print(
            f"{rank:>4d} {path_record['azimuth']:>8d} "
            f"{path_record['elevation']:>6d} "
            f"{path_record['seed']:>8d} {len(path_record['plan']):>8d} "
            f"{path_record['joint_length']:>10.3f} "
            f"{path_record['estimated_grasp_time_s']:>12.3f} "
            f"{path_record['planning_time_s']:>12.3f} "
            f"{selected_marker:>9}"
        )

    if selected_path is not None:
        print("-" * 124)
        print(
            "Selected fastest path: "
            f"azimuth={selected_path['azimuth']} deg, "
            f"elevation={selected_path['elevation']} deg, "
            f"seed={selected_path['seed']}, "
            f"estimated_grasp_time="
            f"{selected_path['estimated_grasp_time_s']:.3f}s"
        )
    if actual_grasp_wall_time_s is not None:
        print(
            "Selected path actual wall-clock time to closed grasp: "
            f"{actual_grasp_wall_time_s:.3f}s"
        )
    print("=" * 124)


def save_candidate_paths(feasible_paths, selected_path):
    """Write all feasible pre-grasp paths for this trial to one CSV file."""
    output_path = TRIAL_RESULTS_DIR / "candidate_paths.csv"
    fieldnames = [
        "trial_id",
        "description",
        "rank",
        "azimuth_deg",
        "elevation_deg",
        "approach_clearance_m",
        "seed",
        "actions",
        "joint_length",
        "estimated_rrt_time_s",
        "estimated_approach_time_s",
        "estimated_close_time_s",
        "estimated_grasp_time_s",
        "planning_wall_time_s",
        "selected",
    ]
    ordered_paths = sorted(
        feasible_paths,
        key=lambda item: (
            item["estimated_grasp_time_s"],
            item["joint_length"],
            item["planning_time_s"],
        ),
    )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, path_record in enumerate(ordered_paths, start=1):
            writer.writerow(
                {
                    "trial_id": TRIAL_ID,
                    "description": TRIAL_DESCRIPTION,
                    "rank": rank,
                    "azimuth_deg": path_record["azimuth"],
                    "elevation_deg": path_record["elevation"],
                    "approach_clearance_m": (
                        f"{path_record['approach_clearance_m']:.6f}"
                    ),
                    "seed": path_record["seed"],
                    "actions": len(path_record["plan"]),
                    "joint_length": f"{path_record['joint_length']:.6f}",
                    "estimated_rrt_time_s": (
                        f"{path_record['estimated_rrt_time_s']:.6f}"
                    ),
                    "estimated_approach_time_s": (
                        f"{path_record['estimated_approach_time_s']:.6f}"
                    ),
                    "estimated_close_time_s": (
                        f"{path_record['estimated_close_time_s']:.6f}"
                    ),
                    "estimated_grasp_time_s": (
                        f"{path_record['estimated_grasp_time_s']:.6f}"
                    ),
                    "planning_wall_time_s": (
                        f"{path_record['planning_time_s']:.6f}"
                    ),
                    "selected": path_record is selected_path,
                }
            )
    return output_path


def upsert_trial_summary(summary):
    """Replace this trial's row in the combined summary instead of duplicating it."""
    output_path = RESULTS_ROOT / "trial_summary.csv"
    fieldnames = list(summary.keys())
    existing_rows = []

    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as input_file:
            existing_rows = list(csv.DictReader(input_file))

    trial_id_text = str(summary["trial_id"])
    existing_rows = [
        row for row in existing_rows if row.get("trial_id") != trial_id_text
    ]
    existing_rows.append({key: summary.get(key, "") for key in fieldnames})
    existing_rows.sort(key=lambda row: int(row["trial_id"]))

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
    return output_path


def save_trial_summary(summary):
    """Write detailed per-trial JSON and update the combined CSV summary."""
    json_path = TRIAL_RESULTS_DIR / "summary.json"
    json_payload = {
        "scene_config": TRIAL_CONFIG,
        "result": summary,
    }
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(json_payload, output_file, indent=2)

    csv_path = upsert_trial_summary(summary)
    return json_path, csv_path


# -----------------------------------------------------------------------------
# Front-facing initial state
# -----------------------------------------------------------------------------

print("=" * 76)
print(f"STANDALONE TRIAL {TRIAL_ID:02d}: {TRIAL_DESCRIPTION}")
print(f"Difficulty: {TRIAL_DIFFICULTY}")
print(f"Purpose: {TRIAL_PURPOSE}")
print(f"Target strawberry position: {FRUIT_POSITION}")
print("=" * 76)
trial_wall_start = time.perf_counter()
print("PHASE 1: front-facing home pose")
franka.set_joint_positions(HOME_JOINTS)
franka.set_joint_velocities(np.zeros(9))
home_action = ArticulationAction(joint_positions=HOME_JOINTS)

for _ in range(120):
    franka.apply_action(home_action)
    apply_gripper(GRIPPER_OPEN)
    world.step(render=MOTION_RENDER)

print("PHASE 1 RGB-D: initial occluded scene")
capture_rgbd(rgbd_camera, "01_initial")

base_position, base_orientation = franka.get_world_pose()
lula_ik.set_robot_base_pose(base_position, base_orientation)

rrt_obstacles = [plant_base, stem, leaf_0, leaf_1]
if leaf_2 is not None:
    rrt_obstacles.append(leaf_2)
rrt_obstacles.append(neighbor_fruit)

for obstacle in rrt_obstacles:
    result = rrt.add_obstacle(obstacle, static=True)
    print(f"RRT obstacle {obstacle.name}: {result}")

rrt.set_robot_base_pose(base_position, base_orientation)
rrt.update_world()


# -----------------------------------------------------------------------------
# Evaluate every candidate/seed path and select the fastest simulated grasp
# -----------------------------------------------------------------------------

print("PHASE 2: candidate planning")
selected_candidate = None
selected_harvest_path = None
harvest_plan = None
feasible_harvest_paths = []
screened_candidates = []

# Stage A: cheaply screen all 12 x 3 spherical approach directions.  This is
# where the final straight approach is checked against leaves and neighboring
# fruit; the old implementation skipped that collision test entirely.
for elevation in CANDIDATE_ELEVATIONS:
    for azimuth in CANDIDATE_AZIMUTHS:
        candidate = make_candidate(azimuth, elevation)
        approach_clear, reason, clearance = candidate_approach_is_clear(
            candidate
        )
        if not approach_clear:
            print(
                f"Candidate azimuth={azimuth} deg, elevation={elevation} deg "
                f"rejected: {reason}"
            )
            continue

        _, near_ik_success = articulation_ik.compute_inverse_kinematics(
            target_position=candidate["near_position"],
            target_orientation=candidate["orientation"],
        )
        if not near_ik_success:
            print(
                f"Candidate azimuth={azimuth} deg, elevation={elevation} deg "
                "rejected: pre-grasp IK failed"
            )
            continue

        candidate["approach_clearance_m"] = clearance
        candidate["home_distance_m"] = float(
            np.linalg.norm(candidate["near_position"] - ROBOT_BASE_POSITION)
        )
        screened_candidates.append(candidate)
        print(
            f"Candidate azimuth={azimuth} deg, elevation={elevation} deg "
            f"passed geometric/IK screening, clearance={clearance:.3f} m"
        )

# Stage B: run expensive RRT only for the safest directions.  Six directions
# times six seeds preserves the previous total of 36 sequential RRT calls.
screened_candidates.sort(
    key=lambda item: (
        -item["approach_clearance_m"],
        item["home_distance_m"],
    ),
)
rrt_candidates = screened_candidates[:MAX_RRT_DIRECTIONS]
print(
    f"3-D screening retained {len(screened_candidates)}/"
    f"{len(CANDIDATE_AZIMUTHS) * len(CANDIDATE_ELEVATIONS)} directions; "
    f"running RRT for {len(rrt_candidates)} safest directions."
)

for candidate in rrt_candidates:
    azimuth = candidate["azimuth"]
    elevation = candidate["elevation"]
    print(
        f"RRT candidate azimuth={azimuth} deg, "
        f"elevation={elevation} deg"
    )
    seed_paths = compute_rrt_seed_plans(
        candidate["near_position"],
        candidate["orientation"],
    )

    for seed_path in seed_paths:
        time_estimate = estimate_grasp_time(seed_path["plan"])
        path_record = {
            **seed_path,
            "angle": azimuth,
            "azimuth": azimuth,
            "elevation": elevation,
            "candidate": candidate,
            "approach_clearance_m": candidate["approach_clearance_m"],
            "estimated_grasp_time_s": time_estimate["total_time_s"],
            "estimated_rrt_time_s": time_estimate["rrt_time_s"],
            "estimated_approach_time_s": time_estimate["approach_time_s"],
            "estimated_close_time_s": time_estimate["close_time_s"],
        }
        feasible_harvest_paths.append(path_record)
        print(
            f"    feasible path: azimuth={azimuth} deg, "
            f"elevation={elevation} deg, "
            f"seed={seed_path['seed']}, "
            f"estimated_grasp_time={time_estimate['total_time_s']:.3f}s"
        )

if feasible_harvest_paths:
    selected_harvest_path = min(
        feasible_harvest_paths,
        key=lambda item: (
            item["estimated_grasp_time_s"],
            item["joint_length"],
            item["planning_time_s"],
        ),
    )
    selected_candidate = selected_harvest_path["candidate"]
    harvest_plan = selected_harvest_path["plan"]
    print(
        "Selected fastest harvesting path: "
        f"azimuth={selected_harvest_path['azimuth']} deg, "
        f"elevation={selected_harvest_path['elevation']} deg, "
        f"seed={selected_harvest_path['seed']}, "
        f"estimated_grasp_time="
        f"{selected_harvest_path['estimated_grasp_time_s']:.3f}s"
    )


harvest_succeeded = False
planning_success = harvest_plan is not None
approach_success = False
grasp_closed_success = False
drop_planning_success = False
collision_detected = False
failure_stage = ""
failure_reason = ""
final_hold_action = home_action
final_gripper_target = GRIPPER_OPEN
actual_grasp_wall_time_s = None
grasp_wall_accumulator_s = 0.0

if harvest_plan is None:
    print("No collision-free harvesting path was found.")
    failure_stage = "candidate_planning"
    failure_reason = "No collision-free pre-grasp path was found."

else:
    # -------------------------------------------------------------------------
    # Move to the near-grasp point and approach the fruit
    # -------------------------------------------------------------------------
    print("PHASE 3: move to near-grasp pose")
    segment_start = time.perf_counter()
    execute_plan(harvest_plan, GRIPPER_OPEN, label="harvest RRT")
    grasp_wall_accumulator_s += time.perf_counter() - segment_start

    print("PHASE 3 RGB-D: pre-grasp pose")
    capture_rgbd(rgbd_camera, "02_pregrasp")

    print("PHASE 4: straight approach")
    segment_start = time.perf_counter()
    approach_actions = execute_ik_line(
        selected_candidate["near_position"],
        FRUIT_POSITION,
        selected_candidate["orientation"],
        GRIPPER_OPEN,
    )
    grasp_wall_accumulator_s += time.perf_counter() - segment_start

    if approach_actions is None:
        print("Approach failed; the gripper will not close.")
        failure_stage = "straight_approach"
        failure_reason = "IK failed during the straight approach."

    else:
        approach_success = True
        # ---------------------------------------------------------------------
        # Verify the physical gripper pose before logical attachment
        # ---------------------------------------------------------------------
        print("PHASE 5: verify grasp pose, close gripper, and detach fruit")
        grasp_action = approach_actions[-1]

        segment_start = time.perf_counter()
        grasp_pose_reached, preclose_error = wait_until_grasp_pose(
            grasp_action
        )

        if not grasp_pose_reached:
            failure_stage = "grasp_pose"
            failure_reason = (
                "The gripper did not reach the strawberry; fruit was not "
                f"attached (position error={preclose_error:.4f} m)."
            )
            print(f"Grasp aborted: {failure_reason}")

        else:
            print(
                "Grasp pose reached: "
                f"position error={preclose_error:.4f} m"
            )
            for step in range(GRIPPER_CLOSE_FRAMES):
                alpha = float(step + 1) / float(GRIPPER_CLOSE_FRAMES)
                finger_target = (
                    (1.0 - alpha) * GRIPPER_OPEN
                    + alpha * GRIPPER_CLOSED
                )
                hold_action(grasp_action, finger_target, 1, carry=False)

            ee_position, _ = articulation_ik.compute_end_effector_pose()
            postclose_error = np.linalg.norm(
                np.asarray(ee_position, dtype=float) - FRUIT_POSITION
            )
            if postclose_error > GRASP_POSITION_TOLERANCE:
                failure_stage = "grasp_pose"
                failure_reason = (
                    "The gripper moved away while closing; fruit was not "
                    f"attached (position error={postclose_error:.4f} m)."
                )
                print(f"Grasp aborted: {failure_reason}")
            else:
                grasp_closed_success = True
                update_carried_fruit()
                final_hold_action = grasp_action
                final_gripper_target = GRIPPER_CLOSED

        grasp_wall_accumulator_s += time.perf_counter() - segment_start
        actual_grasp_wall_time_s = grasp_wall_accumulator_s

        # ---------------------------------------------------------------------
        # Pull the fruit back only to the collision-free pre-grasp point. This
        # is the necessary extraction motion; it does not return the arm home.
        # ---------------------------------------------------------------------
        print("PHASE 6: extract fruit to the pre-grasp point")
        if not grasp_closed_success:
            extraction_actions = []
            extraction_clear = False
            extraction_reason = "grasp was not established"
            print("Fruit extraction skipped because the grasp was not valid.")
        else:
            extraction_actions = list(reversed(approach_actions))
            extraction_actions.append(harvest_plan[-1])
            extraction_clear, extraction_reason = (
                carried_strawberry_path_is_clear(extraction_actions)
            )

        if not extraction_clear:
            if grasp_closed_success:
                collision_detected = True
                failure_stage = "fruit_extraction"
                failure_reason = extraction_reason
                print(
                    "Fruit extraction rejected by payload collision check: "
                    f"{extraction_reason}"
                )

        else:
            for index, action in enumerate(extraction_actions):
                hold_action(
                    action,
                    GRIPPER_CLOSED,
                    IK_HOLD_FRAMES,
                    carry=True,
                )
                print(
                    f"fruit extraction: {index + 1}/{len(extraction_actions)}"
                )

            final_hold_action = extraction_actions[-1]

            # -----------------------------------------------------------------
            # From the cleared pre-grasp point, plan directly to the drop zone.
            # The arm does not return home before or after placing the fruit.
            # -----------------------------------------------------------------
            print(
                "PHASE 7: plan to drop zone with carried-fruit "
                "collision checking"
            )
            drop_plan, drop_length = compute_best_rrt_plan(
                DROP_HOVER_POSITION,
                DROP_ORIENTATION,
                check_carried_strawberry=True,
            )

            if drop_plan is None:
                failure_stage = "drop_planning"
                failure_reason = "No collision-free path to the drop zone."
                print(
                    "Drop-zone planning failed; holding the fruit at the "
                    "pre-grasp point."
                )

            else:
                drop_planning_success = True
                print(f"Drop path selected, length={drop_length:.3f}")
                execute_plan(
                    drop_plan,
                    GRIPPER_CLOSED,
                    carry=True,
                    label="drop RRT",
                )
                final_hold_action = drop_plan[-1]

                # -------------------------------------------------------------
                # Lower toward the ground
                # -------------------------------------------------------------
                print("PHASE 8: lower fruit")
                lower_actions = execute_ik_line(
                    DROP_HOVER_POSITION,
                    DROP_RELEASE_POSITION,
                    DROP_ORIENTATION,
                    GRIPPER_CLOSED,
                    carry=True,
                )

                if lower_actions is None:
                    failure_stage = "lowering"
                    failure_reason = "IK failed while lowering the fruit."
                    print("Lowering failed; holding the fruit above the drop zone.")

                else:
                    release_action = lower_actions[-1]

                    # ---------------------------------------------------------
                    # Open the gripper and place the visual fruit on the ground
                    # ---------------------------------------------------------
                    print("PHASE 9: release fruit")
                    for step in range(GRIPPER_RELEASE_FRAMES):
                        alpha = float(step + 1) / float(GRIPPER_RELEASE_FRAMES)
                        finger_target = (
                            (1.0 - alpha) * GRIPPER_CLOSED
                            + alpha * GRIPPER_OPEN
                        )
                        hold_action(
                            release_action,
                            finger_target,
                            1,
                            carry=False,
                        )

                    fruit_start = target_strawberry_position.copy()
                    for position in np.linspace(
                        fruit_start,
                        DROP_GROUND_POSITION,
                        30,
                    ):
                        set_target_strawberry_position(position)
                        hold_action(
                            release_action,
                            GRIPPER_OPEN,
                            1,
                            carry=False,
                        )

                    set_target_strawberry_position(DROP_GROUND_POSITION)
                    final_hold_action = release_action
                    final_gripper_target = GRIPPER_OPEN
                    harvest_succeeded = True
                    failure_stage = ""
                    failure_reason = ""


if harvest_succeeded:
    capture_rgbd(rgbd_camera, "03_complete")
    print("=" * 76)
    print("HARVEST COMPLETE: fruit placed on ground.")
    print(f"RGB-D output directory: {RGBD_OUTPUT_DIR}")
    print("=" * 76)
else:
    capture_rgbd(rgbd_camera, "03_failure")
    print("=" * 76)
    print("Sequence ended before the strawberry was placed on the ground.")
    print(f"RGB-D failure evidence: {RGBD_OUTPUT_DIR}")
    print("=" * 76)


print_harvest_path_comparison(
    feasible_harvest_paths,
    selected_harvest_path,
    actual_grasp_wall_time_s,
)


candidate_csv_path = save_candidate_paths(
    feasible_harvest_paths,
    selected_harvest_path,
)
total_wall_time_s = time.perf_counter() - trial_wall_start
planning_failure = failure_stage in {"candidate_planning", "drop_planning"}
grasp_harvest_failure = (
    not harvest_succeeded and not planning_failure and not collision_detected
)

summary = {
    "trial_id": TRIAL_ID,
    "description": TRIAL_DESCRIPTION,
    "difficulty": TRIAL_DIFFICULTY,
    "fruit_position": json.dumps(FRUIT_POSITION.tolist()),
    "candidate_direction_count": (
        len(CANDIDATE_AZIMUTHS) * len(CANDIDATE_ELEVATIONS)
    ),
    "screened_safe_direction_count": len(screened_candidates),
    "rrt_direction_count": len(rrt_candidates),
    "feasible_path_count": len(feasible_harvest_paths),
    "selected_azimuth_deg": (
        selected_harvest_path["azimuth"]
        if selected_harvest_path is not None
        else None
    ),
    # Backward-compatible alias for older combined CSV files.
    "selected_angle_deg": (
        selected_harvest_path["azimuth"]
        if selected_harvest_path is not None
        else None
    ),
    "selected_elevation_deg": (
        selected_harvest_path["elevation"]
        if selected_harvest_path is not None
        else None
    ),
    "selected_seed": (
        selected_harvest_path["seed"]
        if selected_harvest_path is not None
        else None
    ),
    "selected_actions": (
        len(selected_harvest_path["plan"])
        if selected_harvest_path is not None
        else None
    ),
    "selected_joint_length": (
        round(selected_harvest_path["joint_length"], 6)
        if selected_harvest_path is not None
        else None
    ),
    "selected_estimated_grasp_time_s": (
        round(selected_harvest_path["estimated_grasp_time_s"], 6)
        if selected_harvest_path is not None
        else None
    ),
    "selected_planning_wall_time_s": (
        round(selected_harvest_path["planning_time_s"], 6)
        if selected_harvest_path is not None
        else None
    ),
    "actual_grasp_wall_time_s": (
        round(actual_grasp_wall_time_s, 6)
        if actual_grasp_wall_time_s is not None
        else None
    ),
    "planning_success": planning_success,
    "approach_success": approach_success,
    "grasp_closed_success": grasp_closed_success,
    "drop_planning_success": drop_planning_success,
    "collision_detected": collision_detected,
    "planning_failure": planning_failure,
    "grasp_harvest_failure": grasp_harvest_failure,
    "harvest_success": harvest_succeeded,
    "failure_stage": failure_stage,
    "failure_reason": failure_reason,
    "total_wall_time_s": round(total_wall_time_s, 6),
}
summary_json_path, combined_csv_path = save_trial_summary(summary)
print(f"Candidate-path records: {candidate_csv_path}")
print(f"Trial detail: {summary_json_path}")
print(f"Combined summary: {combined_csv_path}")


# Static display only. Keep the arm at its final release pose.
if args.auto_close:
    # The sequential launcher needs this process to exit before starting the
    # next trial. A short settling period is enough because all files are saved.
    for _ in range(30):
        if not simulation_app.is_running():
            break
        franka.apply_action(final_hold_action)
        apply_gripper(final_gripper_target)
        world.step(render=MOTION_RENDER)
else:
    while simulation_app.is_running():
        franka.apply_action(final_hold_action)
        apply_gripper(final_gripper_target)
        world.step(render=True)

simulation_app.close()


from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import time
from pathlib import Path
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

TRIAL_ID = 2
TRIAL_DESCRIPTION = "target left and slightly lower"
FRUIT_POSITION = np.array([0.520, 0.030, 0.480], dtype=float)
FRUIT_RADIUS = 0.030
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

CANDIDATE_ANGLES = [0, 30, -30, 60, -60, -90]
RRT_SEEDS = [123456, 2026, 42, 7, 99, 314159]

RRT_HOLD_FRAMES = 3
IK_HOLD_FRAMES = 6
APPROACH_WAYPOINTS = 16
LOWER_WAYPOINTS = 12
GRIPPER_CLOSE_FRAMES = 60
GRIPPER_RELEASE_FRAMES = 60
PHYSICS_DT = 1.0 / 60.0

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
    Path.cwd() / f"harvest_rgbd_output_trial_{TRIAL_ID:02d}"
)

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.10, 0.35, 1.00])
SOIL = np.array([0.48, 0.30, 0.12])


# Collision geometry is deliberately kept simple for Lula, while separate USD
# meshes below provide a more natural strawberry-and-leaf appearance.
PLANT_BASE_POSITION = np.array([0.55, 0.0, 0.06], dtype=float)
PLANT_BASE_SCALE = np.array([0.28, 0.28, 0.12], dtype=float)
STEM_POSITION = np.array([0.55, 0.0, 0.29], dtype=float)
STEM_SCALE = np.array([0.025, 0.025, 0.36], dtype=float)
LEAF_0_POSITION = np.array([0.485, 0.025, 0.525], dtype=float)
LEAF_0_SCALE = np.array([0.17, 0.060, 0.018], dtype=float)
LEAF_1_POSITION = np.array([0.595, -0.020, 0.530], dtype=float)
LEAF_1_SCALE = np.array([0.16, 0.065, 0.016], dtype=float)
NEIGHBOR_FRUIT_POSITION = np.array([0.555, -0.105, 0.490], dtype=float)
NEIGHBOR_FRUIT_RADIUS = 0.030


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


def make_candidate(angle_degrees):
    angle = np.deg2rad(angle_degrees)
    direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
    near_position = FRUIT_POSITION - NEAR_GRASP_DISTANCE * direction
    return {
        "angle": angle_degrees,
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


def create_strawberry_visual(stage, root_path, position, body_color):
    """Create a tapered strawberry body, calyx, stem, and visible seeds."""
    root = UsdGeom.Xform.Define(stage, root_path)
    translate_op = root.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*[float(value) for value in position]))

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


def create_leaf_visual(stage, path, position, length, width, color):
    """Create a pointed, slightly curved leaf with a lighter center vein."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*[float(value) for value in position]))

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
)
target_strawberry_position = FRUIT_POSITION.copy()

# These cuboids are Lula collision proxies. Their rendering is hidden after
# creation and replaced by pointed, curved leaf meshes of matching size.
leaf_0 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_0_Collision",
        name="leaf_0",
        position=LEAF_0_POSITION,
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
)
create_leaf_visual(
    world.stage,
    "/World/Plant/Leaf_1",
    LEAF_1_POSITION,
    LEAF_1_SCALE[0],
    max(0.065, LEAF_1_SCALE[1]),
    np.array([0.12, 0.70, 0.20]),
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
    np.array([0.80, 0.12, 0.08]),
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
        world.step(render=True)
        if carry:
            update_carried_fruit()


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


def sphere_intersects_box(sphere_center, sphere_radius, box_center, box_scale):
    half_extents = 0.5 * np.asarray(box_scale, dtype=float)
    lower = np.asarray(box_center, dtype=float) - half_extents
    upper = np.asarray(box_center, dtype=float) + half_extents
    closest = np.clip(np.asarray(sphere_center, dtype=float), lower, upper)
    distance = np.linalg.norm(np.asarray(sphere_center, dtype=float) - closest)
    return distance <= sphere_radius


def carried_strawberry_collision(fruit_center):
    """Return the first obstacle hit by the carried strawberry, or None."""
    effective_radius = PAYLOAD_COLLISION_RADIUS + PAYLOAD_CLEARANCE
    boxes = [
        ("plant_base", PLANT_BASE_POSITION, PLANT_BASE_SCALE),
        ("stem", STEM_POSITION, STEM_SCALE),
        ("leaf_0", LEAF_0_POSITION, LEAF_0_SCALE),
        ("leaf_1", LEAF_1_POSITION, LEAF_1_SCALE),
    ]
    for name, center, scale in boxes:
        if sphere_intersects_box(fruit_center, effective_radius, center, scale):
            # The target begins attached to the stem. Permit that local contact
            # only during the first few centimeters of the detachment motion.
            if (
                name == "stem"
                and np.linalg.norm(fruit_center - FRUIT_POSITION) < 0.045
            ):
                continue
            return name

    neighbor_clearance = (
        effective_radius + NEIGHBOR_FRUIT_RADIUS
    )
    if np.linalg.norm(fruit_center - NEIGHBOR_FRUIT_POSITION) <= neighbor_clearance:
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
    print("=" * 108)
    print("AVAILABLE HARVEST PATH TIME COMPARISON")
    print(
        "Time estimate = RRT execution + straight approach + gripper closing "
        "in simulated seconds."
    )

    if not feasible_paths:
        print("No feasible pre-grasp path was found.")
        print("=" * 108)
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
        f"{'Rank':>4} {'Angle':>7} {'Seed':>8} {'Actions':>8} "
        f"{'JointLen':>10} {'EstGrasp(s)':>12} {'PlanWall(s)':>12} {'Selected':>9}"
    )
    print("-" * 108)
    for rank, path_record in enumerate(ordered_paths, start=1):
        selected_marker = "YES" if path_record is selected_path else ""
        print(
            f"{rank:>4d} {path_record['angle']:>7d} "
            f"{path_record['seed']:>8d} {len(path_record['plan']):>8d} "
            f"{path_record['joint_length']:>10.3f} "
            f"{path_record['estimated_grasp_time_s']:>12.3f} "
            f"{path_record['planning_time_s']:>12.3f} "
            f"{selected_marker:>9}"
        )

    if selected_path is not None:
        print("-" * 108)
        print(
            "Selected fastest path: "
            f"angle={selected_path['angle']} deg, "
            f"seed={selected_path['seed']}, "
            f"estimated_grasp_time="
            f"{selected_path['estimated_grasp_time_s']:.3f}s"
        )
    if actual_grasp_wall_time_s is not None:
        print(
            "Selected path actual wall-clock time to closed grasp: "
            f"{actual_grasp_wall_time_s:.3f}s"
        )
    print("=" * 108)


# -----------------------------------------------------------------------------
# Front-facing initial state
# -----------------------------------------------------------------------------

print("=" * 76)
print(f"STANDALONE TRIAL {TRIAL_ID:02d}: {TRIAL_DESCRIPTION}")
print(f"Target strawberry position: {FRUIT_POSITION}")
print("=" * 76)
print("PHASE 1: front-facing home pose")
franka.set_joint_positions(HOME_JOINTS)
franka.set_joint_velocities(np.zeros(9))
home_action = ArticulationAction(joint_positions=HOME_JOINTS)

for _ in range(120):
    franka.apply_action(home_action)
    apply_gripper(GRIPPER_OPEN)
    world.step(render=True)

print("PHASE 1 RGB-D: initial occluded scene")
capture_rgbd(rgbd_camera, "01_initial")

base_position, base_orientation = franka.get_world_pose()
lula_ik.set_robot_base_pose(base_position, base_orientation)

for obstacle in [plant_base, stem, leaf_0, leaf_1, neighbor_fruit]:
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

for angle in CANDIDATE_ANGLES:
    candidate = make_candidate(angle)
    print(f"Candidate angle={angle} deg")
    seed_paths = compute_rrt_seed_plans(
        candidate["near_position"],
        candidate["orientation"],
    )

    for seed_path in seed_paths:
        time_estimate = estimate_grasp_time(seed_path["plan"])
        path_record = {
            **seed_path,
            "angle": angle,
            "candidate": candidate,
            "estimated_grasp_time_s": time_estimate["total_time_s"],
            "estimated_rrt_time_s": time_estimate["rrt_time_s"],
            "estimated_approach_time_s": time_estimate["approach_time_s"],
            "estimated_close_time_s": time_estimate["close_time_s"],
        }
        feasible_harvest_paths.append(path_record)
        print(
            f"    feasible path: angle={angle} deg, "
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
        f"angle={selected_harvest_path['angle']} deg, "
        f"seed={selected_harvest_path['seed']}, "
        f"estimated_grasp_time="
        f"{selected_harvest_path['estimated_grasp_time_s']:.3f}s"
    )


harvest_succeeded = False
final_hold_action = home_action
final_gripper_target = GRIPPER_OPEN
actual_grasp_wall_time_s = None
grasp_wall_accumulator_s = 0.0

if harvest_plan is None:
    print("No collision-free harvesting path was found.")

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

    else:
        # ---------------------------------------------------------------------
        # Close and detach logically
        # ---------------------------------------------------------------------
        print("PHASE 5: close gripper and detach fruit")
        grasp_action = approach_actions[-1]

        segment_start = time.perf_counter()
        for step in range(GRIPPER_CLOSE_FRAMES):
            alpha = float(step + 1) / float(GRIPPER_CLOSE_FRAMES)
            finger_target = (1.0 - alpha) * GRIPPER_OPEN + alpha * GRIPPER_CLOSED
            hold_action(grasp_action, finger_target, 1)
        grasp_wall_accumulator_s += time.perf_counter() - segment_start
        actual_grasp_wall_time_s = grasp_wall_accumulator_s

        update_carried_fruit()
        final_hold_action = grasp_action
        final_gripper_target = GRIPPER_CLOSED

        # ---------------------------------------------------------------------
        # Pull the fruit back only to the collision-free pre-grasp point. This
        # is the necessary extraction motion; it does not return the arm home.
        # ---------------------------------------------------------------------
        print("PHASE 6: extract fruit to the pre-grasp point")
        extraction_actions = list(reversed(approach_actions))
        extraction_actions.append(harvest_plan[-1])
        extraction_clear, extraction_reason = carried_strawberry_path_is_clear(
            extraction_actions
        )

        if not extraction_clear:
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
                print(
                    "Drop-zone planning failed; holding the fruit at the "
                    "pre-grasp point."
                )

            else:
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


# Static display only. Keep the arm at its final release pose.
while simulation_app.is_running():
    franka.apply_action(final_hold_action)
    apply_gripper(final_gripper_target)
    world.step(render=True)

simulation_app.close()

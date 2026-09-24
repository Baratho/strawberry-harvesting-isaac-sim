from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

from pathlib import Path

import numpy as np
from PIL import Image
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, FixedSphere, VisualSphere
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

FRUIT_POSITION = np.array([0.55, 0.00, 0.50], dtype=float)
ROBOT_BASE_POSITION = np.array([-0.15, 0.0, 0.0], dtype=float)
HOME_JOINTS = np.array(
    [0.00, -1.00, 0.00, -2.20, 0.00, 2.40, 0.80, 0.04, 0.04],
    dtype=float,
)

# A fixed third-person camera is easiest for the assignment video and for
# comparing occlusion across trials.
CAMERA_POSITION = np.array([1.20, -1.10, 0.95], dtype=float)
CAMERA_LOOK_AT = np.array([0.32, 0.00, 0.38], dtype=float)
CAMERA_RESOLUTION = (640, 480)  # width, height

OUTPUT_DIR = Path.cwd() / "rgbd_output"

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
SOIL = np.array([0.48, 0.30, 0.12])


# -----------------------------------------------------------------------------
# Camera orientation and image saving
# -----------------------------------------------------------------------------

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


def make_look_at_orientation(camera_position, target_position):
    """Camera pose for Isaac Sim's world axes: +X forward and +Z up."""
    forward = normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0])

    # Remove the forward component from world_up, producing the camera's +Z.
    camera_up = normalize(world_up - np.dot(world_up, forward) * forward)
    camera_left = normalize(np.cross(camera_up, forward))

    # Columns are the camera's local +X, +Y, +Z axes expressed in world frame.
    rotation = np.column_stack((forward, camera_left, camera_up))
    return rotation_matrix_to_quaternion(rotation)


def save_rgbd(camera, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    rgba = np.asarray(camera.get_rgba())
    rgb = rgba[..., :3]
    if rgb.dtype != np.uint8:
        if rgb.size > 0 and np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    frame = camera.get_current_frame()
    depth = frame.get("distance_to_image_plane")
    if depth is None:
        # Isaac Sim 5.0 also exposes the same metric-depth data directly.
        depth = camera.get_depth()

    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("The camera returned no valid depth pixels")

    # Save the original metric data for later visibility and 3D calculations.
    np.save(output_dir / "depth_meters.npy", depth)

    # Save a human-readable depth image. Near pixels are bright; far pixels dark.
    near = float(np.percentile(depth[valid], 1.0))
    far = float(np.percentile(depth[valid], 99.0))
    if far <= near:
        far = near + 1.0e-6

    depth_clipped = np.clip(depth, near, far)
    depth_visual = 255.0 * (1.0 - (depth_clipped - near) / (far - near))
    depth_visual[~valid] = 0.0
    depth_visual = depth_visual.astype(np.uint8)

    Image.fromarray(rgb).save(output_dir / "rgb.png")
    Image.fromarray(depth_visual, mode="L").save(output_dir / "depth_visual.png")

    camera_position, camera_orientation = camera.get_world_pose(camera_axes="world")
    np.savez(
        output_dir / "camera_metadata.npz",
        position=np.asarray(camera_position),
        orientation_wxyz=np.asarray(camera_orientation),
        resolution=np.asarray(CAMERA_RESOLUTION),
        depth_unit="meter",
    )

    return near, far, int(np.count_nonzero(valid)), depth.size


# -----------------------------------------------------------------------------
# Scene: same geometry used by the harvesting program
# -----------------------------------------------------------------------------

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

world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Base",
        name="plant_base",
        position=np.array([0.55, 0.0, 0.06]),
        size=1.0,
        scale=np.array([0.28, 0.28, 0.12]),
        color=SOIL,
    )
)

world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Stem",
        name="stem",
        position=np.array([0.55, 0.0, 0.29]),
        size=1.0,
        scale=np.array([0.025, 0.025, 0.36]),
        color=GREEN,
    )
)

world.scene.add(
    VisualSphere(
        prim_path="/World/Plant/TargetStrawberry",
        name="target_strawberry",
        position=FRUIT_POSITION,
        radius=0.030,
        color=RED,
    )
)

world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_0",
        name="leaf_0",
        position=np.array([0.485, 0.025, 0.525]),
        size=1.0,
        scale=np.array([0.17, 0.050, 0.025]),
        color=GREEN,
    )
)

world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_1",
        name="leaf_1",
        position=np.array([0.595, -0.020, 0.530]),
        size=1.0,
        scale=np.array([0.16, 0.050, 0.025]),
        color=GREEN,
    )
)

world.scene.add(
    FixedSphere(
        prim_path="/World/Plant/NeighborFruit_0",
        name="neighbor_fruit_0",
        position=np.array([0.555, -0.105, 0.490]),
        radius=0.030,
        color=np.array([0.75, 0.10, 0.08]),
    )
)

rgbd_camera = Camera(
    prim_path="/World/RGBDCamera",
    name="rgbd_camera",
    frequency=20,
    resolution=CAMERA_RESOLUTION,
)

world.reset()

# Camera.initialize() must be called after World.reset().
rgbd_camera.initialize()
rgbd_camera.set_world_pose(
    position=CAMERA_POSITION,
    orientation=make_look_at_orientation(CAMERA_POSITION, CAMERA_LOOK_AT),
    camera_axes="world",
)
rgbd_camera.set_clipping_range(near_distance=0.05, far_distance=10.0)
rgbd_camera.add_distance_to_image_plane_to_frame()

franka.set_joint_positions(HOME_JOINTS)
franka.set_joint_velocities(np.zeros(9))
home_action = ArticulationAction(joint_positions=HOME_JOINTS)

print("Warming up the RGB-D camera...")
for _ in range(120):
    if not simulation_app.is_running():
        break
    franka.apply_action(home_action)
    world.step(render=True)

try:
    near, far, valid_count, pixel_count = save_rgbd(rgbd_camera, OUTPUT_DIR)
    print("=" * 76)
    print("RGB-D CAPTURE COMPLETE")
    print(f"RGB image:       {OUTPUT_DIR / 'rgb.png'}")
    print(f"Depth preview:   {OUTPUT_DIR / 'depth_visual.png'}")
    print(f"Metric depth:    {OUTPUT_DIR / 'depth_meters.npy'}")
    print(f"Camera metadata: {OUTPUT_DIR / 'camera_metadata.npz'}")
    print(f"Valid depth:     {valid_count}/{pixel_count} pixels")
    print(f"Preview range:   {near:.3f} m to {far:.3f} m")
    print("=" * 76)
except Exception as error:
    print("=" * 76)
    print(f"RGB-D CAPTURE FAILED: {error}")
    print("=" * 76)

# Leave the GUI visible briefly, then close automatically.
for _ in range(300):
    if not simulation_app.is_running():
        break
    franka.apply_action(home_action)
    world.step(render=True)

simulation_app.close()

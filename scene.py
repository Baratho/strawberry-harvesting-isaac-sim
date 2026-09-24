# scene.py
# Isaac Sim 5.0.0
#
# Stage 1:
#   1. Load Franka Panda
#   2. Create a simplified strawberry plant
#   3. Add partial occlusions
#   4. Add an RGB-D camera
#   5. Save RGB and depth images
#   6. Keep the GUI open for inspection

from isaacsim import SimulationApp


# ---------------------------------------------------------------------
# Isaac Sim application
#
# headless=False:
#   Open the Isaac Sim visualization window.
# ---------------------------------------------------------------------
simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)


# Isaac Sim modules must be imported after SimulationApp is created.
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, FixedSphere
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera


# ---------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helper function: convert Isaac Sim RGB/RGBA image to uint8
# ---------------------------------------------------------------------
def convert_rgb_to_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise RuntimeError(f"Unexpected RGB image shape: {image.shape}")

    rgb = image[:, :, :3]

    # Some APIs return float images in [0, 1].
    # Others return uint8 images in [0, 255].
    if np.issubdtype(rgb.dtype, np.floating):
        if np.nanmax(rgb) <= 1.0:
            rgb = rgb * 255.0

    return np.clip(rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------
# Helper function: convert metric depth to a visible grayscale image
# ---------------------------------------------------------------------
def create_depth_visualization(depth):
    depth = np.asarray(depth).squeeze()

    valid_mask = np.isfinite(depth) & (depth > 0.0)
    visualization = np.zeros(depth.shape, dtype=np.uint8)

    if not np.any(valid_mask):
        return visualization

    valid_depth = depth[valid_mask]

    # Percentiles reduce the influence of very distant background pixels.
    near = np.percentile(valid_depth, 2)
    far = np.percentile(valid_depth, 98)

    if far <= near:
        far = near + 1e-6

    normalized = (depth - near) / (far - near)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Near objects are bright; distant objects are dark.
    visualization[valid_mask] = (
        (1.0 - normalized[valid_mask]) * 255.0
    ).astype(np.uint8)

    return visualization


# ---------------------------------------------------------------------
# 1. Create simulation world
# ---------------------------------------------------------------------
world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 30.0,
)

world.scene.add_default_ground_plane()


# ---------------------------------------------------------------------
# 2. Add lighting
# ---------------------------------------------------------------------
stage = world.stage

sunlight = UsdLux.DistantLight.Define(stage, "/World/Sunlight")
sunlight.CreateIntensityAttr(3000.0)
sunlight.CreateAngleAttr(1.0)


# ---------------------------------------------------------------------
# 3. Add Franka Panda
#
# The robot base is located at the world origin.
# ---------------------------------------------------------------------
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0]),
    )
)


# ---------------------------------------------------------------------
# 4. Add a small platform for the plant
# ---------------------------------------------------------------------
plant_platform = world.scene.add(
    FixedCuboid(
        prim_path="/World/PlantPlatform",
        name="plant_platform",
        position=np.array([0.55, 0.0, 0.05]),
        scale=np.array([0.32, 0.32, 0.10]),
        color=np.array([0.35, 0.20, 0.08]),
    )
)


# ---------------------------------------------------------------------
# 5. Add the plant stem
# ---------------------------------------------------------------------
stem = world.scene.add(
    FixedCuboid(
        prim_path="/World/Stem",
        name="stem",
        position=np.array([0.55, 0.0, 0.31]),
        scale=np.array([0.025, 0.025, 0.42]),
        color=np.array([0.08, 0.55, 0.10]),
    )
)


# ---------------------------------------------------------------------
# 6. Add a simplified strawberry
#
# FixedSphere is used during scene-development.
# Non-uniform scale makes it look more like an ellipsoid.
#
# Later, this will be replaced with a graspable/detachable object.
# ---------------------------------------------------------------------
strawberry = world.scene.add(
    FixedSphere(
        prim_path="/World/Strawberry",
        name="strawberry",
        position=np.array([0.55, 0.0, 0.50]),
        radius=0.045,
        scale=np.array([1.0, 1.0, 1.30]),
        color=np.array([0.95, 0.03, 0.03]),
    )
)


# ---------------------------------------------------------------------
# 7. Add leaves as collision obstacles
#
# They partially obscure the strawberry from some approach directions.
# ---------------------------------------------------------------------
leaf_left = world.scene.add(
    FixedCuboid(
        prim_path="/World/LeafLeft",
        name="leaf_left",
        position=np.array([0.52, 0.09, 0.53]),
        orientation=np.array(
            [0.9914, 0.0, 0.1305, 0.0]
        ),
        scale=np.array([0.22, 0.075, 0.012]),
        color=np.array([0.04, 0.75, 0.10]),
    )
)

leaf_right = world.scene.add(
    FixedCuboid(
        prim_path="/World/LeafRight",
        name="leaf_right",
        position=np.array([0.58, -0.09, 0.51]),
        orientation=np.array(
            [0.9914, 0.0, -0.1305, 0.0]
        ),
        scale=np.array([0.20, 0.075, 0.012]),
        color=np.array([0.04, 0.65, 0.08]),
    )
)


# ---------------------------------------------------------------------
# 8. Add another fruit as an additional obstacle
# ---------------------------------------------------------------------
neighbor_fruit = world.scene.add(
    FixedSphere(
        prim_path="/World/NeighborFruit",
        name="neighbor_fruit",
        position=np.array([0.57, -0.14, 0.45]),
        radius=0.040,
        scale=np.array([1.0, 1.0, 1.25]),
        color=np.array([0.95, 0.30, 0.05]),
    )
)


# ---------------------------------------------------------------------
# 9. Create an overhead RGB-D camera
#
# USD camera convention:
#   camera looks along its local -Z axis.
#
# With identity orientation, this camera looks downward along world -Z.
# ---------------------------------------------------------------------
camera = Camera(
    prim_path="/World/RGBCamera",
    position=np.array([0.48, 0.0, 1.60]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    frequency=30,
    resolution=(640, 480),
)


# ---------------------------------------------------------------------
# 10. Initialize the simulation
# ---------------------------------------------------------------------
print("Resetting world...", flush=True)
world.reset()

print("Initializing RGB-D camera...", flush=True)
camera.initialize()

# Attach the depth annotator.
# This gives depth measured perpendicular to the image plane.
camera.add_distance_to_image_plane_to_frame()


# ---------------------------------------------------------------------
# 11. Print useful scene information
# ---------------------------------------------------------------------
strawberry_position, strawberry_orientation = strawberry.get_world_pose()

print("Scene initialization complete.", flush=True)
print(
    f"Ground-truth strawberry position: {strawberry_position}",
    flush=True,
)
print(
    f"Ground-truth strawberry orientation: {strawberry_orientation}",
    flush=True,
)
print(
    "Close the Isaac Sim window to exit.",
    flush=True,
)


# ---------------------------------------------------------------------
# 12. Main simulation loop
# ---------------------------------------------------------------------
frame_index = 0
sensor_data_saved = False

while simulation_app.is_running():
    world.step(render=True)
    frame_index += 1

    # Wait for the renderer and camera annotators to become ready.
    if frame_index >= 60 and not sensor_data_saved:
        print("Reading camera data...", flush=True)

        # -------------------------------------------------------------
        # Save RGB
        # -------------------------------------------------------------
        rgba = camera.get_rgba()

        if rgba is None or np.asarray(rgba).size == 0:
            print(
                "Warning: RGB camera returned an empty image.",
                flush=True,
            )
        else:
            rgb_uint8 = convert_rgb_to_uint8(rgba)

            rgb_path = OUTPUT_DIR / "scene_rgb.png"
            Image.fromarray(rgb_uint8).save(rgb_path)

            print(f"RGB image saved to: {rgb_path}", flush=True)

        # -------------------------------------------------------------
        # Save metric depth and depth visualization
        # -------------------------------------------------------------
        camera_frame = camera.get_current_frame()
        depth = camera_frame.get("distance_to_image_plane")

        if depth is None or np.asarray(depth).size == 0:
            print(
                "Warning: depth camera returned an empty image.",
                flush=True,
            )
        else:
            depth = np.asarray(depth).squeeze()

            # Raw depth is saved in meters.
            depth_npy_path = OUTPUT_DIR / "scene_depth_meters.npy"
            np.save(depth_npy_path, depth)

            # Save a human-viewable grayscale depth image.
            depth_visualization = create_depth_visualization(depth)
            depth_png_path = OUTPUT_DIR / "scene_depth_visualization.png"
            Image.fromarray(depth_visualization).save(depth_png_path)

            valid_depth = depth[
                np.isfinite(depth) & (depth > 0.0)
            ]

            print(
                f"Raw depth saved to: {depth_npy_path}",
                flush=True,
            )
            print(
                f"Depth visualization saved to: {depth_png_path}",
                flush=True,
            )

            if valid_depth.size > 0:
                print(
                    "Valid depth range: "
                    f"{valid_depth.min():.3f} m to "
                    f"{valid_depth.max():.3f} m",
                    flush=True,
                )

        sensor_data_saved = True
        print(
            "Sensor data saved. The GUI will remain open.",
            flush=True,
        )


# ---------------------------------------------------------------------
# 13. Clean shutdown
# ---------------------------------------------------------------------
print("Closing Isaac Sim...", flush=True)
simulation_app.close()
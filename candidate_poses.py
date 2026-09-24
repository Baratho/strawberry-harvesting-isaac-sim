from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)


import numpy as np
from pxr import UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import (
    FixedCuboid,
    FixedSphere,
    VisualCuboid,
    VisualSphere,
)
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera


# =====================================================================
# Configuration
# =====================================================================

STRAWBERRY_POSITION = np.array([0.55, 0.0, 0.50])

PREGRASP_DISTANCE = 0.18
STRAWBERRY_SURFACE_DISTANCE = 0.05
GRIPPER_CLEARANCE = 0.035
NUMBER_OF_CANDIDATES = 12

CLEAR_COLOR = np.array([0.05, 1.0, 0.10])
BLOCKED_COLOR = np.array([1.0, 0.05, 0.05])


# =====================================================================
# Geometry functions
# =====================================================================

def normalize(vector):
    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        raise ValueError("Cannot normalize a zero vector.")

    return vector / norm


def quaternion_from_x_axis(direction):
    """
    Return a WXYZ quaternion that rotates local +X toward direction.

    This is used to align each visual line segment with its approach
    direction.
    """
    source = np.array([1.0, 0.0, 0.0])
    target = normalize(direction)

    dot_product = np.clip(
        np.dot(source, target),
        -1.0,
        1.0,
    )

    if dot_product > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])

    if dot_product < -0.999999:
        # 180-degree rotation about Z.
        return np.array([0.0, 0.0, 0.0, 1.0])

    rotation_axis = np.cross(source, target)

    quaternion = np.array(
        [
            1.0 + dot_product,
            rotation_axis[0],
            rotation_axis[1],
            rotation_axis[2],
        ]
    )

    return normalize(quaternion)


def segment_intersects_sphere(
    segment_start,
    segment_end,
    sphere_center,
    sphere_radius,
):
    """
    Test whether a line segment intersects an inflated sphere.
    """
    segment = segment_end - segment_start
    segment_length_squared = np.dot(segment, segment)

    if segment_length_squared < 1e-12:
        return (
            np.linalg.norm(segment_start - sphere_center)
            <= sphere_radius
        )

    interpolation = np.dot(
        sphere_center - segment_start,
        segment,
    ) / segment_length_squared

    interpolation = np.clip(interpolation, 0.0, 1.0)

    closest_point = (
        segment_start + interpolation * segment
    )

    distance = np.linalg.norm(
        closest_point - sphere_center
    )

    return distance <= sphere_radius


def segment_intersects_aabb(
    segment_start,
    segment_end,
    box_center,
    box_size,
    margin=0.0,
):
    """
    Test intersection between a line segment and an axis-aligned box.

    The box is inflated by margin to approximately account for the
    physical width of the gripper.
    """
    half_size = box_size / 2.0 + margin
    minimum = box_center - half_size
    maximum = box_center + half_size

    direction = segment_end - segment_start

    minimum_time = 0.0
    maximum_time = 1.0

    for axis in range(3):
        if abs(direction[axis]) < 1e-10:
            if (
                segment_start[axis] < minimum[axis]
                or segment_start[axis] > maximum[axis]
            ):
                return False
        else:
            inverse_direction = 1.0 / direction[axis]

            time_1 = (
                minimum[axis] - segment_start[axis]
            ) * inverse_direction

            time_2 = (
                maximum[axis] - segment_start[axis]
            ) * inverse_direction

            near_time = min(time_1, time_2)
            far_time = max(time_1, time_2)

            minimum_time = max(minimum_time, near_time)
            maximum_time = min(maximum_time, far_time)

            if minimum_time > maximum_time:
                return False

    return True


# =====================================================================
# Candidate generation
# =====================================================================

def generate_candidate_poses(target_position):
    """
    Generate approach candidates around the strawberry.

    approach_direction:
        Direction in which the gripper travels toward the strawberry.

    pregrasp_position:
        Position at which the robot pauses before the final approach.

    approach_end:
        A point near the strawberry surface used for collision checking.
    """
    candidates = []

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        NUMBER_OF_CANDIDATES,
        endpoint=False,
    )

    for candidate_id, angle in enumerate(angles):
        approach_direction = np.array(
            [
                np.cos(angle),
                np.sin(angle),
                0.0,
            ]
        )

        approach_direction = normalize(
            approach_direction
        )

        pregrasp_position = (
            target_position
            - PREGRASP_DISTANCE * approach_direction
        )

        approach_end = (
            target_position
            - STRAWBERRY_SURFACE_DISTANCE
            * approach_direction
        )

        candidates.append(
            {
                "id": candidate_id,
                "angle_degrees": np.degrees(angle),
                "approach_direction": approach_direction,
                "pregrasp_position": pregrasp_position,
                "approach_end": approach_end,
                "blocked": False,
                "blocked_by": [],
            }
        )

    return candidates


# =====================================================================
# Collision pre-filter
# =====================================================================

def evaluate_candidate_collisions(
    candidates,
    box_obstacles,
    sphere_obstacles,
):
    for candidate in candidates:
        start = candidate["pregrasp_position"]
        end = candidate["approach_end"]

        for obstacle in box_obstacles:
            collision = segment_intersects_aabb(
                segment_start=start,
                segment_end=end,
                box_center=obstacle["center"],
                box_size=obstacle["size"],
                margin=GRIPPER_CLEARANCE,
            )

            if collision:
                candidate["blocked"] = True
                candidate["blocked_by"].append(
                    obstacle["name"]
                )

        for obstacle in sphere_obstacles:
            collision = segment_intersects_sphere(
                segment_start=start,
                segment_end=end,
                sphere_center=obstacle["center"],
                sphere_radius=(
                    obstacle["radius"]
                    + GRIPPER_CLEARANCE
                ),
            )

            if collision:
                candidate["blocked"] = True
                candidate["blocked_by"].append(
                    obstacle["name"]
                )

    return candidates


# =====================================================================
# Candidate visualization
# =====================================================================

def visualize_candidates(world, candidates):
    for candidate in candidates:
        candidate_id = candidate["id"]

        start = candidate["pregrasp_position"]
        end = candidate["approach_end"]

        color = (
            BLOCKED_COLOR
            if candidate["blocked"]
            else CLEAR_COLOR
        )

        # Visual marker at the pregrasp position.
        world.scene.add(
            VisualSphere(
                prim_path=(
                    f"/World/Candidates/"
                    f"Candidate_{candidate_id:02d}_Marker"
                ),
                name=(
                    f"candidate_{candidate_id:02d}_marker"
                ),
                position=start,
                radius=0.014,
                color=color,
            )
        )

        # Visual line from pregrasp to the strawberry surface.
        line_vector = end - start
        line_length = np.linalg.norm(line_vector)
        line_center = (start + end) / 2.0

        line_orientation = quaternion_from_x_axis(
            line_vector
        )

        world.scene.add(
            VisualCuboid(
                prim_path=(
                    f"/World/Candidates/"
                    f"Candidate_{candidate_id:02d}_Line"
                ),
                name=f"candidate_{candidate_id:02d}_line",
                position=line_center,
                orientation=line_orientation,
                scale=np.array(
                    [
                        line_length,
                        0.007,
                        0.007,
                    ]
                ),
                color=color,
            )
        )


# =====================================================================
# Scene creation
# =====================================================================

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 30.0,
)

world.scene.add_default_ground_plane()


# Lighting
sunlight = UsdLux.DistantLight.Define(
    world.stage,
    "/World/Sunlight",
)

sunlight.CreateIntensityAttr(3000.0)
sunlight.CreateAngleAttr(1.0)


# Franka robot
franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0]),
    )
)


# Plant platform
world.scene.add(
    FixedCuboid(
        prim_path="/World/PlantPlatform",
        name="plant_platform",
        position=np.array([0.55, 0.0, 0.05]),
        scale=np.array([0.32, 0.32, 0.10]),
        color=np.array([0.35, 0.20, 0.08]),
    )
)


# Stem
world.scene.add(
    FixedCuboid(
        prim_path="/World/Stem",
        name="stem",
        position=np.array([0.55, 0.0, 0.28]),
        scale=np.array([0.025, 0.025, 0.36]),
        color=np.array([0.08, 0.55, 0.10]),
    )
)


# Target strawberry
strawberry = world.scene.add(
    FixedSphere(
        prim_path="/World/Strawberry",
        name="strawberry",
        position=STRAWBERRY_POSITION,
        radius=0.045,
        scale=np.array([1.0, 1.0, 1.30]),
        color=np.array([0.95, 0.03, 0.03]),
    )
)


# Leaves
leaf_left_center = np.array([0.52, 0.09, 0.53])
leaf_left_size = np.array([0.22, 0.075, 0.012])

world.scene.add(
    FixedCuboid(
        prim_path="/World/LeafLeft",
        name="leaf_left",
        position=leaf_left_center,
        orientation=np.array(
            [0.9914, 0.0, 0.1305, 0.0]
        ),
        scale=leaf_left_size,
        color=np.array([0.04, 0.75, 0.10]),
    )
)


leaf_right_center = np.array([0.58, -0.09, 0.51])
leaf_right_size = np.array([0.20, 0.075, 0.012])

world.scene.add(
    FixedCuboid(
        prim_path="/World/LeafRight",
        name="leaf_right",
        position=leaf_right_center,
        orientation=np.array(
            [0.9914, 0.0, -0.1305, 0.0]
        ),
        scale=leaf_right_size,
        color=np.array([0.04, 0.65, 0.08]),
    )
)


# Neighboring fruit
neighbor_center = np.array([0.57, -0.14, 0.45])
neighbor_radius = 0.040

world.scene.add(
    FixedSphere(
        prim_path="/World/NeighborFruit",
        name="neighbor_fruit",
        position=neighbor_center,
        radius=neighbor_radius,
        scale=np.array([1.0, 1.0, 1.25]),
        color=np.array([0.35, 0.75, 0.15]),
    )
)


# RGB-D camera
camera = Camera(
    prim_path="/World/RGBCamera",
    position=np.array([0.48, 0.0, 1.60]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    frequency=30,
    resolution=(640, 480),
)


# =====================================================================
# Generate and evaluate candidates
# =====================================================================

box_obstacles = [
    {
        "name": "leaf_left",
        "center": leaf_left_center,
        "size": leaf_left_size,
    },
    {
        "name": "leaf_right",
        "center": leaf_right_center,
        "size": leaf_right_size,
    },
]

sphere_obstacles = [
    {
        "name": "neighbor_fruit",
        "center": neighbor_center,
        "radius": neighbor_radius,
    },
]

candidates = generate_candidate_poses(
    STRAWBERRY_POSITION
)

candidates = evaluate_candidate_collisions(
    candidates=candidates,
    box_obstacles=box_obstacles,
    sphere_obstacles=sphere_obstacles,
)

visualize_candidates(
    world=world,
    candidates=candidates,
)


# =====================================================================
# Initialize
# =====================================================================

world.reset()
camera.initialize()
camera.add_distance_to_image_plane_to_frame()


# =====================================================================
# Print candidate evaluation
# =====================================================================

clear_candidates = []

print("\nCandidate approach evaluation")
print("=" * 80)

for candidate in candidates:
    if candidate["blocked"]:
        status = "BLOCKED"
        reason = ", ".join(
            candidate["blocked_by"]
        )
    else:
        status = "CLEAR"
        reason = "-"
        clear_candidates.append(candidate)

    print(
        f"Candidate {candidate['id']:02d} | "
        f"angle={candidate['angle_degrees']:6.1f} deg | "
        f"pregrasp={np.round(candidate['pregrasp_position'], 3)} | "
        f"{status:7s} | "
        f"blocked_by={reason}"
    )

print("=" * 80)
print(
    f"Clear candidates: "
    f"{len(clear_candidates)}/{len(candidates)}"
)
print(
    "Green = clear, red = geometrically blocked."
)
print(
    "Close the Isaac Sim window to exit.\n"
)


# =====================================================================
# Main loop
# =====================================================================

while simulation_app.is_running():
    world.step(render=True)


simulation_app.close()
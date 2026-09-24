from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

import numpy as np

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


# -----------------------------------------------------------------------------
# Scene and motion parameters
# -----------------------------------------------------------------------------

FRUIT_POSITION = np.array([0.55, 0.00, 0.50], dtype=float)
FRUIT_RADIUS = 0.030
NEAR_GRASP_DISTANCE = 0.08

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
RRT_SEEDS = [123456, 2026, 42]

RRT_HOLD_FRAMES = 3
IK_HOLD_FRAMES = 6
APPROACH_WAYPOINTS = 16
LOWER_WAYPOINTS = 12

# Drop zone is away from the plant.
DROP_HOVER_POSITION = np.array([0.25, -0.35, 0.24], dtype=float)
DROP_RELEASE_POSITION = np.array([0.25, -0.35, 0.09], dtype=float)
DROP_GROUND_POSITION = np.array([0.25, -0.35, FRUIT_RADIUS], dtype=float)

# Local +Z of the gripper points down.
DROP_ORIENTATION = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.10, 0.35, 1.00])
SOIL = np.array([0.48, 0.30, 0.12])


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


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 60.0,
)
world.scene.add_default_ground_plane()

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
        position=np.array([0.55, 0.0, 0.29]),
        size=1.0,
        scale=np.array([0.025, 0.025, 0.36]),
        color=GREEN,
    )
)

# VisualSphere is intentional: the assignment does not require a detachment
# model. After closing the gripper, the fruit is logically attached to it.
target_strawberry = world.scene.add(
    VisualSphere(
        prim_path="/World/Plant/TargetStrawberry",
        name="target_strawberry",
        position=FRUIT_POSITION,
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
        scale=np.array([0.17, 0.050, 0.025]),
        color=GREEN,
    )
)

leaf_1 = world.scene.add(
    FixedCuboid(
        prim_path="/World/Plant/Leaf_1",
        name="leaf_1",
        position=np.array([0.595, -0.020, 0.530]),
        size=1.0,
        scale=np.array([0.16, 0.050, 0.025]),
        color=GREEN,
    )
)

neighbor_fruit = world.scene.add(
    FixedSphere(
        prim_path="/World/Plant/NeighborFruit_0",
        name="neighbor_fruit_0",
        position=np.array([0.555, -0.105, 0.490]),
        radius=0.030,
        color=np.array([0.75, 0.10, 0.08]),
    )
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

world.reset()


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


def update_carried_fruit():
    ee_position, _ = articulation_ik.compute_end_effector_pose()
    target_strawberry.set_world_pose(position=ee_position)


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


def compute_best_rrt_plan(target_position, target_orientation):
    """Run several seeds and keep the shortest valid joint-space path."""
    rrt.set_end_effector_target(target_position, target_orientation)

    best_plan = None
    best_length = np.inf

    for seed in RRT_SEEDS:
        rrt.set_random_seed(seed)
        rrt.update_world()
        plan = path_visualizer.compute_plan_as_articulation_actions(
            max_cspace_dist=0.01
        )
        length = plan_length(plan)
        print(f"  seed={seed}, path_length={length:.3f}")

        if plan is not None and len(plan) > 0 and length < best_length:
            best_plan = plan
            best_length = length

    return best_plan, best_length


# -----------------------------------------------------------------------------
# Front-facing initial state
# -----------------------------------------------------------------------------

print("PHASE 1: front-facing home pose")
franka.set_joint_positions(HOME_JOINTS)
franka.set_joint_velocities(np.zeros(9))
home_action = ArticulationAction(joint_positions=HOME_JOINTS)

for _ in range(120):
    franka.apply_action(home_action)
    apply_gripper(GRIPPER_OPEN)
    world.step(render=True)

base_position, base_orientation = franka.get_world_pose()
lula_ik.set_robot_base_pose(base_position, base_orientation)

for obstacle in [plant_base, stem, leaf_0, leaf_1, neighbor_fruit]:
    result = rrt.add_obstacle(obstacle, static=True)
    print(f"RRT obstacle {obstacle.name}: {result}")

rrt.set_robot_base_pose(base_position, base_orientation)
rrt.update_world()


# -----------------------------------------------------------------------------
# Select a reachable candidate and prefer the shortest path among RRT seeds
# -----------------------------------------------------------------------------

print("PHASE 2: candidate planning")
selected_candidate = None
harvest_plan = None

for angle in CANDIDATE_ANGLES:
    candidate = make_candidate(angle)
    print(f"Candidate angle={angle} deg")
    plan, length = compute_best_rrt_plan(
        candidate["near_position"],
        candidate["orientation"],
    )

    if plan is not None:
        selected_candidate = candidate
        harvest_plan = plan
        print(f"Selected angle={angle} deg, shortest_length={length:.3f}")
        break


harvest_succeeded = False
final_home_action = home_action

if harvest_plan is None:
    print("No collision-free harvesting path was found.")

else:
    # -------------------------------------------------------------------------
    # Move to the near-grasp point and approach the fruit
    # -------------------------------------------------------------------------
    print("PHASE 3: move to near-grasp pose")
    execute_plan(harvest_plan, GRIPPER_OPEN, label="harvest RRT")

    print("PHASE 4: straight approach")
    approach_actions = execute_ik_line(
        selected_candidate["near_position"],
        FRUIT_POSITION,
        selected_candidate["orientation"],
        GRIPPER_OPEN,
    )

    if approach_actions is None:
        print("Approach failed; the gripper will not close.")

    else:
        # ---------------------------------------------------------------------
        # Close and detach logically
        # ---------------------------------------------------------------------
        print("PHASE 5: close gripper and detach fruit")
        grasp_action = approach_actions[-1]

        for step in range(60):
            alpha = float(step + 1) / 60.0
            finger_target = (1.0 - alpha) * GRIPPER_OPEN + alpha * GRIPPER_CLOSED
            hold_action(grasp_action, finger_target, 1)

        update_carried_fruit()

        # ---------------------------------------------------------------------
        # Retract to home while carrying the fruit
        # ---------------------------------------------------------------------
        print("PHASE 6: retract with fruit")
        for index, action in enumerate(reversed(approach_actions)):
            hold_action(action, GRIPPER_CLOSED, IK_HOLD_FRAMES, carry=True)
            print(f"approach retract: {index + 1}/{len(approach_actions)}")

        execute_plan(
            harvest_plan,
            GRIPPER_CLOSED,
            carry=True,
            reverse=True,
            label="harvest return",
        )

        # ---------------------------------------------------------------------
        # Plan to the ground drop zone
        # ---------------------------------------------------------------------
        print("PHASE 7: plan to drop zone")
        drop_plan, drop_length = compute_best_rrt_plan(
            DROP_HOVER_POSITION,
            DROP_ORIENTATION,
        )

        if drop_plan is None:
            print("Drop-zone planning failed; holding the harvested fruit at home.")

        else:
            print(f"Drop path selected, length={drop_length:.3f}")
            execute_plan(
                drop_plan,
                GRIPPER_CLOSED,
                carry=True,
                label="drop RRT",
            )

            # -----------------------------------------------------------------
            # Lower toward the ground
            # -----------------------------------------------------------------
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

                # -------------------------------------------------------------
                # Open the gripper and place the visual fruit on the ground
                # -------------------------------------------------------------
                print("PHASE 9: release fruit")
                for step in range(60):
                    alpha = float(step + 1) / 60.0
                    finger_target = (
                        (1.0 - alpha) * GRIPPER_CLOSED + alpha * GRIPPER_OPEN
                    )
                    hold_action(release_action, finger_target, 1, carry=False)

                fruit_start, _ = target_strawberry.get_world_pose()
                for position in np.linspace(fruit_start, DROP_GROUND_POSITION, 30):
                    target_strawberry.set_world_pose(position=position)
                    hold_action(release_action, GRIPPER_OPEN, 1, carry=False)

                target_strawberry.set_world_pose(position=DROP_GROUND_POSITION)

                # -------------------------------------------------------------
                # Raise the gripper and reverse the drop path to home
                # -------------------------------------------------------------
                print("PHASE 10: reset robot")
                for action in reversed(lower_actions):
                    hold_action(action, GRIPPER_OPEN, IK_HOLD_FRAMES, carry=False)

                execute_plan(
                    drop_plan,
                    GRIPPER_OPEN,
                    carry=False,
                    reverse=True,
                    label="reset RRT",
                )

                final_home_action = drop_plan[0]
                harvest_succeeded = True


if harvest_succeeded:
    print("=" * 76)
    print("HARVEST COMPLETE: fruit placed on ground and robot reset.")
    print("=" * 76)
else:
    print("=" * 76)
    print("Sequence ended before place-and-reset completed.")
    print("=" * 76)


# Static display only. Do not recompute FK or move the fruit every frame.
while simulation_app.is_running():
    franka.apply_action(final_home_action)
    apply_gripper(GRIPPER_OPEN if harvest_succeeded else GRIPPER_CLOSED)
    world.step(render=True)

simulation_app.close()

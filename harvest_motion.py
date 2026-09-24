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
from isaacsim.core.api.objects import (
    FixedCuboid,
    FixedSphere,
    VisualSphere,
)
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.robot.manipulators.examples.franka import Franka

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    PathPlannerVisualizer,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT



FRUIT_POSITION = np.array(
    [0.55, 0.00, 0.50],
    dtype=float,
)


FRUIT_RADIUS = 0.030

NEAR_GRASP_DISTANCE = 0.08

CANDIDATE_ANGLES = [
    0,
    30,
    -30,
    60,
    -60,
    -90,
]


SAFE_HOME_JOINTS = np.array(
    [
        1.20,
        -1.00,
        0.00,
        -2.20,
        0.00,
        2.40,
        0.80,
        0.04,
        0.04,
    ],
    dtype=float,
)

GRIPPER_OPEN = np.array(
    [0.04, 0.04],
    dtype=float,
)


GRIPPER_CLOSED = np.array(
    [0.025, 0.025],
    dtype=float,
)

RRT_ACTION_HOLD_FRAMES = 3
IK_ACTION_HOLD_FRAMES = 6
APPROACH_WAYPOINTS = 16

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.10, 0.35, 1.00])
SOIL = np.array([0.48, 0.30, 0.12])




def normalize(vector):
    length = np.linalg.norm(vector)

    if length < 1.0e-8:
        raise ValueError("cannot nom")

    return vector / length


def rotation_matrix_to_quaternion(rotation):
    """
    3x3 to Isaac Sim 4：
    [w, x, y, z]
    """
    m = rotation
    trace = np.trace(m)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0

        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s

    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(
            1.0 + m[0, 0] - m[1, 1] - m[2, 2]
        ) * 2.0

        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s

    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(
            1.0 + m[1, 1] - m[0, 0] - m[2, 2]
        ) * 2.0

        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s

    else:
        s = np.sqrt(
            1.0 + m[2, 2] - m[0, 0] - m[1, 1]
        ) * 2.0

        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array(
        [w, x, y, z],
        dtype=float,
    )

    return normalize(quaternion)


def make_grasp_orientation(approach_direction):
    """
    gripper, vertical to z axis of stra
    """
    z_axis = normalize(approach_direction)
    world_up = np.array([0.0, 0.0, 1.0])

    y_axis = normalize(
        np.cross(z_axis, world_up)
    )

    x_axis = normalize(
        np.cross(y_axis, z_axis)
    )

    rotation_matrix = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return rotation_matrix_to_quaternion(
        rotation_matrix
    )


def make_candidate(angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)

    # approach direction
    approach_direction = np.array(
        [
            np.cos(angle_radians),
            np.sin(angle_radians),
            0.0,
        ],
        dtype=float,
    )

    near_grasp_position = (
        FRUIT_POSITION
        - NEAR_GRASP_DISTANCE
        * approach_direction
    )

    orientation = make_grasp_orientation(
        approach_direction
    )

    return {
        "angle": angle_degrees,
        "direction": approach_direction,
        "near_position": near_grasp_position,
        "orientation": orientation,
    }



def apply_gripper_target(target_positions):
    franka.gripper.apply_action(
        ArticulationAction(
            joint_positions=np.array(
                target_positions,
                dtype=float,
            )
        )
    )


def hold_arm_action(
    arm_action,
    gripper_positions,
    frames,
    carry_strawberry=False,
):
    
    for _ in range(frames):
        if not simulation_app.is_running():
            return

        franka.apply_action(arm_action)
        apply_gripper_target(gripper_positions)

        world.step(render=True)

        if carry_strawberry:
            update_attached_strawberry()


def update_attached_strawberry():
    """
    put target to gripper center
    """
    ee_position, _ = (
        articulation_ik_solver
        .compute_end_effector_pose()
    )

    target_strawberry.set_world_pose(
        position=ee_position
    )



candidate_goals = [
    make_candidate(angle)
    for angle in CANDIDATE_ANGLES
]



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
        position=np.array([0.0, 0.0, 0.0]),
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



world.reset()

print()
print("=" * 76)
print("PHASE 1：initial")

franka.set_joint_positions(
    SAFE_HOME_JOINTS
)

franka.set_joint_velocities(
    np.zeros(9)
)

safe_home_action = ArticulationAction(
    joint_positions=SAFE_HOME_JOINTS
)

for _ in range(120):
    franka.apply_action(safe_home_action)
    apply_gripper_target(GRIPPER_OPEN)
    world.step(render=True)

print(
    "current position：",
    np.round(
        franka.get_joint_positions(),
        4,
    ),
)


# ============================================================
#  Lula IK
# ============================================================

kinematics_config = (
    interface_config_loader
    .load_supported_lula_kinematics_solver_config(
        "Franka"
    )
)

lula_ik_solver = LulaKinematicsSolver(
    **kinematics_config
)

articulation_ik_solver = (
    ArticulationKinematicsSolver(
        franka,
        lula_ik_solver,
        "right_gripper",
    )
)

robot_base_position, robot_base_orientation = (
    franka.get_world_pose()
)

lula_ik_solver.set_robot_base_pose(
    robot_base_position,
    robot_base_orientation,
)


# ============================================================
#  RRT
# ============================================================



rrt_config = (
    interface_config_loader
    .load_supported_path_planner_config(
        "Franka",
        "RRT",
    )
)

rrt = RRT(**rrt_config)
rrt.set_max_iterations(15000)

path_visualizer = PathPlannerVisualizer(
    franka,
    rrt,
)

planning_obstacles = [
    plant_base,
    stem,
    leaf_0,
    leaf_1,
    neighbor_fruit,
]

for obstacle in planning_obstacles:
    success = rrt.add_obstacle(
        obstacle,
        static=True,
    )

    print(
        f" {obstacle.name}: "
        f"{'SUCCESS' if success else 'FAILED'}"
    )

rrt.set_robot_base_pose(
    robot_base_position,
    robot_base_orientation,
)

rrt.update_world()


# ============================================================
# search RRT
# ============================================================

print("=" * 76)
print("PHASE 3：search the route")
print("=" * 76)

selected_candidate = None
selected_plan = None

random_seeds = [
    123456,
    2026,
    42,
]

for candidate in candidate_goals:
    print()
    print(
        f"检查 angle={candidate['angle']} deg"
    )
    print(
        "near the position：",
        np.round(
            candidate["near_position"],
            4,
        ),
    )

    rrt.set_end_effector_target(
        candidate["near_position"],
        candidate["orientation"],
    )

    for seed in random_seeds:
        print(f"  RRT seed={seed}")

        rrt.set_random_seed(seed)
        rrt.update_world()

        plan = (
            path_visualizer
            .compute_plan_as_articulation_actions(
                max_cspace_dist=0.01
            )
        )

        if plan is not None and len(plan) > 0:
            selected_candidate = candidate
            selected_plan = plan

            print("  RRT PATH FOUND")
            print(
                f"  num：{len(plan)}"
            )
            break

        print("  no path")

    if selected_plan is not None:
        break


# ============================================================
# gras
# ============================================================

if selected_plan is None:
    print()
    print("=" * 76)
    print("all fail。")
    print("=" * 76)

else:
    print()
    print("=" * 76)
    print(
        f"  angle="
        f"{selected_candidate['angle']} deg"
    )
    print(
        "  near_position=",
        np.round(
            selected_candidate["near_position"],
            4,
        ),
    )


    world.scene.add(
        VisualSphere(
            prim_path="/World/Targets/NearGrasp",
            name="near_grasp_marker",
            position=selected_candidate[
                "near_position"
            ],
            radius=0.012,
            color=BLUE,
        )
    )

    # --------------------------------------------------------
    # wait for 3 s
    # --------------------------------------------------------

    for _ in range(180):
        franka.apply_action(safe_home_action)
        apply_gripper_target(GRIPPER_OPEN)
        world.step(render=True)

    # --------------------------------------------------------
    #  RRT to grasp piont
    # --------------------------------------------------------

    print("PHASE 4： RRT route")

    for action_index, action in enumerate(
        selected_plan
    ):
        hold_arm_action(
            arm_action=action,
            gripper_positions=GRIPPER_OPEN,
            frames=RRT_ACTION_HOLD_FRAMES,
        )

        if (
            action_index % 25 == 0
            or action_index
            == len(selected_plan) - 1
        ):
            print(
                f"  RRT "
                f"{action_index + 1}/"
                f"{len(selected_plan)}"
            )

    near_grasp_arm_action = selected_plan[-1]


    print("PHASE 5：open the girpper")

    hold_arm_action(
        arm_action=near_grasp_arm_action,
        gripper_positions=GRIPPER_OPEN,
        frames=90,
    )


    print("PHASE 6：approch the target")

    approach_positions = np.linspace(
        selected_candidate["near_position"],
        FRUIT_POSITION,
        APPROACH_WAYPOINTS,
    )

    approach_actions = []
    approach_success = True

    for waypoint_index, waypoint_position in enumerate(
        approach_positions[1:]
    ):
        ik_action, ik_success = (
            articulation_ik_solver
            .compute_inverse_kinematics(
                target_position=waypoint_position,
                target_orientation=selected_candidate[
                    "orientation"
                ],
            )
        )

        if not ik_success:
            print(
               
                f"{waypoint_index + 1} fail"
            )

            approach_success = False
            break

        approach_actions.append(ik_action)

        hold_arm_action(
            arm_action=ik_action,
            gripper_positions=GRIPPER_OPEN,
            frames=IK_ACTION_HOLD_FRAMES,
        )

        print(
            f"  approch "
            f"{waypoint_index + 1}/"
            f"{APPROACH_WAYPOINTS - 1}"
        )

    # --------------------------------------------------------
    # withdraw
    # --------------------------------------------------------

    if not approach_success:
        print()
        print("=" * 76)
        print("fail")
        print("=" * 76)

    else:
        grasp_arm_action = approach_actions[-1]

        print("PHASE 7：close gripper")

       
        closing_steps = 60

        for step in range(closing_steps):
            alpha = (
                float(step + 1)
                / float(closing_steps)
            )

            finger_target = (
                (1.0 - alpha) * GRIPPER_OPEN
                + alpha * GRIPPER_CLOSED
            )

            hold_arm_action(
                arm_action=grasp_arm_action,
                gripper_positions=finger_target,
                frames=1,
            )

     
        update_attached_strawberry()

        print("PHASE 8：get the target")

        for reverse_index, action in enumerate(
            reversed(approach_actions)
        ):
            hold_arm_action(
                arm_action=action,
                gripper_positions=GRIPPER_CLOSED,
                frames=IK_ACTION_HOLD_FRAMES,
                carry_strawberry=True,
            )

            print(
    
                f"{reverse_index + 1}/"
                f"{len(approach_actions)}"
            )

        


# ============================================================
# keep window open
# ============================================================

while simulation_app.is_running():
    if (
        selected_plan is not None
        and "approach_success" in globals()
        and approach_success
    ):
        apply_gripper_target(
            GRIPPER_CLOSED
        )
        update_attached_strawberry()

    world.step(render=True)

simulation_app.close()

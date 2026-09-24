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
    PathPlannerVisualizer,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT


# ============================================================
# 参数
# ============================================================

FRUIT_POSITION = np.array(
    [0.55, 0.00, 0.50],
    dtype=float,
)

FRUIT_RADIUS = 0.045
PREGRASP_DISTANCE = 0.16

# 前一步中通过几何过滤和 IK 的候选
REACHABLE_CANDIDATE_ANGLES = [
    0,      # Candidate 00
    30,     # Candidate 01
    -30,    # Candidate 02
    60,     # Candidate 03
    -60,    # Candidate 04
    -90,    # Candidate 06
]

# 安全初始姿态：7 个手臂关节 + 2 个手指关节
# q1 设置为 1.2，使机械臂先转到植物侧面
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

ACTION_HOLD_FRAMES = 3
MAX_PLANNING_ATTEMPTS = 3

GREEN = np.array([0.10, 0.75, 0.25])
RED = np.array([0.90, 0.08, 0.08])
BLUE = np.array([0.10, 0.35, 1.00])
SOIL = np.array([0.48, 0.30, 0.12])


# ============================================================
# 数学工具
# ============================================================

def normalize(vector):
    length = np.linalg.norm(vector)

    if length < 1.0e-8:
        raise ValueError("不能归一化零向量")

    return vector / length


def rotation_matrix_to_quaternion(rotation):
    """
    旋转矩阵转换成 Isaac Sim 四元数：
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
    让 right_gripper 的局部 +Z 轴朝向草莓，
    同时尽量保持夹爪向上。
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

    approach_direction = np.array(
        [
            np.cos(angle_radians),
            np.sin(angle_radians),
            0.0,
        ],
        dtype=float,
    )

    position = (
        FRUIT_POSITION
        - PREGRASP_DISTANCE * approach_direction
    )

    orientation = make_grasp_orientation(
        approach_direction
    )

    return {
        "angle": angle_degrees,
        "position": position,
        "orientation": orientation,
    }


# ============================================================
# 创建候选列表
# ============================================================

candidate_goals = [
    make_candidate(angle)
    for angle in REACHABLE_CANDIDATE_ANGLES
]


# ============================================================
# 创建场景
# ============================================================

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 60.0,
    rendering_dt=1.0 / 60.0,
)

world.scene.add_default_ground_plane()


# ============================================================
# 创建 Franka
# ============================================================

franka = world.scene.add(
    Franka(
        prim_path="/World/Franka",
        name="franka",
        position=np.array([0.0, 0.0, 0.0]),
    )
)


# ============================================================
# 创建植物
# ============================================================

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
    FixedSphere(
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
        radius=0.035,
        color=np.array([0.75, 0.10, 0.08]),
    )
)


# ============================================================
# 初始化仿真
# ============================================================

world.reset()


# ============================================================
# 将 Franka 放到安全初始姿态
# ============================================================

print()
print("=" * 76)
print("正在将 Franka 放到安全初始姿态……")

# 立即设置关节状态
franka.set_joint_positions(
    SAFE_HOME_JOINTS
)

franka.set_joint_velocities(
    np.zeros(9)
)

# 同时把安全姿态设为控制目标，防止机器人弹回默认姿态
safe_home_action = ArticulationAction(
    joint_positions=SAFE_HOME_JOINTS
)

for _ in range(120):
    if not simulation_app.is_running():
        break

    franka.apply_action(safe_home_action)
    world.step(render=True)

actual_start_joints = franka.get_joint_positions()

print(
    "当前 Franka 关节位置：",
    np.round(actual_start_joints, 4),
)

print("=" * 76)


# ============================================================
# 初始化 Lula RRT
# ============================================================

print("正在加载 Franka 的 Lula RRT 配置……")

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


# ============================================================
# 添加障碍物
# ============================================================

# 目标草莓没有加入下面的列表。
# 原因：它是机器人最终需要接触的对象。
# 当前目标在草莓外 16 cm，仍属于预抓取位置。
planning_obstacles = [
    plant_base,
    stem,
    leaf_0,
    leaf_1,
    neighbor_fruit,
]

print()
print("正在向 RRT 添加障碍物：")

all_obstacles_added = True

for obstacle in planning_obstacles:
    success = rrt.add_obstacle(
        obstacle,
        static=True,
    )

    print(
        f"  {obstacle.name}: "
        f"{'SUCCESS' if success else 'FAILED'}"
    )

    if not success:
        all_obstacles_added = False

if not all_obstacles_added:
    print("警告：至少有一个障碍物没有成功加入 RRT。")


# ============================================================
# 更新机器人底座
# ============================================================

robot_base_position, robot_base_orientation = (
    franka.get_world_pose()
)

rrt.set_robot_base_pose(
    robot_base_position,
    robot_base_orientation,
)

rrt.update_world()


# ============================================================
# 对所有 IK 可达候选运行 RRT
# ============================================================

print()
print("=" * 76)
print("开始对 6 个 IK 可达候选运行 RRT")
print("=" * 76)

selected_candidate = None
selected_plan = None

random_seeds = [
    123456,
    2026,
    42,
]

for candidate_index, candidate in enumerate(
    candidate_goals
):
    print()
    print(
        f"检查候选 {candidate_index}: "
        f"angle={candidate['angle']} deg"
    )

    print(
        "  position=",
        np.round(candidate["position"], 4),
    )

    print(
        "  orientation=",
        np.round(candidate["orientation"], 4),
    )

    rrt.set_end_effector_target(
        candidate["position"],
        candidate["orientation"],
    )

    for attempt, seed in enumerate(random_seeds):
        print(
            f"  RRT 尝试 {attempt + 1}/"
            f"{MAX_PLANNING_ATTEMPTS}, "
            f"seed={seed}"
        )

        rrt.set_random_seed(seed)
        rrt.update_world()

        candidate_plan = (
            path_visualizer
            .compute_plan_as_articulation_actions(
                max_cspace_dist=0.01
            )
        )

        if (
            candidate_plan is not None
            and len(candidate_plan) > 0
        ):
            selected_candidate = candidate
            selected_plan = candidate_plan

            print("  RRT PATH FOUND")
            print(
                f"  路径动作数量："
                f"{len(selected_plan)}"
            )
            break

        print("  本次没有找到路径。")

    if selected_plan is not None:
        break


# ============================================================
# 显示并执行选中的路径
# ============================================================

if selected_plan is None:
    print()
    print("=" * 76)
    print("没有候选位姿获得完整 RRT 路径。")
    print("机械臂不会运动。")
    print("请把 Candidate 检查部分的输出发给我。")
    print("=" * 76)

else:
    print()
    print("=" * 76)
    print("RRT 路径规划成功")
    print(
        f"选中角度："
        f"{selected_candidate['angle']} deg"
    )
    print(
        "选中位置：",
        np.round(
            selected_candidate["position"],
            4,
        ),
    )
    print(
        f"路径动作数量：{len(selected_plan)}"
    )
    print("3 秒后开始执行。")
    print("=" * 76)

    # 显示最终选择的预抓取点
    world.scene.add(
        VisualSphere(
            prim_path="/World/Targets/SelectedPregrasp",
            name="selected_pregrasp",
            position=selected_candidate["position"],
            radius=0.018,
            color=BLUE,
        )
    )

    # 等待约 3 秒
    for _ in range(180):
        if not simulation_app.is_running():
            break

        franka.apply_action(safe_home_action)
        world.step(render=True)

    final_action = selected_plan[-1]

    for action_index, action in enumerate(
        selected_plan
    ):
        if not simulation_app.is_running():
            break

        for _ in range(ACTION_HOLD_FRAMES):
            franka.apply_action(action)
            world.step(render=True)

        if (
            action_index % 25 == 0
            or action_index
            == len(selected_plan) - 1
        ):
            print(
                f"执行路径点 "
                f"{action_index + 1}/"
                f"{len(selected_plan)}"
            )

    print()
    print("=" * 76)
    print("RRT 路径执行完成。")
    print("机械臂应位于蓝色预抓取点附近。")
    print("=" * 76)

    # 保持最终姿态约 2 秒
    for _ in range(120):
        if not simulation_app.is_running():
            break

        franka.apply_action(final_action)
        world.step(render=True)


# ============================================================
# 保持窗口开启
# ============================================================

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()

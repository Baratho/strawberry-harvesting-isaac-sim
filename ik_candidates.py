from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
    }
)

# Isaac Sim 模块必须在 SimulationApp 创建之后导入
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.objects import (
    FixedCuboid,
    FixedSphere,
    VisualCuboid,
    VisualSphere,
)
from isaacsim.robot.manipulators.examples.franka import Franka

from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)


# ============================================================
# 场景和候选位姿参数
# ============================================================

FRUIT_POSITION = np.array([0.55, 0.00, 0.50], dtype=float)
FRUIT_RADIUS = 0.045

# 预抓取点距离草莓中心 16 cm
PREGRASP_DISTANCE = 0.16

# 碰撞线终点距离草莓表面再保留 1 cm
SURFACE_OFFSET = 0.010

# 粗略几何过滤的安全余量
LEAF_CLEARANCE = 0.008
NEIGHBOR_CLEARANCE = 0.015

# 优先从靠近机器人一侧开始测试
CANDIDATE_ANGLES_DEG = [
    0,
    30,
    -30,
    60,
    -60,
    90,
    -90,
    120,
    -120,
    150,
    -150,
    180,
]

# 颜色
RED = np.array([0.90, 0.08, 0.08])
GREEN = np.array([0.10, 0.75, 0.25])
BLUE = np.array([0.10, 0.35, 1.00])
YELLOW = np.array([1.00, 0.70, 0.05])
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
    把 3x3 旋转矩阵转换成 Isaac Sim 使用的四元数：
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

    quaternion = np.array([w, x, y, z], dtype=float)

    return normalize(quaternion)


def make_grasp_orientation(approach_direction):
    """
    创建夹爪目标姿态。

    right_gripper 的局部 +Z 轴朝向草莓。
    局部 X 轴尽量保持向上。
    """
    z_axis = normalize(approach_direction)

    world_up = np.array([0.0, 0.0, 1.0])

    # 当前候选均为水平接近，所以不会与 world_up 平行
    y_axis = normalize(np.cross(z_axis, world_up))
    x_axis = normalize(np.cross(y_axis, z_axis))

    rotation_matrix = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return rotation_matrix_to_quaternion(rotation_matrix)


def quaternion_for_x_axis(direction):
    """
    创建一个四元数，使长方体局部 X 轴沿着 direction。
    用于显示候选接近线。
    """
    x_axis = normalize(direction)
    helper = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(x_axis, helper)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0])

    y_axis = normalize(np.cross(helper, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    rotation_matrix = np.column_stack(
        (
            x_axis,
            y_axis,
            z_axis,
        )
    )

    return rotation_matrix_to_quaternion(rotation_matrix)


# ============================================================
# 几何碰撞检测
# ============================================================

def segment_intersects_aabb(
    start,
    end,
    box_center,
    box_half_size,
    padding=0.0,
):
    """
    检查线段是否穿过轴对齐包围盒 AABB。
    """
    lower = box_center - box_half_size - padding
    upper = box_center + box_half_size + padding

    direction = end - start

    t_min = 0.0
    t_max = 1.0

    for axis in range(3):
        if abs(direction[axis]) < 1.0e-8:
            if (
                start[axis] < lower[axis]
                or start[axis] > upper[axis]
            ):
                return False

        else:
            inverse_direction = 1.0 / direction[axis]

            t1 = (
                lower[axis] - start[axis]
            ) * inverse_direction

            t2 = (
                upper[axis] - start[axis]
            ) * inverse_direction

            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)

            if t_min > t_max:
                return False

    return True


def point_segment_distance(point, start, end):
    """
    计算一个点到线段的最短距离。
    """
    segment = end - start
    denominator = np.dot(segment, segment)

    if denominator < 1.0e-10:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / denominator
    t = np.clip(t, 0.0, 1.0)

    closest_point = start + t * segment

    return np.linalg.norm(point - closest_point)


def segment_intersects_sphere(
    start,
    end,
    sphere_center,
    sphere_radius,
    padding=0.0,
):
    """
    检查线段是否与球体相交。
    """
    distance = point_segment_distance(
        sphere_center,
        start,
        end,
    )

    return distance <= sphere_radius + padding


# ============================================================
# 候选位姿可视化
# ============================================================

def add_candidate_marker(
    world,
    index,
    position,
    color,
):
    world.scene.add(
        VisualSphere(
            prim_path=f"/World/Candidates/Marker_{index:02d}",
            name=f"candidate_marker_{index:02d}",
            position=position,
            radius=0.015,
            color=color,
        )
    )


def add_approach_line(
    world,
    index,
    start,
    fruit_center,
    color,
):
    """
    从预抓取点画到草莓表面外侧。
    """
    direction = normalize(fruit_center - start)

    line_end = (
        fruit_center
        - (FRUIT_RADIUS + SURFACE_OFFSET) * direction
    )

    line_vector = line_end - start
    line_length = np.linalg.norm(line_vector)
    midpoint = (start + line_end) * 0.5

    orientation = quaternion_for_x_axis(line_vector)

    world.scene.add(
        VisualCuboid(
            prim_path=f"/World/Candidates/Line_{index:02d}",
            name=f"candidate_line_{index:02d}",
            position=midpoint,
            orientation=orientation,
            size=1.0,
            scale=np.array(
                [
                    line_length,
                    0.006,
                    0.006,
                ]
            ),
            color=color,
        )
    )


# ============================================================
# 创建仿真世界
# ============================================================

world = World(stage_units_in_meters=1.0)

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
# 创建花盆和植物
# ============================================================

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


# ============================================================
# 创建目标草莓
# ============================================================

world.scene.add(
    FixedSphere(
        prim_path="/World/Plant/TargetStrawberry",
        name="target_strawberry",
        position=FRUIT_POSITION,
        radius=FRUIT_RADIUS,
        color=RED,
    )
)


# ============================================================
# 创建叶子
# ============================================================

leaf_obstacles = [
    {
        "center": np.array(
            [0.485, 0.025, 0.525]
        ),
        "size": np.array(
            [0.17, 0.050, 0.025]
        ),
    },
    {
        "center": np.array(
            [0.595, -0.020, 0.530]
        ),
        "size": np.array(
            [0.16, 0.050, 0.025]
        ),
    },
]

for leaf_index, leaf in enumerate(leaf_obstacles):
    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Plant/Leaf_{leaf_index}",
            name=f"leaf_{leaf_index}",
            position=leaf["center"],
            size=1.0,
            scale=leaf["size"],
            color=GREEN,
        )
    )


# ============================================================
# 创建邻近草莓
# ============================================================

neighbor_fruits = [
    {
        "center": np.array(
            [0.555, -0.105, 0.490]
        ),
        "radius": 0.035,
    },
]

for fruit_index, fruit in enumerate(neighbor_fruits):
    world.scene.add(
        FixedSphere(
            prim_path=(
                f"/World/Plant/NeighborFruit_{fruit_index}"
            ),
            name=f"neighbor_fruit_{fruit_index}",
            position=fruit["center"],
            radius=fruit["radius"],
            color=np.array([0.75, 0.10, 0.08]),
        )
    )


# ============================================================
# 初始化场景
# ============================================================

world.reset()


# ============================================================
# 初始化 Lula IK
# ============================================================

kinematics_config = (
    interface_config_loader
    .load_supported_lula_kinematics_solver_config(
        "Franka"
    )
)

lula_solver = LulaKinematicsSolver(
    **kinematics_config
)

ik_solver = ArticulationKinematicsSolver(
    franka,
    lula_solver,
    "right_gripper",
)

robot_base_position, robot_base_orientation = (
    franka.get_world_pose()
)

lula_solver.set_robot_base_pose(
    robot_base_position,
    robot_base_orientation,
)


# ============================================================
# 候选位姿几何筛选和 IK 筛选
# ============================================================

reachable_candidates = []

print()
print("=" * 76)
print("开始进行候选预抓取位姿筛选")
print("蓝色：几何路径通过，并且 IK 成功")
print("黄色：几何路径通过，但是 IK 失败")
print("红色：被叶子或邻近草莓阻挡")
print("=" * 76)


for candidate_index, angle_degrees in enumerate(
    CANDIDATE_ANGLES_DEG
):
    angle_radians = np.deg2rad(angle_degrees)

    # approach_direction：
    # 从预抓取点指向草莓中心的方向
    approach_direction = np.array(
        [
            np.cos(angle_radians),
            np.sin(angle_radians),
            0.0,
        ],
        dtype=float,
    )

    # 预抓取位置
    pregrasp_position = (
        FRUIT_POSITION
        - PREGRASP_DISTANCE * approach_direction
    )

    # 修复重点：
    # 碰撞检测只进行到草莓表面外侧，
    # 不再把线段终点放到草莓中心。
    collision_end = (
        FRUIT_POSITION
        - (
            FRUIT_RADIUS
            + SURFACE_OFFSET
        ) * approach_direction
    )

    blocked = False
    blocking_object = ""

    # --------------------------------------------------------
    # 检查叶子
    # --------------------------------------------------------

    for leaf_index, leaf in enumerate(leaf_obstacles):
        leaf_half_size = leaf["size"] * 0.5

        if segment_intersects_aabb(
            pregrasp_position,
            collision_end,
            leaf["center"],
            leaf_half_size,
            LEAF_CLEARANCE,
        ):
            blocked = True
            blocking_object = f"Leaf_{leaf_index}"
            break

    # --------------------------------------------------------
    # 检查邻近草莓
    # --------------------------------------------------------

    if not blocked:
        for fruit_index, fruit in enumerate(
            neighbor_fruits
        ):
            if segment_intersects_sphere(
                pregrasp_position,
                collision_end,
                fruit["center"],
                fruit["radius"],
                NEIGHBOR_CLEARANCE,
            ):
                blocked = True
                blocking_object = (
                    f"NeighborFruit_{fruit_index}"
                )
                break

    # --------------------------------------------------------
    # 如果几何路径没有被挡住，则运行 IK
    # --------------------------------------------------------

    if blocked:
        marker_color = RED
        status = f"BLOCKED by {blocking_object}"

    else:
        target_orientation = make_grasp_orientation(
            approach_direction
        )

        action, ik_success = (
            ik_solver.compute_inverse_kinematics(
                target_position=pregrasp_position,
                target_orientation=target_orientation,
            )
        )

        if ik_success:
            marker_color = BLUE
            status = "IK SUCCESS"

            reachable_candidates.append(
                {
                    "index": candidate_index,
                    "angle_degrees": angle_degrees,
                    "position": (
                        pregrasp_position.copy()
                    ),
                    "orientation": (
                        target_orientation.copy()
                    ),
                    "joint_positions": (
                        action.joint_positions.copy()
                    ),
                }
            )

        else:
            marker_color = YELLOW
            status = "IK FAILED"

    # --------------------------------------------------------
    # 显示候选点和接近方向
    # --------------------------------------------------------

    add_candidate_marker(
        world,
        candidate_index,
        pregrasp_position,
        marker_color,
    )

    add_approach_line(
        world,
        candidate_index,
        pregrasp_position,
        FRUIT_POSITION,
        marker_color,
    )

    print(
        f"Candidate {candidate_index:02d} | "
        f"angle={angle_degrees:>4} deg | "
        f"position={np.round(pregrasp_position, 3)} | "
        f"{status}"
    )


# ============================================================
# 输出最佳候选
# ============================================================

print("=" * 76)

if len(reachable_candidates) == 0:
    print("没有找到蓝色候选位姿。")
    print("检查上面的输出：")
    print("如果是黄色，说明需要调整夹爪朝向或预抓取距离。")
    print("如果是红色，说明需要调整障碍物过滤参数。")

else:
    best_candidate = reachable_candidates[0]

    print(
        f"找到 {len(reachable_candidates)} "
        "个蓝色可达候选位姿。"
    )

    print()
    print("暂定最佳候选：")
    print(
        f"  编号：Candidate "
        f"{best_candidate['index']:02d}"
    )
    print(
        f"  角度："
        f"{best_candidate['angle_degrees']} deg"
    )
    print(
        "  位置：",
        np.round(
            best_candidate["position"],
            4,
        ),
    )
    print(
        "  四元数：",
        np.round(
            best_candidate["orientation"],
            4,
        ),
    )
    print(
        "  关节角：",
        np.round(
            best_candidate["joint_positions"],
            4,
        ),
    )

    print()
    print("当前只完成候选位姿和 IK 筛选。")
    print("机械臂暂时不会运动。")
    print("下一步将为最佳候选生成无碰撞运动路径。")

print("=" * 76)


# ============================================================
# 保持 Isaac Sim 窗口运行
# ============================================================

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()

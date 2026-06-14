"""
飞行任务管理器
支持多种飞行任务模式：穿越门框、定点降落、环绕飞行
"""
import numpy as np
import time
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


class MissionType(Enum):
    """任务类型"""
    GATE_PASS = "gate_pass"           # 穿越门框
    PRECISION_LAND = "precision_land"  # 定点降落
    ORBIT = "orbit"                   # 环绕飞行


class MissionStatus(Enum):
    """任务状态"""
    IDLE = "idle"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Gate:
    """穿越门框"""
    position: np.ndarray          # 门框中心位置 (x, y, z)
    rotation: float = 0.0         # 门框朝向角度（弧度）
    width: float = 2.0            # 门框宽度
    height: float = 2.0           # 门框高度
    passed: bool = False          # 是否已穿过
    color: Tuple[float, float, float] = (1.0, 0.8, 0.0)  # 金色


@dataclass
class LandingPad:
    """降落平台"""
    position: np.ndarray          # 平台中心位置 (x, y, z)
    radius: float = 1.0           # 平台半径
    landed: bool = False          # 是否已降落
    color: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # 绿色


@dataclass
class OrbitTarget:
    """环绕飞行目标点"""
    position: np.ndarray          # 目标位置 (x, y, z)
    radius: float = 5.0           # 环绕半径
    total_laps: int = 2           # 需要环绕的圈数
    completed_laps: float = 0.0   # 已完成的圈数
    color: Tuple[float, float, float] = (1.0, 0.5, 0.0)  # 橙色


@dataclass
class MissionResult:
    """任务结果"""
    mission_type: MissionType
    status: MissionStatus = MissionStatus.IDLE
    score: float = 0.0            # 总分 (0-100)
    time_elapsed: float = 0.0     # 用时（秒）
    accuracy: float = 0.0         # 精度得分
    collision_count: int = 0      # 碰撞次数
    details: Dict = field(default_factory=dict)
    completed_at: Optional[float] = None


class MissionManager:
    """飞行任务管理器"""

    def __init__(self):
        self.current_mission: Optional[MissionType] = None
        self.status: MissionStatus = MissionStatus.IDLE

        # 任务元素
        self.gates: List[Gate] = []
        self.landing_pad: Optional[LandingPad] = None
        self.orbit_target: Optional[OrbitTarget] = None

        # 任务计时
        self.mission_start_time: float = 0.0
        self.mission_time_limit: float = 120.0  # 默认120秒超时

        # 当前任务结果
        self.result: Optional[MissionResult] = None

        # 历史记录
        self.history: List[MissionResult] = []

        # 环绕飞行追踪
        self._orbit_angle_prev: float = 0.0
        self._orbit_angle_total: float = 0.0

        # 碰撞检测
        self.collision_threshold: float = 0.3  # 碰撞距离阈值

        print("[OK] 飞行任务管理器初始化完成")

    # ============ 任务生命周期 ============

    def start_mission(self, mission_type: MissionType) -> bool:
        """开始新任务"""
        if self.status == MissionStatus.RUNNING:
            print("[WARNING] 已有任务运行中，请先结束当前任务")
            return False

        self.current_mission = mission_type
        self.status = MissionStatus.READY
        self._setup_mission(mission_type)
        self.mission_start_time = time.time()
        self.result = MissionResult(mission_type=mission_type)
        self.status = MissionStatus.RUNNING

        print(f"\n{'='*50}")
        print(f"  任务开始: {self._get_mission_name(mission_type)}")
        print(f"{'='*50}")
        print(self._get_mission_description(mission_type))
        print(f"  时间限制: {self.mission_time_limit:.0f}秒")
        return True

    def complete_mission(self) -> Optional[MissionResult]:
        """完成任务并评分"""
        if self.status != MissionStatus.RUNNING:
            return None

        elapsed = time.time() - self.mission_start_time
        self.result.time_elapsed = elapsed

        # 计算分数
        if self.current_mission == MissionType.GATE_PASS:
            self._score_gate_pass()
        elif self.current_mission == MissionType.PRECISION_LAND:
            self._score_precision_land()
        elif self.current_mission == MissionType.ORBIT:
            self._score_orbit()

        self.result.status = MissionStatus.COMPLETED
        self.result.completed_at = time.time()
        self.history.append(self.result)
        self.status = MissionStatus.COMPLETED

        self._print_result()
        self._save_result()
        return self.result

    def fail_mission(self, reason: str = "") -> Optional[MissionResult]:
        """任务失败"""
        if self.status != MissionStatus.RUNNING:
            return None

        elapsed = time.time() - self.mission_start_time
        self.result.time_elapsed = elapsed
        self.result.status = MissionStatus.FAILED
        self.result.details['fail_reason'] = reason
        self.result.score = 0
        self.result.completed_at = time.time()
        self.history.append(self.result)
        self.status = MissionStatus.FAILED

        print(f"\n[FAIL] 任务失败: {reason}")
        return self.result

    def cancel_mission(self):
        """取消任务"""
        self.status = MissionStatus.IDLE
        self.current_mission = None
        self.gates.clear()
        self.landing_pad = None
        self.orbit_target = None
        self.result = None
        print("\n[INFO] 任务已取消")

    def reset_all(self):
        """重置所有状态"""
        self.cancel_mission()
        self.history.clear()
        self._orbit_angle_prev = 0.0
        self._orbit_angle_total = 0.0
        print("[OK] 任务管理器已完全重置")

    # ============ 任务设置 ============

    def _setup_mission(self, mission_type: MissionType):
        """根据任务类型设置场景"""
        if mission_type == MissionType.GATE_PASS:
            self._setup_gate_pass()
        elif mission_type == MissionType.PRECISION_LAND:
            self._setup_precision_land()
        elif mission_type == MissionType.ORBIT:
            self._setup_orbit()

    def _setup_gate_pass(self):
        """设置穿越门框任务 - 生成3个门框"""
        self.gates = [
            Gate(
                position=np.array([0.0, 2.5, -8.0]),
                rotation=0.0,
                width=3.0,
                height=2.5,
                color=(1.0, 0.8, 0.0)  # 金色
            ),
            Gate(
                position=np.array([5.0, 3.0, -16.0]),
                rotation=np.radians(30),
                width=2.5,
                height=2.5,
                color=(0.0, 1.0, 0.8)  # 青绿色
            ),
            Gate(
                position=np.array([-3.0, 4.0, -24.0]),
                rotation=np.radians(-20),
                width=3.5,
                height=2.0,
                color=(1.0, 0.4, 0.4)  # 红色
            ),
        ]
        self.mission_time_limit = 90.0
        print(f"  生成了 {len(self.gates)} 个穿越门框")

    def _setup_precision_land(self):
        """设置定点降落任务"""
        self.landing_pad = LandingPad(
            position=np.array([3.0, 0.05, -10.0]),
            radius=1.5,
            color=(0.0, 1.0, 0.0)
        )
        self.mission_time_limit = 60.0
        print(f"  降落平台位置: ({self.landing_pad.position[0]:.1f}, "
              f"{self.landing_pad.position[2]:.1f})")

    def _setup_orbit(self):
        """设置环绕飞行任务"""
        self.orbit_target = OrbitTarget(
            position=np.array([0.0, 3.0, -8.0]),
            radius=5.0,
            total_laps=2,
            color=(1.0, 0.5, 0.0)
        )
        self.mission_time_limit = 120.0
        self._orbit_angle_prev = 0.0
        self._orbit_angle_total = 0.0
        print(f"  环绕目标: ({self.orbit_target.position[0]:.1f}, "
              f"{self.orbit_target.position[2]:.1f}), "
              f"半径{self.orbit_target.radius:.1f}m, "
              f"{self.orbit_target.total_laps}圈")

    # ============ 物理更新 & 碰撞检测 ============

    def update(self, drone_position: np.ndarray, dt: float):
        """每帧更新 - 检测任务进度"""
        if self.status != MissionStatus.RUNNING:
            return

        # 超时检测
        elapsed = time.time() - self.mission_start_time
        if elapsed > self.mission_time_limit:
            self.fail_mission("任务超时")
            return

        if self.current_mission == MissionType.GATE_PASS:
            self._update_gate_pass(drone_position)
        elif self.current_mission == MissionType.PRECISION_LAND:
            self._update_precision_land(drone_position)
        elif self.current_mission == MissionType.ORBIT:
            self._update_orbit(drone_position)

    def _update_gate_pass(self, drone_position: np.ndarray):
        """检测穿越门框"""
        for gate in self.gates:
            if gate.passed:
                continue

            # 计算无人机到门框平面的距离
            gate_pos = gate.position
            gate_normal = np.array([-np.sin(gate.rotation), 0.0, -np.cos(gate.rotation)])
            dist_to_plane = abs(np.dot(drone_position - gate_pos, gate_normal))

            # 检查是否在门框范围内
            in_width = abs(drone_position[0] - gate_pos[0]) < gate.width / 2
            in_height = abs(drone_position[1] - gate_pos[1]) < gate.height / 2
            close_enough = dist_to_plane < 0.8  # 距离门框平面足够近

            if in_width and in_height and close_enough:
                gate.passed = True
                passed_count = sum(1 for g in self.gates if g.passed)
                print(f"\n[OK] 穿过门框 {passed_count}/{len(self.gates)}!")
                print(f"  位置偏差: X={abs(drone_position[0]-gate_pos[0]):.2f}m, "
                      f"Y={abs(drone_position[1]-gate_pos[1]):.2f}m")

                # 检测所有门是否都已通过
                if all(g.passed for g in self.gates):
                    self.complete_mission()

    def _update_precision_land(self, drone_position: np.ndarray):
        """检测定点降落"""
        if self.landing_pad is None or self.landing_pad.landed:
            return

        pad_pos = self.landing_pad.position
        horizontal_dist = np.sqrt(
            (drone_position[0] - pad_pos[0])**2 +
            (drone_position[2] - pad_pos[2])**2
        )
        vertical_dist = abs(drone_position[1] - pad_pos[1])

        # 检测是否降落在平台上
        if horizontal_dist < self.landing_pad.radius and vertical_dist < 0.2:
            self.landing_pad.landed = True
            self.result.details['landing_deviation'] = horizontal_dist
            self.result.details['vertical_error'] = vertical_dist
            self.complete_mission()

    def _update_orbit(self, drone_position: np.ndarray):
        """更新环绕飞行进度"""
        if self.orbit_target is None:
            return

        target_pos = self.orbit_target.position
        dx = drone_position[0] - target_pos[0]
        dz = drone_position[2] - target_pos[2]

        # 计算当前角度
        current_angle = np.arctan2(dz, dx)

        # 计算角度变化
        angle_diff = current_angle - self._orbit_angle_prev
        # 处理角度跨越 -pi 到 pi 的跳变
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        self._orbit_angle_total += angle_diff
        self._orbit_angle_prev = current_angle

        # 计算已完成的圈数
        completed_laps = abs(self._orbit_angle_total) / (2 * np.pi)
        self.orbit_target.completed_laps = completed_laps

        # 检查是否完成
        if completed_laps >= self.orbit_target.total_laps:
            self.complete_mission()

    def check_collision(self, drone_position: np.ndarray) -> bool:
        """检查无人机是否与任务元素碰撞（门框边缘等）"""
        if self.current_mission == MissionType.GATE_PASS:
            for gate in self.gates:
                if gate.passed:
                    continue
                gate_pos = gate.position
                # 检查是否撞到门框边缘
                dist = np.linalg.norm(drone_position - gate_pos)
                in_width = abs(drone_position[0] - gate_pos[0]) < gate.width / 2
                in_height = abs(drone_position[1] - gate_pos[1]) < gate.height / 2
                # 距离门框平面很近但不在开口内
                plane_dist = abs(drone_position[2] - gate_pos[2])
                if plane_dist < 0.3 and not (in_width and in_height):
                    return True
        return False

    # ============ 评分系统 ============

    def _score_gate_pass(self):
        """穿越门框评分"""
        total_gates = len(self.gates)
        passed_gates = sum(1 for g in self.gates if g.passed)

        # 完成度得分 (60%)
        completion_score = (passed_gates / total_gates) * 60

        # 时间得分 (30%) - 越快越好
        time_score = max(0, 30 - (self.result.time_elapsed / self.mission_time_limit) * 30)

        # 精度得分 (10%)
        accuracy_score = 10  # 基础精度分

        self.result.score = completion_score + time_score + accuracy_score
        self.result.accuracy = accuracy_score
        self.result.details['gates_passed'] = passed_gates
        self.result.details['total_gates'] = total_gates

    def _score_precision_land(self):
        """定点降落评分"""
        deviation = self.result.details.get('landing_deviation', 999)
        pad_radius = self.landing_pad.radius if self.landing_pad else 1.0

        # 精度得分 (50%) - 越靠近中心越高
        accuracy_ratio = max(0, 1 - deviation / pad_radius)
        accuracy_score = accuracy_ratio * 50

        # 时间得分 (30%)
        time_score = max(0, 30 - (self.result.time_elapsed / self.mission_time_limit) * 30)

        # 降落质量 (20%) - 垂直误差越小越好
        vert_error = self.result.details.get('vertical_error', 1.0)
        quality_score = max(0, 20 - vert_error * 100)

        self.result.score = accuracy_score + time_score + quality_score
        self.result.accuracy = accuracy_score
        self.result.details['accuracy_ratio'] = accuracy_ratio

    def _score_orbit(self):
        """环绕飞行评分"""
        completed = self.orbit_target.completed_laps if self.orbit_target else 0
        total = self.orbit_target.total_laps if self.orbit_target else 1

        # 完成度得分 (50%)
        completion_ratio = min(completed / total, 1.0)
        completion_score = completion_ratio * 50

        # 时间得分 (30%)
        time_score = max(0, 30 - (self.result.time_elapsed / self.mission_time_limit) * 30)

        # 轨迹质量 (20%) - 偏离环绕半径的程度
        quality_score = 20

        self.result.score = completion_score + time_score + quality_score
        self.result.accuracy = quality_score
        self.result.details['completed_laps'] = completed
        self.result.details['total_laps'] = total

    def get_score_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return "S (完美)"
        elif score >= 80:
            return "A (优秀)"
        elif score >= 70:
            return "B (良好)"
        elif score >= 60:
            return "C (合格)"
        else:
            return "D (需改进)"

    # ============ 信息获取 ============

    def get_mission_info(self) -> Dict:
        """获取当前任务信息（用于UI显示）"""
        info = {
            'mission_type': self.current_mission.value if self.current_mission else 'none',
            'mission_name': self._get_mission_name(self.current_mission) if self.current_mission else '无',
            'status': self.status.value,
            'elapsed': 0.0,
            'time_limit': self.mission_time_limit,
            'score': 0.0,
            'grade': '',
            'progress': ''
        }

        if self.status == MissionStatus.RUNNING:
            info['elapsed'] = time.time() - self.mission_start_time

            if self.current_mission == MissionType.GATE_PASS:
                passed = sum(1 for g in self.gates if g.passed)
                info['progress'] = f"门框: {passed}/{len(self.gates)}"
            elif self.current_mission == MissionType.PRECISION_LAND:
                info['progress'] = "降落中..."
            elif self.current_mission == MissionType.ORBIT:
                laps = self.orbit_target.completed_laps if self.orbit_target else 0
                total = self.orbit_target.total_laps if self.orbit_target else 0
                info['progress'] = f"圈数: {laps:.1f}/{total}"

        if self.result and self.result.status in [MissionStatus.COMPLETED, MissionStatus.FAILED]:
            info['score'] = self.result.score
            info['grade'] = self.get_score_grade(self.result.score)

        return info

    def get_gates(self) -> List[Gate]:
        """获取门框列表（供3D渲染使用）"""
        return self.gates

    def get_landing_pad(self) -> Optional[LandingPad]:
        """获取降落平台（供3D渲染使用）"""
        return self.landing_pad

    def get_orbit_target(self) -> Optional[OrbitTarget]:
        """获取环绕目标（供3D渲染使用）"""
        return self.orbit_target

    def _get_mission_name(self, mission_type: Optional[MissionType]) -> str:
        if mission_type is None:
            return "无"
        names = {
            MissionType.GATE_PASS: "穿越门框",
            MissionType.PRECISION_LAND: "定点降落",
            MissionType.ORBIT: "环绕飞行",
        }
        return names.get(mission_type, "未知")

    def _get_mission_description(self, mission_type: MissionType) -> str:
        descriptions = {
            MissionType.GATE_PASS: (
                "  任务目标: 操控无人机依次穿越3个门框\n"
                "  提示: 注意调整高度和方向，从门框中间穿过"
            ),
            MissionType.PRECISION_LAND: (
                "  任务目标: 操控无人机精准降落到绿色平台上\n"
                "  提示: 缓慢降低高度，尽量降落在平台中心"
            ),
            MissionType.ORBIT: (
                "  任务目标: 操控无人机围绕橙色目标点飞行指定圈数\n"
                "  提示: 保持与目标点的距离，匀速环绕飞行"
            ),
        }
        return descriptions.get(mission_type, "")

    def _print_result(self):
        """打印任务结果"""
        if self.result is None:
            return
        grade = self.get_score_grade(self.result.score)
        print(f"\n{'='*50}")
        print(f"  任务完成! {self._get_mission_name(self.current_mission)}")
        print(f"{'='*50}")
        print(f"  状态: {'成功' if self.result.status == MissionStatus.COMPLETED else '失败'}")
        print(f"  用时: {self.result.time_elapsed:.1f}秒")
        print(f"  得分: {self.result.score:.1f}/100")
        print(f"  评级: {grade}")
        print(f"  精度: {self.result.accuracy:.1f}分")
        print(f"  碰撞: {self.result.collision_count}次")
        print(f"{'='*50}\n")

    def _save_result(self):
        """保存任务结果到文件"""
        if self.result is None:
            return
        try:
            history_data = []
            for r in self.history:
                history_data.append({
                    'mission_type': r.mission_type.value,
                    'status': r.status.value,
                    'score': r.score,
                    'time_elapsed': r.time_elapsed,
                    'accuracy': r.accuracy,
                    'collisions': r.collision_count,
                    'details': r.details,
                })

            with open('mission_history.json', 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] 保存任务记录失败: {e}")

    def get_history_summary(self) -> str:
        """获取历史记录摘要"""
        if not self.history:
            return "暂无任务记录"

        lines = ["\n=== 任务历史记录 ==="]
        for i, r in enumerate(self.history, 1):
            name = self._get_mission_name(r.mission_type)
            grade = self.get_score_grade(r.score)
            status = "成功" if r.status == MissionStatus.COMPLETED else "失败"
            lines.append(
                f"  {i}. [{name}] {status} | "
                f"得分: {r.score:.0f} | 用时: {r.time_elapsed:.1f}s | 评级: {grade}"
            )
        return "\n".join(lines)

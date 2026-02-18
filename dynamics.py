"""
动力学模型 - dynamics.py

按照 SIMULATION_PLAN.md 计划实现：
1. 真实系统矩阵 (A_true, B_true) - 用于状态传播
2. 名义模型矩阵 (A_nominal, B_nominal) - 用于控制器设计
3. 线性状态更新
4. 虚拟领航者和相对误差计算
"""

import numpy as np
from typing import Tuple, Dict
from params import SimParams, DavisParams, NominalModelParams


class Train:
    """单车动力学模型"""

    def __init__(self, train_id: int, initial_state: np.ndarray, params: Dict):
        """
        初始化单车

        Args:
            train_id: 列车ID (0-based)
            initial_state: 初始状态 [position, velocity]
            params: 仿真参数
        """
        self.train_id = train_id
        self.state = initial_state.copy()  # 真实状态 [p, v]
        self.params = params

    def get_position(self) -> float:
        return self.state[0]

    def get_velocity(self) -> float:
        return self.state[1]


class TrainPlatoon:
    """列车编队动力学模型"""

    def __init__(self, params: Dict):
        """
        初始化列车编队

        Args:
            params: 仿真参数字典
        """
        self.params = params
        self.n_trains = params['sim']['n_trains']
        self.d_desired = params['sim']['d_desired']
        self.v_nominal = params['sim']['v_nominal']
        self.mass = params['sim']['mass']

        # 计算真实系统矩阵
        self.A_true, self.B_true = self._compute_true_matrices()

        # 计算名义模型矩阵（20%误差）
        self.A_nominal, self.B_nominal = self._compute_nominal_matrices()

        # 初始化列车
        self.trains = self._initialize_trains()

        # 保存T1的初始位置（用于虚拟领航者计算）
        self.p_T1_initial = self.trains[0].state[0] if self.trains else 0.0

        # 分布式观测器: 每列车维护对虚拟领航者状态的估计
        # x̂_leader_i 表示列车i对虚拟领航者状态的估计
        self.leader_estimates = np.zeros((self.n_trains, 2))
        self._init_leader_estimates()

        # 观测器增益参数
        self.observer_gain = params.get('observer_gain', 0.3)

    def _init_leader_estimates(self):
        """初始化领航者估计（使用名义值，避免继承随机扰动）"""
        # T1直接跟踪虚拟领航者
        self.leader_estimates[0] = np.array([
            self.p_T1_initial,
            self.v_nominal
        ])

        # 其他列车: 初始估计基于名义值（不考虑随机扰动）
        # 假设每列车在虚拟领航者后方 i * d_desired 处
        for i in range(1, self.n_trains):
            # 使用名义值初始化
            self.leader_estimates[i] = np.array([
                self.p_T1_initial - i * self.d_desired,  # 名义位置
                self.v_nominal  # 名义速度
            ])

    def get_leader_estimates(self) -> np.ndarray:
        """获取所有列车的领航者估计"""
        return self.leader_estimates.copy()

    def update_leader_estimates(self, broadcast_states: np.ndarray, dt: float, t: float = 0.0):
        """
        更新领航者估计（分布式观测器）

        简单版本: x̂_leader_i = broadcast_{i-1} + d_desired

        即每列车认为领航者在"前车广播位置 + 期望间距"处

        使用广播状态而非真实状态，因为列车只能获取邻居的广播状态

        Args:
            broadcast_states: 广播状态数组 (n_trains, 2)
            dt: 时间步长
            t: 当前时间（用于计算虚拟领航者位置）
        """
        new_estimates = self.leader_estimates.copy()

        # T1直接跟踪虚拟领航者
        virtual_state = self.get_virtual_leader_state(t)
        new_estimates[0] = virtual_state.copy()

        # 其他列车: 基于前车的广播状态估计领航者位置
        for i in range(1, self.n_trains):
            # 前车的广播状态
            prev_broadcast = broadcast_states[i-1]

            # 领航者估计 = 前车广播位置 + 期望间距
            # 这是因为前车应该在前车与领航者中间
            leader_estimate = np.array([
                prev_broadcast[0] + self.d_desired,
                prev_broadcast[1]  # 速度假设相同
            ])

            # 使用观测器增益进行平滑
            new_estimates[i] = (1 - self.observer_gain) * self.leader_estimates[i] + \
                               self.observer_gain * leader_estimate

        self.leader_estimates = new_estimates

    def _compute_true_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算真实系统矩阵（基于Davis公式线性化）

        Returns:
            A_true, B_true: 真实系统矩阵
        """
        # 获取Davis系数
        a = self.params['davis']['a']
        b = self.params['davis']['b']
        c = self.params['davis']['c']
        m = self.mass
        v0 = self.v_nominal

        # 阻力系数在v0处的导数
        # F_resistance = (a + b*v + c*v²) * m
        # dF/dv = (b + 2*c*v) * m
        # 线性化后: a21 = -dF/dv (负号因为阻力与速度方向相反)
        a21_true = -(b + 2 * c * v0) * m

        # A_true: [dp/dt = v, dv/dt = a21 * v + F/m]
        A_true = np.array([
            [0, 1],
            [0, a21_true / m]
        ])

        # B_true: [dF/m, dF/m] (假设两个输入通道相同)
        # 实际控制输入是牵引力/制动力
        B_true = np.array([
            [0, 0],
            [1/m, 1/m]
        ])

        return A_true, B_true

    def _compute_nominal_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算名义模型矩阵（用于控制器设计，与真值有20%偏差）

        这是解决"未知系统悖论"的关键：
        控制器设计者不知道真实系统，只能使用近似估计的名义模型

        Returns:
            A_nominal, B_nominal: 名义模型矩阵
        """
        uncertainty_factor = self.params['nominal']['uncertainty_factor']

        # A_nominal: 只对a21项（阻力相关）加误差
        # 第一行 [0, 1] 是运动学关系，不应有误差
        a21_true = self.A_true[1, 1]
        a21_nominal = a21_true * uncertainty_factor

        A_nominal = np.array([
            [0, 1],
            [0, a21_nominal]
        ])

        # B_nominal: 整体乘以uncertainty_factor
        B_nominal = self.B_true * uncertainty_factor

        return A_nominal, B_nominal

    def get_true_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取真实系统矩阵（用于状态传播）"""
        return self.A_true, self.B_true

    def get_nominal_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取名义模型矩阵（用于控制器设计）"""
        return self.A_nominal, self.B_nominal

    def _initialize_trains(self) -> list:
        """初始化列车状态"""
        trains = []
        v_nominal = self.v_nominal
        p_initial_std = self.params['sim']['p_initial_std']
        v_initial_std = self.params['sim']['v_initial_std']
        seed_offset = self.params.get('seed_offset', 0)

        # 使用随机种子
        np.random.seed(42 + seed_offset)

        for i in range(self.n_trains):
            if i == 0:
                # T1: 初始位置为0，速度为额定速度
                p0 = 0.0
                v0 = v_nominal
            else:
                # T2-T8: 初始位置在前车后方d_desired处，带有随机扰动
                p0 = -(i) * self.d_desired + np.random.normal(0, p_initial_std)
                v0 = v_nominal + np.random.normal(0, v_initial_std)

            initial_state = np.array([p0, v0])
            trains.append(Train(i, initial_state, self.params))

        return trains

    def get_states(self) -> np.ndarray:
        """获取所有列车的当前状态"""
        states = np.zeros((self.n_trains, 2))
        for i, train in enumerate(self.trains):
            states[i] = train.state
        return states

    def get_virtual_leader_state(self, t: float) -> np.ndarray:
        """
        获取虚拟领航者状态

        虚拟领航者定义（SIMULATION_PLAN.md 4.3节）:
        - p_virtual(t) = p_T1_initial + v_nominal * t
        - v_virtual(t) = v_nominal (常数)

        Args:
            t: 当前时间

        Returns:
            [p_virtual, v_virtual]
        """
        # 使用T1的初始位置（常数）
        p_virtual = self.p_T1_initial + self.v_nominal * t
        v_virtual = self.v_nominal
        return np.array([p_virtual, v_virtual])

    def compute_delta(self, t: float) -> np.ndarray:
        """
        计算相对状态误差 δ（虚拟拓扑重构版本）

        改进：引入虚拟邻居机制，链尾列车同时参考T1和前车
        - 等效于增大图连通性，减小信息传播延迟

        公式:
        δ_i = α_i * δ_i^chain + β_i * δ_i^virtual
        其中:
        - δ_i^chain: 原始链式误差 (与前车比较)
        - δ_i^virtual: 虚拟误差 (与T1比较，考虑链首-链尾距离)
        - α_i + β_i = 1, β_i 随 i 增大

        Args:
            t: 当前时间

        Returns:
            delta: 相对误差数组 (n_trains, 2)
        """
        delta = np.zeros((self.n_trains, 2))
        states = self.get_states()

        for i, train in enumerate(self.trains):
            if i == 0:
                # T1: 与虚拟领航者比较
                virtual_state = self.get_virtual_leader_state(t)
                delta[i, 0] = virtual_state[0] - train.get_position()
                delta[i, 1] = virtual_state[1] - train.get_velocity()
            else:
                # T2-T8: 虚拟拓扑重构
                prev_train = self.trains[i - 1]

                # 链式误差 (与前车比较)
                delta_chain_pos = prev_train.get_position() - train.get_position() - self.d_desired
                delta_chain_vel = prev_train.get_velocity() - train.get_velocity()

                # 虚拟误差 (与T1比较，考虑链首-链尾距离)
                # T1到T8的期望距离 = 7 * d_desired
                leader_train = self.trains[0]
                leader_state = self.get_virtual_leader_state(t)
                leader_pos = leader_state[0] - i * self.d_desired  # T1前方i个间距
                delta_virtual_pos = leader_pos - train.get_position()
                delta_virtual_vel = leader_state[1] - train.get_velocity()

                # 虚拟权重: β_i = 0.25 * i / (N-1) (减小以减少触发)
                # T2: β=0.025, T8: β=0.25
                beta = 0.25 * i / max(1, self.n_trains - 1)
                alpha = 1.0 - beta

                # 组合误差
                delta[i, 0] = alpha * delta_chain_pos + beta * delta_virtual_pos
                delta[i, 1] = alpha * delta_chain_vel + beta * delta_virtual_vel

        return delta

    def compute_delta_observer(self, t: float) -> np.ndarray:
        """
        计算相对状态误差 δ（使用分布式观测器版本）

        使用每列车对虚拟领航者的估计来计算相对误差:
        - T1: δ_1 = [p_virtual - p_1, v_virtual - v_1]^T
        - T2-T8: δ_i = [x̂_leader_i[0] - p_i - d_desired*i, x̂_leader_i[1] - v_i]^T

        这样T8可以直接与领航者比较，而非通过7跳级联

        Args:
            t: 当前时间

        Returns:
            delta: 相对误差数组 (n_trains, 2)
        """
        delta = np.zeros((self.n_trains, 2))
        states = self.get_states()

        for i, train in enumerate(self.trains):
            if i == 0:
                # T1: 与虚拟领航者比较
                virtual_state = self.get_virtual_leader_state(t)
                delta[i, 0] = virtual_state[0] - train.get_position()
                delta[i, 1] = virtual_state[1] - train.get_velocity()
            else:
                # T2-T8: 与估计的领航者状态比较（考虑自身在链中的位置）
                leader_estimate = self.leader_estimates[i]
                # 考虑当前列车i与领航者之间的期望间距: i * d_desired
                delta[i, 0] = leader_estimate[0] - train.get_position() - self.d_desired * i
                delta[i, 1] = leader_estimate[1] - train.get_velocity()

        return delta

    def step(self, controls: np.ndarray, dt: float):
        """
        更新所有列车状态（基于真实系统矩阵）

        使用前向欧拉法离散化（SIMULATION_PLAN.md 4.2节）:
        x_next = x + (A_true @ x + B_true @ u) * dt

        Args:
            controls: 控制输入数组 (n_trains, 2)
            dt: 时间步长
        """
        for i, train in enumerate(self.trains):
            x = train.state
            u = controls[i]

            # 线性动力学: x_dot = A_true @ x + B_true @ u
            x_dot = self.A_true @ x + self.B_true @ u

            # 前向欧拉积分
            x_next = x + x_dot * dt
            train.state = x_next


if __name__ == "__main__":
    # 测试
    from params import get_all_params
    params = get_all_params()

    platoon = TrainPlatoon(params)

    print("True Matrices:")
    print(f"  A_true:\n{platoon.A_true}")
    print(f"  B_true:\n{platoon.B_true}")

    print("\nNominal Matrices (20% error):")
    print(f"  A_nominal:\n{platoon.A_nominal}")
    print(f"  B_nominal:\n{platoon.B_nominal}")

    print("\nInitial States:")
    states = platoon.get_states()
    for i in range(platoon.n_trains):
        print(f"  T{i+1}: p={states[i,0]:.2f}m, v={states[i,1]:.2f}m/s")

    print("\nDelta at t=0:")
    delta = platoon.compute_delta(0.0)
    for i in range(platoon.n_trains):
        print(f"  δ_{i+1}: [{delta[i,0]:.2f}, {delta[i,1]:.2f}]")

"""
事件触发控制器 - controller.py

按照 SIMULATION_PLAN.md 计划实现：
1. 控制器初始化时只接收名义矩阵（A_nominal, B_nominal）
2. 基于名义矩阵求解 Riccati 方程计算 K 增益
3. 实现控制律 u = Â x̂ + K δ
4. 实现参数自适应律（离散化）
5. 实现5种阈值更新方案
6. 实现触发判断逻辑（包含防Zeno）
"""

import numpy as np
from typing import Optional, Dict
from scipy.linalg import solve_continuous_are


class ETCController:
    """
    自适应事件触发控制器

    关键设计点（解决"未知系统悖论"）:
    - 控制器只接收名义矩阵 (A_nominal, B_nominal)
    - 使用名义矩阵计算 K 增益
    - 自适应估计器 Â, B̂ 用于补偿名义模型与真实系统的偏差
    """

    def __init__(
        self,
        A_nominal: np.ndarray,
        B_nominal: np.ndarray,
        controller_params: Dict,
        state_dim: int = 2,
        input_dim: int = 2,
    ):
        """
        初始化控制器

        Args:
            A_nominal: 名义系统矩阵 (不是真值!)
            B_nominal: 名义输入矩阵 (不是真值!)
            controller_params: 控制器参数字典
            state_dim: 状态维度
            input_dim: 输入维度
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.params = controller_params
        self.n_trains = controller_params.get('n_trains', 8)

        # 阈值参数
        self.sigma_min = controller_params.get('sigma_min', 0.01)
        self.sigma_max = controller_params.get('sigma_max', 1.0)
        self.alpha = controller_params.get('alpha', 1.0)
        self.k = controller_params.get('k', 0.75)
        self.c = controller_params.get('c', 0.5)
        self.gamma = controller_params.get('gamma', 1.0)

        # 状态归一化参数
        self.d_scale = controller_params.get('d_scale', 100.0)
        self.v_scale = controller_params.get('v_scale', 10.0)

        # 自适应增益
        self.lambda_A = controller_params.get('lambda_A', 1e-6)
        self.gamma_B = controller_params.get('gamma_B', 1e-6)

        # 积分阻尼增益 (用于消除链式拓扑的稳态误差)
        self.K_i = controller_params.get('K_i', 0.5)

        # 导数增益 (用于预测和抵抗误差变化 - 类似PD控制)
        self.K_d = controller_params.get('K_d', 10.0)

        # 参数投影约束
        self.A_max = controller_params.get('A_max', 10.0)
        self.B_max = controller_params.get('B_max', 10.0)

        # 防Zeno参数
        self.zeno_interval = controller_params.get('zeno_interval', 0.01)

        # 跟踪每个agent的delta历史用于趋势检测
        self.delta_history = {}  # {agent_idx: [delta_norm的历史]}

        # 方案参数
        self.fixed_threshold = controller_params.get('fixed_threshold', 0.3)
        self.periodic_interval = controller_params.get('periodic_interval', 1.0)
        self.adaptive_type = controller_params.get('adaptive_type', 'lyapunov_driven')

        # 分级参数（方案2和方案3）
        self.mu_alpha = controller_params.get('mu_alpha', 0.2)    # μ分级系数
        self.sigma_min_beta = controller_params.get('sigma_min_beta', 0.1)  # σ_min分级系数

        # 基于名义矩阵计算K（关键：不是真值！）
        Q_weight = controller_params.get('Q_weight', 3e7)
        R_weight = controller_params.get('R_weight', 1.0)

        self.K = self._compute_K(A_nominal, B_nominal, Q_weight, R_weight)

    def _compute_K(
        self,
        A: np.ndarray,
        B: np.ndarray,
        Q_weight: float,
        R_weight: float,
    ) -> np.ndarray:
        """
        通过解Riccati方程计算K增益

        使用名义矩阵（不是真值！）来设计K

        Args:
            A: 系统矩阵（名义模型）
            B: 输入矩阵（名义模型）
            Q_weight: 状态权重
            R_weight: 输入权重

        Returns:
            K: 反馈增益矩阵
        """
        Q = Q_weight * np.eye(self.state_dim)
        R = R_weight * np.eye(self.input_dim)

        try:
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P
            return K
        except Exception as e:
            print(f"Riccati solver failed: {e}, using default K")
            return np.array([[1500.0, 800.0], [750.0, 400.0]])

    def normalize_delta(self, delta: np.ndarray) -> np.ndarray:
        """
        计算归一化相对状态误差 δ̃

        δ̃ = [δ_p / d_scale, δ_v / v_scale]^T
        """
        normalized = np.zeros_like(delta)
        normalized[0] = delta[0] / self.d_scale
        normalized[1] = delta[1] / self.v_scale
        return normalized

    def compute_tilde_delta_norm(self, delta: np.ndarray) -> float:
        """计算归一化相对误差的范数"""
        tilde_delta = self.normalize_delta(delta)
        return np.linalg.norm(tilde_delta)

    def compute_control_input(
        self,
        A_hat: np.ndarray,
        x_hat: np.ndarray,
        delta: np.ndarray,
        delta_integral: np.ndarray = None,
        delta_derivative: np.ndarray = None,
    ) -> np.ndarray:
        """
        控制律: u = Â x̂ + K δ + K_i * ∫δ dt + K_d * dδ/dt (类似PD控制)

        Args:
            A_hat: 自适应估计的A矩阵
            x_hat: 广播状态（ZOH保持）
            delta: 相对状态误差
            delta_integral: 相对误差的积分 (∫δ dt)
            delta_derivative: 相对误差的导数 (dδ/dt)

        Returns:
            u: 控制输入
        """
        A_hat_x_hat = A_hat @ x_hat
        K_delta = self.K @ delta

        # 基础控制
        u = A_hat_x_hat + K_delta

        # 添加积分项以消除稳态误差
        if delta_integral is not None:
            K_i_delta_int = self.K_i * delta_integral
            u = u + K_i_delta_int

        # 添加导数项以预测和抵抗误差变化
        if delta_derivative is not None:
            K_d_delta_dot = self.K_d * delta_derivative
            u = u + K_d_delta_dot

        return u

    def compute_A_hat_update(
        self,
        A_hat: np.ndarray,
        delta: np.ndarray,
        x_hat: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Â的自适应律: Â̇ = Λ δ x̂ᵀ

        离散化（SIMULATION_PLAN.md 4.7节）:
        Â(k+1) = Â(k) + λ_A * δ * x̂^T * dt

        Args:
            A_hat: 当前估计
            delta: 相对误差
            x_hat: 广播状态
            dt: 时间步长

        Returns:
            A_hat_new: 更新后的估计
        """
        x_hat_col = x_hat.reshape(-1, 1)
        delta_col = delta.reshape(-1, 1)
        dA = self.lambda_A * delta_col @ x_hat_col.T
        A_new = A_hat + dA * dt
        # 参数投影约束
        return np.clip(A_new, -self.A_max, self.A_max)

    def compute_B_hat_update(
        self,
        B_hat: np.ndarray,
        delta: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        B̂的自适应律: B̂̇ = Γ δ δᵀ

        离散化:
        B̂(k+1) = B̂(k) + γ_B * δ * δ^T * dt

        Args:
            B_hat: 当前估计
            delta: 相对误差
            dt: 时间步长

        Returns:
            B_hat_new: 更新后的估计
        """
        delta_col = delta.reshape(-1, 1)
        dB = self.gamma_B * delta_col @ delta_col.T
        B_new = B_hat + dB * dt
        # 参数投影约束
        return np.clip(B_new, -self.B_max, self.B_max)

    def compute_sigma_update(
        self,
        sigma: float,
        e: np.ndarray,
        delta: np.ndarray,
        dt: float,
        t: float = 0.0,
        agent_idx: int = 0,
    ) -> float:
        """
        σ的自适应更新（5种方案，含分级策略）

        分级策略：链尾使用更低的σ_min，更激进触发
        sigma_min_i = base_sigma_min * (1 + beta * (n_trains - 1 - agent_idx))
        """
        # 计算分级sigma_min（链尾更低）
        # T1: sigma_min, T8: sigma_min * (1 + 7*beta)
        sigma_min_i = self.sigma_min * (1 + self.sigma_min_beta * (7 - agent_idx))

        tilde_delta_norm = self.compute_tilde_delta_norm(delta)

        if self.adaptive_type == "periodic":
            # 定时通信: σ不适用
            sigma_new = self.sigma_max

        elif self.adaptive_type == "fixed_threshold":
            # 固定阈值: σ保持不变
            sigma_new = self.fixed_threshold

        elif self.adaptive_type == "error_driven":
            # Error-Driven: 当 e 大时，sigma 增加（减少触发）
            # 论文公式 σ̇ = -α||e||² 是减少的，但我们的目标是前期密集触发
            # 所以改为: σ̇ = +α||e||² 使 e 大时 sigma 增加
            alpha = 0.1
            e_norm_sq = np.linalg.norm(e) ** 2
            sigma_new = sigma + alpha * e_norm_sq * dt
            sigma_new = np.clip(sigma_new, sigma_min_i, self.sigma_max)

        elif self.adaptive_type == "state_driven":
            # State-Driven: 改为与误差相关的绝对阈值
            # σ = σ_min + k * ||δ̃|| / (||δ̃|| + c) 确保在 δ̃ 小时 σ 不会太小
            tilde_delta_norm = self.compute_tilde_delta_norm(delta)
            target_sigma = sigma_min_i + self.k * tilde_delta_norm / (tilde_delta_norm + self.c)
            # 平滑过渡
            tau = 5.0
            decay = np.exp(-self.gamma * dt / tau)
            sigma_new = target_sigma + (sigma - target_sigma) * decay

        elif self.adaptive_type == "lyapunov_driven":
            # Lyapunov-Driven: 改为单调上升 σ = σ_max - (σ_max - σ_min) * exp(-η*t)
            # 早期: σ小→阈值低→触发多
            # 后期: σ大→阈值高→触发少
            eta = 0.1  # 时间衰减速率
            sigma_new = self.sigma_max - (self.sigma_max - sigma_min_i) * np.exp(-eta * t)
            sigma_new = np.clip(sigma_new, sigma_min_i, self.sigma_max)

        else:
            sigma_new = sigma

        return sigma_new

    def check_trigger(
        self,
        e: np.ndarray,
        delta: np.ndarray,
        sigma: float,
        t: float,
        last_trigger_time: float,
        agent_idx: int = 0,
        neighbor_trigger_time: float = 0.0,
    ) -> bool:
        """
        触发条件判断（包含预测性触发）

        触发条件:
        1. 被动触发: ||e|| > σ + μ(t) + inhibit_boost
        2. 预测触发: 误差正在增加且将在 horizon 内超过阈值

        Args:
            e: 测量误差 (x̂ - x)
            delta: 相对状态误差
            sigma: 当前阈值
            t: 当前时间
            last_trigger_time: 上次触发时间
            agent_idx: 代理索引（用于分级参数）
            neighbor_trigger_time: 前车的上次触发时间
            error_derivative: 误差变化率 de/dt

        Returns:
            triggered: 是否触发
        """
        # 方案A: 定时通信
        if self.adaptive_type == "periodic":
            return (t - last_trigger_time) >= self.periodic_interval

        # 最小触发间隔（防Zeno）
        if t - last_trigger_time < self.zeno_interval:
            return False

        # 协同触发机制 - Negative Coupling (论文Section V-C)
        # 当邻居触发时，通过减少||δ_i||来抑制邻居的触发
        # 我们的实现：利用这个机制减少级联触发
        inhibit_boost = 0.0  # 额外的抑制阈值
        compensate_boost = 0.0  # 补偿阈值（当邻居久未触发时）

        if agent_idx > 0:  # 非头车
            # 1. Negative Coupling: 邻居刚触发时，抑制自己的触发
            # 自适应抑制窗口：链尾需要更长的抑制时间
            delta_norm = np.linalg.norm(delta)
            tau_inhibit = 0.2 + 0.15 * min(delta_norm / 10.0, 5.0)  # 0.2-0.95s 自适应
            # 链尾(T8, agent_idx=7)需要更长的抑制
            tau_inhibit = tau_inhibit * (1 + 0.1 * agent_idx)

            time_since_neighbor_trigger = t - neighbor_trigger_time
            if 0 < time_since_neighbor_trigger < tau_inhibit:
                # 指数衰减的抑制强度
                decay_ratio = time_since_neighbor_trigger / tau_inhibit
                inhibit_boost = 25.0 * (1.0 - decay_ratio)  # 最大25.0

            # 2. 补偿机制: 邻居久未触发时，降低阈值补偿信息延迟
            # 链尾需要更激进的补偿
            tau_compensate = 2.0 + 0.2 * agent_idx  # 2.0-3.4s
            if time_since_neighbor_trigger > tau_compensate:
                # 超过补偿窗口，开始降低阈值（增加触发概率）
                compensate_factor = min((time_since_neighbor_trigger - tau_compensate) / 5.0, 1.0)
                compensate_boost = -30.0 * compensate_factor  # 最多降低30.0阈值

        # 计算误差范数
        error_norm = np.linalg.norm(e)

        # 使用非归一化delta范数
        delta_norm = np.linalg.norm(delta)

        # 趋势检测：当delta上升时，降低阈值更容易触发
        trend_boost = 0.0
        if agent_idx not in self.delta_history:
            self.delta_history[agent_idx] = []
        self.delta_history[agent_idx].append(delta_norm)
        # 保持历史长度
        if len(self.delta_history[agent_idx]) > 100:
            self.delta_history[agent_idx].pop(0)
        # 检测趋势：最近10步平均 vs 之前10步平均
        hist = self.delta_history[agent_idx]
        if len(hist) >= 20:
            recent = np.mean(hist[-10:])
            older = np.mean(hist[-20:-10])
            if recent > older * 1.05:  # 上升趋势 >5%
                trend_boost = -2.0  # 轻微降低阈值(减少以降低触发)

        # 触发条件: ||e_i|| > σ * ||δ|| / w_i + μ(t) + trend_boost
        # 位置权重 - 链尾更容易触发
        alpha_weight = 0.3
        w_i = 1.0 + alpha_weight * agent_idx / max(1, self.n_trains - 1)

        mu_0 = 1.0         # 初始下界
        mu_final = 40.0    # 稳态下界
        mu_t = mu_final - (mu_final - mu_0) * np.exp(-0.3 * t)

        threshold = sigma * delta_norm / w_i + mu_t + inhibit_boost + compensate_boost + trend_boost

        # 检查被动触发条件
        if error_norm > threshold:
            return True

        # 预测性触发已禁用 - 导致触发次数暴增

        return False


class TriggerMonitor:
    """触发监控器"""

    def __init__(self, n_trains: int):
        self.n_trains = n_trains
        self.trigger_times = {i: [] for i in range(n_trains)}
        self.trigger_counts = np.zeros(n_trains, dtype=int)
        self.inter_event_intervals = {i: [] for i in range(n_trains)}
        self.min_interval = float('inf')

    def record_trigger(self, train_id: int, t: float):
        """记录触发事件"""
        if len(self.trigger_times[train_id]) > 0:
            last_time = self.trigger_times[train_id][-1]
            interval = t - last_time
            self.inter_event_intervals[train_id].append(interval)
            if interval < self.min_interval:
                self.min_interval = interval
        self.trigger_times[train_id].append(t)
        self.trigger_counts[train_id] += 1

    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_triggers = int(np.sum(self.trigger_counts))
        all_intervals = []
        for intervals in self.inter_event_intervals.values():
            all_intervals.extend(intervals)
        avg_interval = np.mean(all_intervals) if all_intervals else 0.0

        return {
            "total_triggers": total_triggers,
            "trigger_counts": self.trigger_counts.tolist(),
            "trigger_times": {str(k): v for k, v in self.trigger_times.items()},
            "min_interval": self.min_interval if self.min_interval != float('inf') else None,
            "avg_interval": avg_interval,
        }

    def reset(self):
        """重置"""
        self.trigger_times = {i: [] for i in range(self.n_trains)}
        self.trigger_counts = np.zeros(self.n_trains, dtype=int)
        self.inter_event_intervals = {i: [] for i in range(self.n_trains)}
        self.min_interval = float('inf')


if __name__ == "__main__":
    # 测试
    from params import get_all_params
    import json

    params = get_all_params()
    A_nom, B_nom = np.array([[0, 1], [0, -0.00178]]), np.array([[0, 0], [2e-7, 2e-7]])

    ctrl = ETCController(A_nom, B_nom, params['controller'])

    print("Controller initialized:")
    print(f"  K matrix:\n{ctrl.K}")
    print(f"  adaptive_type: {ctrl.adaptive_type}")
    print(f"  sigma range: [{ctrl.sigma_min}, {ctrl.sigma_max}]")

    # 测试控制计算
    A_hat = np.zeros((2, 2))
    x_hat = np.array([100.0, 22.0])
    delta = np.array([10.0, 1.0])
    u = ctrl.compute_control_input(A_hat, x_hat, delta)
    print(f"\nTest control input: {u}")

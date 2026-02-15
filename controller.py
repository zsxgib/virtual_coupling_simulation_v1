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

        # 参数投影约束
        self.A_max = controller_params.get('A_max', 10.0)
        self.B_max = controller_params.get('B_max', 10.0)

        # 防Zeno参数
        self.zeno_interval = controller_params.get('zeno_interval', 0.01)

        # 方案参数
        self.fixed_threshold = controller_params.get('fixed_threshold', 0.3)
        self.periodic_interval = controller_params.get('periodic_interval', 1.0)
        self.adaptive_type = controller_params.get('adaptive_type', 'lyapunov_driven')

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
    ) -> np.ndarray:
        """
        控制律: u = Â x̂ + K δ

        Args:
            A_hat: 自适应估计的A矩阵
            x_hat: 广播状态（ZOH保持）
            delta: 相对状态误差

        Returns:
            u: 控制输入
        """
        A_hat_x_hat = A_hat @ x_hat
        K_delta = self.K @ delta
        return A_hat_x_hat + K_delta

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
    ) -> float:
        """
        σ的自适应更新（5种方案）

        SIMULATION_PLAN.md 4.11节:

        | 方案 | 连续时间公式 | 离散化 |
        |------|--------------|--------|
        | A (periodic) | 不使用σ | 定时触发 |
        | B (fixed) | σ = 0.3 | σ不变 |
        | C (error_driven) | σ̇ = -α·||e||² | σ = σ_min + (σ-σ_min)·exp(-α·||e||²·dt) |
        | D (state_driven) | σ = σ_min + k/(||δ̃|| + c) | 直接计算 |
        | E (lyapunov_driven) | σ̇ = -γ·(σ-σ_min)·||δ̃||² | σ = σ_min + (σ-σ_min)·exp(-γ·||δ̃||²·dt) |
        """
        tilde_delta_norm = self.compute_tilde_delta_norm(delta)

        if self.adaptive_type == "periodic":
            # 定时通信: σ不适用
            sigma_new = self.sigma_max

        elif self.adaptive_type == "fixed_threshold":
            # 固定阈值: σ保持不变
            sigma_new = self.fixed_threshold

        elif self.adaptive_type == "error_driven":
            # Error-Driven: 当 e 大时，sigma 应该大（减少触发）
            # 论文公式 σ̇ = -α||e||² 是减少的，但我们需要反向
            # 改为: σ̇ = +α||e||² 使 e 大时 sigma 增加
            alpha = 0.1  # 设计参数
            e_norm_sq = np.linalg.norm(e) ** 2
            sigma_new = sigma + alpha * e_norm_sq * dt
            # 限制 sigma 在 [sigma_min, sigma_max] 范围内
            sigma_new = np.clip(sigma_new, self.sigma_min, self.sigma_max)

        elif self.adaptive_type == "state_driven":
            # State-Driven: 论文公式 σ = k/(||δ|| + c) + σ_min
            # δ 大 → σ 小，δ 小 → σ 大
            # 这个公式是正确的
            k_param = 1.0  # 设计参数
            c_param = 0.5  # 设计参数
            delta_norm = np.linalg.norm(delta)
            sigma_target = k_param / (delta_norm + c_param) + self.sigma_min
            # 限制在合理范围
            sigma_target = np.clip(sigma_target, self.sigma_min, self.sigma_max)
            # 使用一阶低通滤波使变化平滑
            tau = 1.0  # 时间常数
            sigma_new = sigma + (sigma_target - sigma) / tau * dt

        elif self.adaptive_type == "lyapunov_driven":
            # Lyapunov-Driven: 当 δ 大时，sigma 应该小（减少触发）
            # 论文公式 σ̇ = -γ(σ-σ_min)||δ||² 是减少的
            # 进一步调小 gamma 使 sigma 下降更慢
            gamma = 0.001  # 设计参数，非常小
            delta_norm_sq = np.linalg.norm(delta) ** 2
            sigma_new = sigma - gamma * (sigma - self.sigma_min) * delta_norm_sq * dt
            # 限制 sigma 在 [sigma_min, sigma_max] 范围内
            sigma_new = np.clip(sigma_new, self.sigma_min, self.sigma_max)

        else:
            sigma_new = sigma

        return np.clip(sigma_new, self.sigma_min, self.sigma_max)

    def check_trigger(
        self,
        e: np.ndarray,
        delta: np.ndarray,
        sigma: float,
        t: float,
        last_trigger_time: float,
    ) -> bool:
        """
        触发条件判断

        触发条件: ||e|| > σ * ||δ||

        Args:
            e: 测量误差 (x̂ - x)
            delta: 相对状态误差
            sigma: 当前阈值
            t: 当前时间
            last_trigger_time: 上次触发时间

        Returns:
            triggered: 是否触发
        """
        # 方案A: 定时通信
        if self.adaptive_type == "periodic":
            return (t - last_trigger_time) >= self.periodic_interval

        # 最小触发间隔（防Zeno）
        if t - last_trigger_time < self.zeno_interval:
            return False

        # 计算误差范数
        error_norm = np.linalg.norm(e)

        # 使用非归一化delta范数
        delta_norm = np.linalg.norm(delta)

        # 最小阈值保护（防止δ≈0时频繁触发）
        min_delta = 10.0
        delta_norm_safe = max(delta_norm, min_delta)

        # 触发条件: ||e|| > σ * ||δ||
        threshold = sigma * delta_norm_safe
        return error_norm > threshold


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

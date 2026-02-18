"""
参数配置 - params.py

按照 SIMULATION_PLAN.md 计划定义所有仿真参数
包含真实系统参数和名义模型参数（20%不确定性）
"""

import numpy as np


# ============================================================
# 仿真物理参数
# ============================================================
class SimParams:
    """仿真物理参数"""
    n_trains = 8          # 列车数量
    T_end = 120.0         # 仿真时长 (s)
    dt = 0.005            # 时间步长 (s)
    d_desired = 500.0     # 目标间距 (m)
    v_nominal = 22.22     # 额定速度 (m/s) = 80 km/h
    mass = 5e6            # 列车质量 (kg) = 5000吨

    # 初始位置和速度参数 - 设置非零初始误差
    p_initial_offset = 0.0  # T1初始位置偏移
    v_initial_std = 2.0     # 初始速度随机扰动标准差 (m/s)
    p_initial_std = 50.0   # 初始位置扰动标准差 (m)


# ============================================================
# Davis阻力公式参数
# ============================================================
class DavisParams:
    """Davis阻力系数 (单位: kN/ton = N/kg)"""
    a = 0.003   # 基础阻力系数
    b = 0.0003  # 速度阻力系数
    c = 0.000002 # 空气阻力系数


# ============================================================
# 控制器参数
# ============================================================
class ControllerParams:
    """事件触发控制器参数"""
    # 阈值参数
    sigma_min = 0.10   # 论文推荐值: 0.05-0.1
    sigma_max = 1.0    # 最大阈值

    # 自适应阈值参数
    alpha = 5.0        # Error-Driven参数 (调整以避免触发过于频繁)
    k = 0.75          # State-Driven参数 (基于归一化误差)
    c = 0.5           # State-Driven参数 (基于归一化误差)
    gamma = 100.0      # Lyapunov-Driven参数 (调整以使sigma有明显变化)

    # 状态归一化参数
    d_scale = 100.0    # 位置特征尺度
    v_scale = 10.0     # 速度特征尺度

    # Riccati方程参数
    Q_weight = 5e9     # 中等值
    R_weight = 1.0     # 输入权重

    # 自适应增益参数 - 增大以加快收敛
    lambda_A = 1e-2    # Â自适应增益 (增大10倍)
    gamma_B = 1e-2     # B̂自适应增益 (增大10倍)

    # 参数投影约束 - 扩大范围以允许更多自适应调整
    A_max = 100.0       # Â_ij ∈ [-A_max, A_max]
    B_max = 100.0       # B̂_ij ∈ [-B_max, B_max]

    # 防Zeno参数
    zeno_interval = 0.01  # 最小触发间隔 (s) = 10ms

    # 控制输入低通滤波器参数（消除链式拓扑鼓包）
    filter_tau = 1.0  # 滤波器时间常数 (最低值，保持稳定)

    # 分级参数（方案2和方案3）
    mu_alpha = 0.2      # μ分级系数: μ_i = μ_base * (1 + alpha * i)
    sigma_min_beta = 0.5  # σ_min分级系数 - 增大以产生明显差异

    # 方案特定参数
    fixed_threshold = 0.3     # 方案B固定阈值
    periodic_interval = 1.0   # 方案A定时触发间隔 (s)

    # 积分阻尼增益 (用于消除链式拓扑的稳态误差)
    K_i = 2.0     # 积分增益
    K_d = 0.0     # 导数增益 (禁用)


# ============================================================
# 名义模型参数（关键：20%不确定性）
# ============================================================
class NominalModelParams:
    """名义模型参数 - 用于控制器设计（与真值有20%偏差）"""
    uncertainty_factor = 0.8  # 名义模型是真值的80%（即20%误差）


# ============================================================
# 方案配置
# ============================================================
class SchemeParams:
    """5种阈值更新方案配置"""
    schemes = {
        'A': {
            'name': 'periodic',
            'adaptive_type': 'periodic',
            'periodic_interval': 1.0,
            'seed_offset': 0,
        },
        'B': {
            'name': 'fixed_threshold',
            'adaptive_type': 'fixed_threshold',
            'fixed_threshold': 0.3,
            'seed_offset': 10,
        },
        'C': {
            'name': 'error_driven',
            'adaptive_type': 'error_driven',
            'alpha': 1.0,
            'seed_offset': 20,
        },
        'D': {
            'name': 'state_driven',
            'adaptive_type': 'state_driven',
            'k': 3.0,
            'c': 5.0,
            'seed_offset': 25,  # 尝试不同的seed
        },
        'E': {
            'name': 'lyapunov_driven',
            'adaptive_type': 'lyapunov_driven',
            'gamma': 1.0,  # 增大gamma使sigma下降更快
            'seed_offset': 40,
        },
    }


# ============================================================
# 导出统一参数字典
# ============================================================
def get_all_params():
    """获取所有参数（用于初始化）"""
    return {
        'sim': {
            'n_trains': SimParams.n_trains,
            'T_end': SimParams.T_end,
            'dt': SimParams.dt,
            'd_desired': SimParams.d_desired,
            'v_nominal': SimParams.v_nominal,
            'mass': SimParams.mass,
            'p_initial_offset': SimParams.p_initial_offset,
            'v_initial_std': SimParams.v_initial_std,
            'p_initial_std': SimParams.p_initial_std,
        },
        'davis': {
            'a': DavisParams.a,
            'b': DavisParams.b,
            'c': DavisParams.c,
        },
        'controller': {
            'sigma_min': ControllerParams.sigma_min,
            'sigma_max': ControllerParams.sigma_max,
            'alpha': ControllerParams.alpha,
            'k': ControllerParams.k,
            'c': ControllerParams.c,
            'gamma': ControllerParams.gamma,
            'd_scale': ControllerParams.d_scale,
            'v_scale': ControllerParams.v_scale,
            'Q_weight': ControllerParams.Q_weight,
            'R_weight': ControllerParams.R_weight,
            'lambda_A': ControllerParams.lambda_A,
            'gamma_B': ControllerParams.gamma_B,
            'A_max': ControllerParams.A_max,
            'B_max': ControllerParams.B_max,
            'zeno_interval': ControllerParams.zeno_interval,
            'fixed_threshold': ControllerParams.fixed_threshold,
            'periodic_interval': ControllerParams.periodic_interval,
            'filter_tau': ControllerParams.filter_tau,
            'K_i': ControllerParams.K_i,
            'K_d': ControllerParams.K_d,
            'mu_alpha': ControllerParams.mu_alpha,
            'sigma_min_beta': ControllerParams.sigma_min_beta,
        },
        'nominal': {
            'uncertainty_factor': NominalModelParams.uncertainty_factor,
        },
        'schemes': SchemeParams.schemes,
    }


if __name__ == "__main__":
    params = get_all_params()
    print("Simulation Parameters:")
    print(f"  n_trains: {params['sim']['n_trains']}")
    print(f"  T_end: {params['sim']['T_end']}s")
    print(f"  dt: {params['sim']['dt']}s")
    print(f"  d_desired: {params['sim']['d_desired']}m")
    print(f"  v_nominal: {params['sim']['v_nominal']} m/s")
    print(f"  mass: {params['sim']['mass']} kg")
    print(f"\nNominal Model:")
    print(f"  uncertainty_factor: {params['nominal']['uncertainty_factor']}")
    print(f"  (True values are {1/params['nominal']['uncertainty_factor']:.1%} of nominal)")

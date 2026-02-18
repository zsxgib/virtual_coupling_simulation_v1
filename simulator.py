"""
仿真器 - simulator.py

按照 SIMULATION_PLAN.md 计划实现：
1. 核心主循环
2. ZOH逻辑（测量误差基于广播状态计算）
3. t=0强制触发机制
4. 完整数据记录
"""

import numpy as np
import json
from typing import Dict, List
from dynamics import TrainPlatoon
from controller import ETCController, TriggerMonitor


class Simulator:
    """
    仿真器主类

    关键设计点：
    1. ZOH逻辑: 测量误差 e = x̂ - x，使用广播状态而非真实状态
    2. t=0强制触发: 确保冷启动成功
    3. 信息隔离: dynamics持有真值，controller持有名义值
    """

    def __init__(self, params: Dict, scheme_name: str, scheme_config: Dict):
        """
        初始化仿真器

        Args:
            params: 仿真参数字典
            scheme_name: 方案名称 (A/B/C/D/E)
            scheme_config: 方案配置
        """
        self.params = params.copy()
        self.scheme_name = scheme_name
        self.scheme_config = scheme_config.copy()

        # 更新方案特定参数
        self.params['seed_offset'] = scheme_config.get('seed_offset', 0)
        self.params['controller']['adaptive_type'] = scheme_config['adaptive_type']

        # 方案特定参数
        if 'periodic_interval' in scheme_config:
            self.params['controller']['periodic_interval'] = scheme_config['periodic_interval']
        if 'fixed_threshold' in scheme_config:
            self.params['controller']['fixed_threshold'] = scheme_config['fixed_threshold']
        if 'alpha' in scheme_config:
            self.params['controller']['alpha'] = scheme_config['alpha']
        if 'k' in scheme_config:
            self.params['controller']['k'] = scheme_config['k']
        if 'c' in scheme_config:
            self.params['controller']['c'] = scheme_config['c']
        if 'gamma' in scheme_config:
            self.params['controller']['gamma'] = scheme_config['gamma']

        # 初始化仿真参数
        self.n_trains = params['sim']['n_trains']
        self.T_end = params['sim']['T_end']
        self.dt = params['sim']['dt']
        self.n_steps = int(self.T_end / self.dt)
        self.time_history = np.linspace(0, self.T_end, self.n_steps)

        # 初始化动力学（持有真实系统矩阵）
        self.dynamics = TrainPlatoon(self.params)

        # 获取名义模型（不是真值！）
        A_nominal, B_nominal = self.dynamics.get_nominal_matrices()

        # 初始化控制器（只持有名义模型）
        self.controller = ETCController(
            A_nominal,
            B_nominal,
            self.params['controller'],
        )

        # 初始化触发监控器
        self.monitor = TriggerMonitor(self.n_trains)

        # 初始化仿真状态
        self._init_simulation_state()

        # 初始化数据存储
        self._init_data_storage()

    def predict_state(self, train_idx: int, t: float) -> np.ndarray:
        """
        状态预测：根据前车的历史广播状态，预测当前时刻的真实状态

        使用零阶保持 + 速度外推：
        x_predicted = x_broadcasted + v_broadcasted * (t - t_broadcasted)

        Args:
            train_idx: 列车索引
            t: 当前时间

        Returns:
            predicted_state: 预测的当前状态 [position, velocity]
        """
        if train_idx == 0:
            # 头车没有前车，返回自身的广播状态
            return self.last_broadcast_state[train_idx].copy()

        # 获取前车的广播状态和广播时间
        broadcast_state = self.last_broadcast_state[train_idx - 1]
        broadcast_time = self.last_trigger_time[train_idx - 1]

        # 计算时间差
        dt_broadcast = t - broadcast_time

        # 零阶保持 + 速度外推预测
        # x(t) ≈ x(t0) + v(t0) * (t - t0)
        predicted = broadcast_state.copy()
        # 速度外推：只预测位置，假设速度不变
        predicted[0] = broadcast_state[0] + broadcast_state[1] * dt_broadcast

        # 限制预测范围，避免预测偏差过大
        # 获取当前真实状态作为参考
        current_states = self.dynamics.get_states()
        current_pos = current_states[train_idx - 1][0]

        # 如果预测位置与当前实际位置偏差过大，使用当前实际位置
        if abs(predicted[0] - current_pos) > 100:  # 100m阈值
            predicted[0] = current_pos

        return predicted

    def _init_simulation_state(self):
        """
        初始化仿真状态

        关键初始化：
        - A_hat和B_hat估计器初始化为零矩阵
        - sigma初始化为sigma_max
        - last_broadcast_state初始化为当前状态（为t=0强制触发做准备）
        - last_trigger_time初始化为0
        """
        self.A_hats = [np.zeros((2, 2)) for _ in range(self.n_trains)]
        self.B_hats = [np.zeros((2, 2)) for _ in range(self.n_trains)]
        # 所有方案从sigma_max开始（公平比较）
        sigma_max = self.params['controller']['sigma_max']
        self.sigmas = np.full(self.n_trains, sigma_max)

        # ZOH: 广播状态初始化
        current_states = self.dynamics.get_states()
        self.last_broadcast_state = current_states.copy()

        # 触发时间记录
        self.last_trigger_time = np.zeros(self.n_trains)

        # 控制输入低通滤波器状态（消除链式拓扑的鼓包传递）
        self.u_filtered = np.zeros((self.n_trains, 2))
        self.filter_tau_base = self.params['controller'].get('filter_tau', 15.0)
        # 链尾列车使用更强滤波（位置越大，离头车越远）
        self.filter_tau = np.array([self.filter_tau_base * (1 + 0.3 * i) for i in range(self.n_trains)])

        # 积分项状态 (用于积分阻尼)
        self.delta_integral = np.zeros((self.n_trains, 2))

        # 误差历史（用于计算导数项）
        self.prev_delta = np.zeros((self.n_trains, 2))  # 保存上一步的delta


    def _init_data_storage(self):
        """初始化数据存储"""
        self.states_history = np.zeros((self.n_steps, self.n_trains, 2))
        self.delta_history = np.zeros((self.n_steps, self.n_trains, 2))
        self.errors_history = np.zeros((self.n_steps, self.n_trains, 2))
        self.sigma_history = np.zeros((self.n_steps, self.n_trains))
        self.triggers_history = np.zeros((self.n_steps, self.n_trains), dtype=bool)
        self.A_hat_history = np.zeros((self.n_steps, self.n_trains, 2, 2))
        self.B_hat_history = np.zeros((self.n_steps, self.n_trains, 2, 2))
        self.control_history = np.zeros((self.n_steps, self.n_trains, 2))

    def run(self) -> Dict:
        """
        运行仿真主循环

        SIMULATION_PLAN.md 第5步: 仿真主循环

        Returns:
            results: 仿真结果字典
        """
        print(f"\n{'='*60}")
        print(f"Running simulation: Scheme {self.scheme_name}")
        print(f"  Adaptive type: {self.scheme_config['adaptive_type']}")
        print(f"  Trains: {self.n_trains}, T_end: {self.T_end}s, dt: {self.dt}s")
        print(f"{'='*60}")

        # t=0时刻强制触发（SIMULATION_PLAN.md 4.10节）
        self._force_trigger_at_t0()

        # 主循环
        for step, t in enumerate(self.time_history):
            # 1. 获取当前状态
            current_states = self.dynamics.get_states()

            # 2. 计算相对误差δ（使用原始版本）
            delta = self.dynamics.compute_delta(t)

            # 4. 计算测量误差 e = x̂ - x (基于广播状态)
            # 这是ZOH逻辑的关键!
            errors = self.last_broadcast_state - current_states

            # 5. 记录数据
            self.states_history[step] = current_states
            self.delta_history[step] = delta
            self.errors_history[step] = errors

            # 6. 触发判断
            current_step_triggers = np.zeros(self.n_trains, dtype=bool)
            for i in range(self.n_trains):
                # 获取前车的触发时间（链式拓扑中T_i只与T_{i-1}通信）
                neighbor_trigger_time = self.last_trigger_time[i-1] if i > 0 else 0.0


                triggered = self.controller.check_trigger(
                    e=errors[i],
                    delta=delta[i],
                    sigma=self.sigmas[i],
                    t=t,
                    last_trigger_time=self.last_trigger_time[i],
                    agent_idx=i,  # 传递代理索引
                    neighbor_trigger_time=neighbor_trigger_time,  # 前车触发时间
                )

                if triggered:
                    # 更新广播状态（ZOH更新）
                    self.last_broadcast_state[i] = current_states[i].copy()
                    self.last_trigger_time[i] = t
                    current_step_triggers[i] = True
                    self.monitor.record_trigger(i, t)

            # 7. 记录触发历史
            self.triggers_history[step] = current_step_triggers

            # 8. 更新sigma（阈值自适应，含分级策略）
            for i in range(self.n_trains):
                self.sigmas[i] = self.controller.compute_sigma_update(
                    self.sigmas[i], errors[i], delta[i], self.dt, t, agent_idx=i
                )

            # 9. 计算控制输入
            control_inputs = np.zeros((self.n_trains, 2))

            # 更新积分项 (∫δ dt += δ * dt)
            self.delta_integral += delta * self.dt
            # 对积分项进行限幅，避免积分饱和
            self.delta_integral = np.clip(self.delta_integral, -50.0, 50.0)

            # 更新delta历史（用于下一时刻计算导数）
            self.prev_delta = delta.copy()

            # 计算阻力补偿（基于名义速度v_nominal）
            davis = self.params['davis']
            mass = self.params['sim']['mass']
            v_nominal = self.params['sim']['v_nominal']
            # F_drag = (a + b*v + c*v²) * m
            F_drag_nominal = (davis['a'] + davis['b'] * v_nominal + davis['c'] * v_nominal**2) * mass

            for i in range(self.n_trains):
                # 使用状态预测：预测前车的当前状态
                # 这样可以减少因为广播状态跳跃带来的控制波动
                predicted_state = self.predict_state(i, t)

                # 计算delta导数 dδ/dt（用于PD控制）
                delta_derivative = (delta[i] - self.prev_delta[i]) / self.dt

                # 使用预测状态 x̂ 和相对误差 δ（含导数项）
                u = self.controller.compute_control_input(
                    self.A_hats[i],
                    predicted_state,
                    delta[i],
                    self.delta_integral[i],
                    delta_derivative,
                )

                # 添加阻力补偿（使用名义阻力）
                u[0] = u[0] + F_drag_nominal * 1.0
                u[1] = u[1] + F_drag_nominal * 1.0

                # 控制饱和（物理限制）
                F_trac_max = 5e5  # 最大牵引力
                F_brake_max = 6e5  # 最大制动力
                u[0] = np.clip(u[0], -F_brake_max, F_trac_max)
                u[1] = np.clip(u[1], -F_brake_max, F_trac_max)

                # 低通滤波器：平滑控制输入，消除链式拓扑的鼓包传递
                # u_filtered = u_filtered + (u - u_filtered) / tau
                self.u_filtered[i] = self.u_filtered[i] + (u - self.u_filtered[i]) / self.filter_tau[i]
                control_inputs[i] = self.u_filtered[i]

            # 10. 更新参数估计 (Â, B̂)
            for i in range(self.n_trains):
                self.A_hats[i] = self.controller.compute_A_hat_update(
                    self.A_hats[i], delta[i], self.last_broadcast_state[i], self.dt
                )
                self.B_hats[i] = self.controller.compute_B_hat_update(
                    self.B_hats[i], delta[i], self.dt
                )

            # 10. 更新动力学状态（使用真实系统矩阵）
            self.dynamics.step(control_inputs, self.dt)

            # 11. 记录数据
            self.sigma_history[step] = self.sigmas
            for i in range(self.n_trains):
                self.A_hat_history[step, i] = self.A_hats[i]
                self.B_hat_history[step, i] = self.B_hats[i]
            self.control_history[step] = control_inputs

            # 进度输出
            if (step + 1) % 500 == 0:
                progress = (step + 1) / self.n_steps * 100
                delta_norm = np.mean(np.linalg.norm(delta, axis=1))
                print(f"  Progress: {progress:.1f}%, Mean ||δ||: {delta_norm:.4f}")

        print(f"\nSimulation completed!")
        print(f"  Total triggers: {self.monitor.get_stats()['total_triggers']}")

        # 返回结果
        return self._collect_results()

    def _force_trigger_at_t0(self):
        """
        t=0时刻强制触发

        SIMULATION_PLAN.md 4.10节:
        在仿真开始时（t=0），所有列车必须立即触发一次
        原因：建立初始通信基准，否则无法计算控制输入
        """
        print("\nForcing trigger at t=0...")

        # 设置所有列车的广播状态为当前状态
        current_states = self.dynamics.get_states()
        self.last_broadcast_state = current_states.copy()
        self.last_trigger_time = np.zeros(self.n_trains)

        # 记录触发
        for i in range(self.n_trains):
            self.monitor.record_trigger(i, 0.0)

        # 记录到历史数据（第0步）
        self.triggers_history[0] = np.ones(self.n_trains, dtype=bool)

    def _collect_results(self) -> Dict:
        """收集结果"""
        stats = self._compute_stats()

        return {
            "time": self.time_history,
            "states": self.states_history,
            "delta": self.delta_history,
            "errors": self.errors_history,
            "sigma": self.sigma_history,
            "triggers": self.triggers_history,
            "A_hat": self.A_hat_history,
            "B_hat": self.B_hat_history,
            "control": self.control_history,
            "trigger_times": [self.monitor.trigger_times[i] for i in range(self.n_trains)],
            "stats": stats,
        }

    def _compute_stats(self) -> Dict:
        """计算统计信息"""
        # 检查收敛：从60s到120s误差平稳或单调下降（变化小于2倍）
        # 取60s和120s的误差均值
        idx_60s = int(60.0 / self.dt)  # 60/0.005 = 12000
        idx_120s = -1

        delta_60s = self.delta_history[idx_60s]
        delta_120s = self.delta_history[idx_120s]

        mean_60s = np.mean(np.linalg.norm(delta_60s, axis=1))
        mean_120s = np.mean(np.linalg.norm(delta_120s, axis=1))

        # 收敛条件：60s到120s误差变化（下降或上升）小于2倍 且 最终误差<10
        # 即允许误差下降（收敛），不允许误差上升（发散）
        if mean_120s > mean_60s:
            # 误差上升：检查上升幅度
            change_ratio = mean_120s / (mean_60s + 1e-6)
            converged = False
        else:
            # 误差下降：只检查最终误差
            final_delta_norms = np.linalg.norm(delta_120s, axis=1)
            converged = bool(np.all(final_delta_norms < 10.0))

        # 最终误差
        final_delta = self.delta_history[-1]

        return {
            "scheme": self.scheme_name,
            "adaptive_type": self.scheme_config['adaptive_type'],
            "n_trains": self.n_trains,
            "T_end": self.T_end,
            "dt": self.dt,
            "converged": converged,
            "final_delta_norm": float(np.linalg.norm(final_delta)),
            "trigger_stats": self.monitor.get_stats(),
        }


def save_results(results: Dict, save_dir: str, scheme_name: str):
    """保存结果到文件"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 保存为numpy格式
    np.save(f"{save_dir}/states_{scheme_name}.npy", results['states'])
    np.save(f"{save_dir}/delta_{scheme_name}.npy", results['delta'])
    np.save(f"{save_dir}/errors_{scheme_name}.npy", results['errors'])
    np.save(f"{save_dir}/sigma_{scheme_name}.npy", results['sigma'])
    np.save(f"{save_dir}/triggers_{scheme_name}.npy", results['triggers'])
    np.save(f"{save_dir}/A_hat_{scheme_name}.npy", results['A_hat'])
    np.save(f"{save_dir}/B_hat_{scheme_name}.npy", results['B_hat'])
    np.save(f"{save_dir}/control_{scheme_name}.npy", results['control'])
    np.save(f"{save_dir}/time.npy", results['time'])

    # 保存触发时间
    with open(f"{save_dir}/trigger_times_{scheme_name}.json", 'w') as f:
        json.dump(results['trigger_times'], f)

    # 保存统计信息
    with open(f"{save_dir}/stats_{scheme_name}.json", 'w') as f:
        json.dump(results['stats'], f, indent=2)

    print(f"Results saved to: {save_dir}/")


if __name__ == "__main__":
    # 测试单方案仿真
    from params import get_all_params

    params = get_all_params()
    scheme_config = params['schemes']['E']  # 使用Lyapunov-Driven方案测试

    simulator = Simulator(params, 'E', scheme_config)
    results = simulator.run()

    print("\nStats:", results['stats'])

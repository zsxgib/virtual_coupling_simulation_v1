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
        # 初始sigma设为sigma_max（论文中从1.0开始下降）
        self.sigmas = np.full(self.n_trains, self.params['controller']['sigma_max'])

        # ZOH: 广播状态初始化
        current_states = self.dynamics.get_states()
        self.last_broadcast_state = current_states.copy()

        # 触发时间记录
        self.last_trigger_time = np.zeros(self.n_trains)

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

            # 2. 计算相对误差δ
            delta = self.dynamics.compute_delta(t)

            # 3. 计算测量误差 e = x̂ - x (基于广播状态)
            # 这是ZOH逻辑的关键!
            errors = self.last_broadcast_state - current_states

            # 4. 记录数据
            self.states_history[step] = current_states
            self.delta_history[step] = delta
            self.errors_history[step] = errors

            # 5. 触发判断
            current_step_triggers = np.zeros(self.n_trains, dtype=bool)
            for i in range(self.n_trains):
                triggered = self.controller.check_trigger(
                    e=errors[i],
                    delta=delta[i],
                    sigma=self.sigmas[i],
                    t=t,
                    last_trigger_time=self.last_trigger_time[i],
                )

                if triggered:
                    # 更新广播状态（ZOH更新）
                    self.last_broadcast_state[i] = current_states[i].copy()
                    self.last_trigger_time[i] = t
                    current_step_triggers[i] = True
                    self.monitor.record_trigger(i, t)

            # 6. 记录触发历史
            self.triggers_history[step] = current_step_triggers

            # 7. 更新sigma（阈值自适应）
            for i in range(self.n_trains):
                self.sigmas[i] = self.controller.compute_sigma_update(
                    self.sigmas[i], errors[i], delta[i], self.dt, t
                )

            # 8. 计算控制输入
            control_inputs = np.zeros((self.n_trains, 2))

            # 计算阻力补偿（基于名义速度v_nominal）
            davis = self.params['davis']
            mass = self.params['sim']['mass']
            v_nominal = self.params['sim']['v_nominal']
            # F_drag = (a + b*v + c*v²) * m
            F_drag_nominal = (davis['a'] + davis['b'] * v_nominal + davis['c'] * v_nominal**2) * mass

            for i in range(self.n_trains):
                # 使用广播状态 x̂ 和相对误差 δ
                u = self.controller.compute_control_input(
                    self.A_hats[i],
                    self.last_broadcast_state[i],
                    delta[i],
                )

                # 添加阻力补偿（使用名义阻力）
                u[0] = u[0] + F_drag_nominal * 1.0
                u[1] = u[1] + F_drag_nominal * 1.0

                # 控制饱和（物理限制）
                F_trac_max = 5e5  # 最大牵引力
                F_brake_max = 6e5  # 最大制动力
                u[0] = np.clip(u[0], -F_brake_max, F_trac_max)
                u[1] = np.clip(u[1], -F_brake_max, F_trac_max)
                control_inputs[i] = u

            # 9. 更新参数估计 (Â, B̂)
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
        # 检查收敛
        delta_last_1000 = self.delta_history[-1000:]
        delta_norms = np.linalg.norm(delta_last_1000, axis=2)
        converged = bool(np.all(delta_norms < 10.0))

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

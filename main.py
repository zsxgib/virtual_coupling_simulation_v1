"""
主程序 - main.py

按照 SIMULATION_PLAN.md 计划实现：
依次运行5种方案 (A/B/C/D/E)，保存结果到results/
"""

import os
import sys
import json
from datetime import datetime

# 确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from params import get_all_params
from simulator import Simulator, save_results


def run_all_schemes():
    """运行所有5种方案"""
    params = get_all_params()
    schemes = params['schemes']

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # 保存参数配置
    with open(f"{results_dir}/params.json", 'w') as f:
        json.dump(params, f, indent=2, default=str)

    print("="*60)
    print("Virtual Coupling Simulation - All Schemes")
    print("="*60)
    print(f"\nResults will be saved to: {results_dir}/")
    print(f"\nSimulating {len(schemes)} schemes: {list(schemes.keys())}")
    print("="*60)

    # 依次运行各方案
    all_stats = {}

    for scheme_name, scheme_config in schemes.items():
        print(f"\n{'#'*60}")
        print(f"# Scheme {scheme_name}: {scheme_config['name']}")
        print(f"{'#'*60}")

        # 创建仿真器
        simulator = Simulator(params, scheme_name, scheme_config)

        # 运行仿真
        results = simulator.run()

        # 保存结果
        save_results(results, results_dir, scheme_name)

        # 记录统计
        all_stats[scheme_name] = results['stats']
        print(f"\nScheme {scheme_name} completed!")
        print(f"  Total triggers: {results['stats']['trigger_stats']['total_triggers']}")
        print(f"  Converged: {results['stats']['converged']}")

    # 保存汇总统计
    with open(f"{results_dir}/all_stats.json", 'w') as f:
        json.dump(all_stats, f, indent=2)

    print("\n" + "="*60)
    print("All simulations completed!")
    print("="*60)

    # 打印汇总表
    print("\nSummary:")
    print("-"*60)
    print(f"{'Scheme':<10} {'Type':<20} {'Triggers':<10} {'Converged':<10}")
    print("-"*60)
    for scheme_name, stats in all_stats.items():
        adaptive_type = stats['adaptive_type']
        triggers = stats['trigger_stats']['total_triggers']
        converged = "Yes" if stats['converged'] else "No"
        print(f"{scheme_name:<10} {adaptive_type:<20} {triggers:<10} {converged:<10}")
    print("-"*60)

    return results_dir


if __name__ == "__main__":
    results_dir = run_all_schemes()
    print(f"\nFinal results saved to: {results_dir}/")

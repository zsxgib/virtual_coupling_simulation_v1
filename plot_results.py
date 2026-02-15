"""
绘图脚本 - plot_results.py

按照 figure_descriptions.md 规范生成 Fig.1-5 学术图表
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import json
import os

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300

# 数据路径
DATA_DIR = '/home/zsx/tmp/2025-12-07-paper/02_多智能体系统控制/9_Adaptive_Event_Driven_Consensus_Complete/virtual_coupling_simulation_v1/results/20260215_202509'
OUTPUT_DIR = '/home/zsx/tmp/2025-12-07-paper/02_多智能体系统控制/9_Adaptive_Event_Driven_Consensus_Complete/virtual_coupling_simulation_v1/results/figures'

# 方案列表
SCHEMES = ['A', 'B', 'C', 'D', 'E']
SCHEME_NAMES = {
    'A': 'Periodic',
    'B': 'Fixed Threshold',
    'C': 'Error-Driven',
    'D': 'State-Driven',
    'E': 'Lyapunov-Driven'
}

# IEEE 标准颜色 (0-1 range for matplotlib)
IEEE_COLORS = {
    'blue': (0, 0, 0.498),
    'red': (0.784, 0, 0),
    'green': (0, 0.588, 0),
    'orange': (0.863, 0.471, 0),
    'purple': (0.549, 0, 0.549),
    'brown': (0.471, 0.314, 0.157),
    'pink': (0.784, 0.392, 0.588),
    'cyan': (0, 0.588, 0.588),
    'black': (0, 0, 0),
    'gray': (0.502, 0.502, 0.502),
}

# 方案颜色
SCHEME_COLORS = {
    'A': IEEE_COLORS['blue'],
    'B': IEEE_COLORS['red'],
    'C': IEEE_COLORS['green'],
    'D': IEEE_COLORS['orange'],
    'E': IEEE_COLORS['purple'],
}

# 线型
LINE_STYLES = {
    'A': '-',
    'B': '--',
    'C': '-.',
    'D': ':',
    'E': '-',
}

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载所有方案的数据"""
    data = {}

    # 加载时间
    data['time'] = np.load(f'{DATA_DIR}/time.npy')

    for scheme in SCHEMES:
        data[scheme] = {
            'states': np.load(f'{DATA_DIR}/states_{scheme}.npy'),
            'delta': np.load(f'{DATA_DIR}/delta_{scheme}.npy'),
            'sigma': np.load(f'{DATA_DIR}/sigma_{scheme}.npy'),
            'triggers': np.load(f'{DATA_DIR}/triggers_{scheme}.npy'),
            'A_hat': np.load(f'{DATA_DIR}/A_hat_{scheme}.npy'),
            'B_hat': np.load(f'{DATA_DIR}/B_hat_{scheme}.npy'),
        }
        with open(f'{DATA_DIR}/stats_{scheme}.json', 'r') as f:
            data[scheme]['stats'] = json.load(f)

    # 真实系统矩阵
    from dynamics import TrainPlatoon
    from params import get_all_params
    params = get_all_params()
    platoon = TrainPlatoon(params)
    data['A_true'] = platoon.A_true
    data['B_true'] = platoon.B_true

    return data


# ============================================================
# Fig.1: 拓扑与相轨迹
# ============================================================

# ============================================================
# Fig.1(a): 拓扑
# ============================================================
def plot_fig1a(data):
    """绘制 Fig.1(a) 链式拓扑"""
    fig, ax1 = plt.subplots(figsize=(7, 2.5))

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Chain Topology for 8-Train Platoon', fontweight='bold', pad=10)

    n_trains = 8
    rect_width = 0.8
    rect_height = 0.4
    spacing = 1.1
    start_x = 0.8

    for i in range(n_trains):
        x = start_x + i * spacing
        rect = patches.Rectangle((x - rect_width/2, 2.5), rect_width, rect_height,
                                  linewidth=1, edgecolor='black', facecolor=(240/255, 240/255, 240/255))
        ax1.add_patch(rect)
        ax1.text(x, 2.9, f'T{i+1}', ha='center', va='center', fontsize=10)
        ax1.text(x, 1.8, f'{i*500}', ha='center', va='center', fontsize=9)

    for i in range(n_trains - 1):
        x1 = start_x + i * spacing + rect_width/2
        x2 = start_x + (i + 1) * spacing - rect_width/2
        y = 1.2
        ax1.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='<->', color=(0, 0.588, 0), lw=1.5))
        ax1.text((x1+x2)/2, y+0.15, 'd=500m', ha='center', fontsize=9)

    for i in range(n_trains - 1):
        x1 = start_x + i * spacing
        x2 = start_x + (i + 1) * spacing
        y = 2.1
        ax1.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='<->', color=(0, 0, 0.784), lw=1.5))
    ax1.text(start_x + 3.5 * spacing, 1.5, 'C (communication)', ha='center', fontsize=8, style='italic')
    ax1.text(5, 0.3, 'Each train communicates only with its leader and follower',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig1a_Topology.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig1a_Topology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.1a saved.")


# ============================================================
# Fig.1(b): 相轨迹
# ============================================================
def plot_fig1b(data):
    """绘制 Fig.1(b) 相平面轨迹"""
    fig, ax2 = plt.subplots(figsize=(7, 3))

    ax2.set_title('(b) Phase Trajectory', fontweight='bold', pad=10)

    states = data['E']['states']

    train_colors = [
        IEEE_COLORS['blue'], IEEE_COLORS['red'], IEEE_COLORS['green'],
        IEEE_COLORS['orange'], IEEE_COLORS['purple'], IEEE_COLORS['brown'],
        IEEE_COLORS['pink'], IEEE_COLORS['cyan']
    ]

    for i in range(8):
        p = states[:, i, 0]
        v = states[:, i, 1]
        ax2.plot(p, v, color=train_colors[i], linewidth=1.5, alpha=0.8)
        ax2.scatter(p[0], v[0], color=train_colors[i], s=30, marker='o', zorder=5)
        ax2.scatter(p[-1], v[-1], color=train_colors[i], s=30, marker='s', zorder=5)

    legend_elements = [Line2D([0], [0], color=train_colors[i], lw=1.5, label=f'T{i+1}') for i in range(8)]
    ax2.legend(handles=legend_elements, loc='upper left', ncol=2, framealpha=0.9)

    ax2.set_xlabel('Position p (m)')
    ax2.set_ylabel('Velocity v (m/s)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig1b_Trajectory.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig1b_Trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.1b saved.")

def plot_fig2(data):
    """绘制 Fig.2 相对误差对比"""
    fig, ax = plt.subplots(figsize=(7, 4))

    time = data['time']

    for scheme in SCHEMES:
        delta = data[scheme]['delta']

        # 计算平均相对误差: mean(||δ_i(t)||_2)
        delta_norms = np.linalg.norm(delta, axis=2)  # (n_steps, n_trains)
        mean_delta = np.mean(delta_norms, axis=1)   # (n_steps,)

        ax.semilogy(time, mean_delta + 1e-10,
                   color=SCHEME_COLORS[scheme],
                   linestyle=LINE_STYLES[scheme],
                   linewidth=1.5,
                   label=SCHEME_NAMES[scheme])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Mean Relative Error $||\delta(t)||_2$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, time[-1])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig2_Error_Comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig2_Error_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.2 saved.")


# ============================================================
# Fig.3: 触发事件分布
# ============================================================
def plot_fig3(data):
    """绘制 Fig.3 触发事件分布"""
    fig, ax = plt.subplots(figsize=(10, 5))

    time = data['time']
    T_end = time[-1]

    # 5个方案，每行一个
    row_height = 0.15
    bottom_start = 0.1

    for idx, scheme in enumerate(SCHEMES):
        triggers = data[scheme]['triggers']
        n_trains = triggers.shape[1]

        # 收集所有触发时刻
        trigger_times = []
        for i in range(n_trains):
            times = time[triggers[:, i] > 0]
            trigger_times.extend(times)

        # 绘制散点
        y_positions = [bottom_start + idx * (row_height + 0.05)] * len(trigger_times)
        ax.scatter(trigger_times, y_positions,
                  color=SCHEME_COLORS[scheme], s=15, alpha=0.7)

        # 绘制时间轴线
        ax.hlines(bottom_start + idx * (row_height + 0.05), 0, T_end,
                 color='gray', linewidth=0.5, linestyle='-')

        # 方案标签
        ax.text(-1, bottom_start + idx * (row_height + 0.05),
               SCHEME_NAMES[scheme], ha='right', va='center',
               fontsize=9)

    ax.set_xlim(-2, T_end + 1)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel('Time (s)')
    ax.set_yticks([])
    ax.set_title('Trigger Event Distribution', fontweight='bold', pad=10)

    # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig3_Trigger_Events.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig3_Trigger_Events.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.3 saved.")


# ============================================================
# Fig.4(a): 阈值演化
# ============================================================
def plot_fig4a(data):
    """绘制 Fig.4(a) 阈值演化 - 只显示前15秒以突出初始动态"""
    fig, ax1 = plt.subplots(figsize=(7, 3))

    time = data['time']

    # 只显示前15秒
    t_max = 15
    idx_max = int(t_max / (time[1] - time[0]))

    for scheme in ['C', 'D', 'E']:
        sigma = data[scheme]['sigma']
        # 计算所有列车的平均sigma
        sigma_mean = np.mean(sigma, axis=1)
        ax1.plot(time[:idx_max], sigma_mean[:idx_max],
                color=SCHEME_COLORS[scheme],
                linestyle=LINE_STYLES[scheme],
                linewidth=1.5,
                label=f'{scheme_names[scheme]}')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(r'Threshold $\sigma(t)$')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, 1.1)
    ax1.set_title('(a) Threshold Evolution', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig4a_Threshold.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig4a_Threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.4a saved.")


# ============================================================
# Fig.4(b): 参数估计误差
# ============================================================
def plot_fig4b(data):
    """绘制 Fig.4(b) 参数估计误差"""
    fig, ax2 = plt.subplots(figsize=(7, 3))

    time = data['time']
    A_true = data['A_true']
    B_true = data['B_true']

    for scheme in ['E']:
        A_hat = data[scheme]['A_hat']
        B_hat = data[scheme]['B_hat']

        A_error = np.zeros(len(time))
        B_error = np.zeros(len(time))
        for t in range(len(time)):
            A_error[t] = np.linalg.norm(A_hat[t, 0] - A_true, 'fro')
            B_error[t] = np.linalg.norm(B_hat[t, 0] - B_true, 'fro')

        ax2.semilogy(time, A_error + 1e-15,
                    color=IEEE_COLORS['blue'],
                    linestyle='-',
                    linewidth=1.5,
                    label=r'$\|\hat{A} - A\|_{F}$')
        ax2.semilogy(time, B_error + 1e-15,
                    color=IEEE_COLORS['red'],
                    linestyle='--',
                    linewidth=1.5,
                    label=r'$\|\hat{B} - B\|_{F}$')

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Estimation Error')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, time[-1])
    ax2.set_title('(b) Parameter Estimation Error', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig4b_Parameter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig4b_Parameter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.4b saved.")


# ============================================================
# Fig.5: 通信量对比
# ============================================================
def plot_fig5(data):
    """绘制 Fig.5 通信量对比"""
    fig, ax = plt.subplots(figsize=(7, 4))

    # 统计触发次数
    trigger_counts = []
    for scheme in SCHEMES:
        stats = data[scheme]['stats']
        count = stats['trigger_stats']['total_triggers']
        trigger_counts.append(count)

    # 方案B的触发次数作为基准
    baseline = trigger_counts[1]  # B是索引1

    # 绘制柱状图
    x_pos = np.arange(len(SCHEMES))
    bars = ax.bar(x_pos, trigger_counts, color=[SCHEME_COLORS[s] for s in SCHEMES],
                  edgecolor='black', linewidth=1)

    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, trigger_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}',
               ha='center', va='bottom', fontsize=10)

        # 计算相对减少百分比
        if i != 1:  # 不是方案B
            reduction = (baseline - count) / baseline * 100
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'-{reduction:.1f}%',
                   ha='center', va='center', fontsize=9,
                   color='white', fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCHEME_NAMES[s] for s in SCHEMES], rotation=15, ha='right')
    ax.set_ylabel('Total Trigger Count')
    ax.set_title('Communication Load Comparison', fontweight='bold', pad=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/Fig5_Communication_Load.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{OUTPUT_DIR}/Fig5_Communication_Load.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig.5 saved.")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    # 修正方案名称映射（解决变量名问题）
    scheme_names = SCHEME_NAMES

    print("Loading data...")
    data = load_data()

    print("Plotting figures...")
    plot_fig1a(data)
    plot_fig1b(data)
    plot_fig2(data)
    plot_fig3(data)
    plot_fig4a(data)
    plot_fig4b(data)
    plot_fig5(data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")

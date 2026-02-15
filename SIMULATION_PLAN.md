# Part B 仿真执行计划

> 从零开始构建新的仿真框架，用于生成 Part B 虚拟重联论文所需的5个核心图表

---

## 一、仿真目标

生成 Part B 虚拟重联论文所需的5个核心图表：
- Fig.1: 拓扑结构 + 相轨迹
- Fig.2: 相对误差对比（5种方案）
- Fig.3: 触发事件分布（5种方案）
- Fig.4: 阈值演化 + 参数估计
- Fig.5: 通信量对比（5种方案）

---

## 二、理论框架声明

### 2.1 核心假设

**本仿真基于以下理论假设：**

- 论文的理论证明（稳定性、自适应收敛）基于 **LTI（线性时不变）系统**假设
- 系统形式：$\dot{x} = A x + B u$，其中 $A$ 和 $B$ 是**常数矩阵**
- 控制器设计基于该LTI假设

### 2.2 仿真中的系统定义

**"真实系统"（上帝视角 - 用于状态传播）：**

- 使用常值矩阵 $A_{true}$ 和 $B_{true}$ 进行线性状态更新
- $A_{true}$ 和 $B_{true}$ 通过以下方式得到：
  1. 使用Davis公式计算阻力系数
  2. 在额定速度 $v_0 = 22.22$ m/s 处线性化
  3. 得到常值矩阵 $A$ 和 $B$

**"名义模型"（工程师视角 - 用于控制器设计）：**

- 论文的核心假设：**系统矩阵未知**
- 控制器设计者只能使用一组**近似估计**的名义矩阵 $A_0$ 和 $B_0$
- $A_0$ 和 $B_0$ 与真实值存在偏差（例如 20% 误差）
- 这样才能验证论文的"未知系统自适应控制"能力

**这意味着：**

- 仿真传播使用：**线性动力学**（A和B进行一次计算后，整个仿真过程中保持不变）
- 不是：在每个时间步使用非线性阻力公式
- 控制器设计者**不知道**真实系统矩阵，必须依靠自适应估计

---

## 三，可参考的现有代码

| 模块 | 参考文件 | 参考内容 |
|------|----------|----------|
| 动力学模型 | `virtual_coupling_simulation/simulation/train_dynamics.py` | Davis阻力公式、如何从阻力公式得到A和B矩阵 |
| 控制器结构 | `virtual_coupling_simulation/simulation/etc_controller.py` | 控制律、自适应律、触发条件、五种阈值更新方案、Riccati方程求解 |
| 仿真流程 | `virtual_coupling_simulation/simulation/scenario_base.py` | 主循环结构、ZOH状态保持逻辑、数据记录方式 |

---

## 四、关键数学公式

### 4.1 A和B矩阵的计算（初始化阶段，一次性）

```
步骤1：计算阻力系数（在v0处）
F_resistance = (a + b*v + c*v²) * m

步骤2：在v0处线性化
a21 = -2 * c * v0

步骤3：得到常值矩阵A和B
A = [[0, 1], [0, a21]] = [[0, 1], [0, -0.002222]]
B = [[0, 0], [1/m, k/m]]
```

### 4.2 线性动力学（状态传播）

```
# 在每个仿真步使用（常值A_true和B_true）
x_next = x + (A_true * x + B_true * u) * dt
```

### 4.2.1 名义模型（用于控制器设计）

```
# 论文核心：系统矩阵未知，控制器使用近似估计的名义矩阵
# 对涉及物理参数的部分引入20%误差

# 对于A矩阵：只对a21（阻力系数相关项）加误差
a21_true = -2 * c * v_nominal
a21_nominal = a21_true * uncertainty_factor
A_nominal = [[0, 1], [0, a21_nominal]]

# 对于B矩阵：整体乘以uncertainty_factor
B_nominal = B_true * uncertainty_factor

# 控制器使用A_nominal和B_nominal设计K增益
```

### 4.3 虚拟领航者（Virtual Leader）

```
定义：
- p_virtual(t) = p_T1_initial + v_nominal * t
- v_virtual(t) = v_nominal (常数)

用途：作为T1的参考目标
```

### 4.4 相对误差δ的计算

```
对于T1 (i=1)：
δ_1 = [p_virtual - p_1 - 0, v_virtual - v_1]^T
     = [p_virtual - p_1, v_nominal - v_1]^T

对于T2-T8 (i>1)：
δ_i = [p_{i-1} - p_i - d_desired, v_{i-1} - v_i]^T
```

### 4.5 归一化相对误差

```
δ̃ = [δ_p / d_scale, δ_v / v_scale]^T

参数：
- d_scale = 100 (位置特征尺度)
- v_scale = 10 (速度特征尺度)
```

### 4.6 控制律

```
u_i = Â_i x̂_i + K δ_i

注意：控制使用的是未归一化的δ
```

### 4.7 参数自适应律（连续时间形式）

```
Â̇_i = Λ_i δ_i x̂_i^T
B̂̇_i = Γ_i δ_i δ_i^T
```

**离散化方法：前向欧拉法**
```
Â_i(k+1) = Â_i(k) + Λ_i δ_i(k) x̂_i(k)^T * dt
B̂_i(k+1) = B̂_i(k) + Γ_i δ_i(k) δ_i(k)^T * dt
```

### 4.8 触发条件

```
||e_i(t)|| > σ_i * ||δ_i|| / L_ii

其中：
- e_i(t) = x̂_i(t) - x_i(t)：测量误差
- x̂_i(t)：广播状态（ZOH保持）
- x_i(t)：真实状态
- L_ii = 1（链式拓扑）
```

### 4.9 状态保持逻辑（ZOH）

```
关键理解：
- 在触发时刻 t_k：x̂_i(t_k) = x_i(t_k)，然后 e = 0
- 在触发间隔 [t_k, t_{k+1})：x̂_i(t) = x̂_i(t_k)，保持不变
- 在 t_{k+1} 时刻：检查触发条件，使用当前 x̂_i 和 x_i
```

### 4.10 t=0时刻的强制触发

```
在仿真开始时（t=0），所有列车必须立即触发一次
原因：建立初始通信基准，否则无法计算控制输入

实现：
- 在主循环开始时，设置所有列车的last_broadcast_state = current_state
- 或者在第一个时间步强制所有列车触发
```

### 4.11 五种阈值更新方案

| 方案 | 连续时间公式 | 离散化 |
|------|--------------|--------|
| A (periodic) | 不使用σ | 定时触发 |
| B (fixed) | σ = 0.3 | σ不变 |
| C (error_driven) | σ̇ = -α·||e||² | σ = σ_min + (σ-σ_min)·exp(-α·||e||²·dt) |
| D (state_driven) | σ = σ_min + k/(\|\|δ̃\|\|+c) | 直接计算 |
| E (lyapunov_driven) | σ̇ = -γ·(σ-σ_min)·\|\|δ̃\|\|² | σ = σ_min + (σ-σ_min)·exp(-γ·\|\|δ̃\|\|²·dt) |

---

## 五、需要开发的模块

### 5.1 模块列表

```
virtual_coupling_simulation_v1/
├── params.py      # 参数配置
├── dynamics.py    # 线性动力学模型（A和B矩阵计算、状态传播）
├── controller.py  # 事件触发控制器（K计算、自适应、触发判断）
├── simulator.py   # 仿真主循环（ZOH逻辑、数据记录）
├── main.py       # 主程序入口
└── results/      # 输出目录
```

### 5.2 模块功能

| 模块 | 功能 | 关键实现点 |
|------|------|-----------|
| params.py | 集中管理参数 | 仿真参数、控制器参数、五种方案参数 |
| dynamics.py | 线性动力学模型 | A和B矩阵计算、线性状态更新、虚拟领航者 |
| controller.py | 事件触发控制 | Riccati方程求解K、自适应律、触发判断 |
| simulator.py | 时间步进 | ZOH状态保持、t=0强制触发、数据记录 |
| main.py | 方案运行 | 5种方案依次运行、结果保存 |

---

## 六，执行顺序

### 步骤1：创建目录结构

```
mkdir -p virtual_coupling_simulation_v1/results
```

### 步骤2：实现 params.py

定义以下参数：

- 仿真参数：n_trains=8, T_end=10, dt=0.005, d_desired=500, v_nominal=22.22, mass=5e6
- 控制器参数：sigma_min=0.1, sigma_max=1.0, d_scale=100, v_scale=10, Q_weight=3e7, R_weight=1.0, lambda_A=1e-6, gamma_B=1e-6, A_max=10, B_max=10, zeno_interval=0.01
- **名义模型参数**：uncertainty_factor=0.8（名义模型是真值的80%，即20%误差）
- 方案参数：A/B/C/D/E各自参数
- 随机种子：A=0, B=10, C=20, D=30, E=40

### 步骤3：实现 dynamics.py

实现以下内容：

**Train类：**

- __init__：存储A_true和B_true矩阵（常值）
  - 使用Davis系数计算A_true和B_true（一次性）
- update(x, u, dt)：使用**真实系统矩阵**进行状态更新
  - 使用：x_next = x + (A_true @ x + B_true @ u) * dt

**TrainPlatoon类：**

- __init__：初始化8辆列车、创建虚拟领航者
- **计算名义模型**（A_0, B_0）：
  - 只对涉及物理参数的部分加误差（如阻力系数相关的a21）
  - 对于运动学关系 [0,1] 保持不变
  - A_0 = [[0, 1], [0, a21_true * uncertainty_factor]]
  - B_0 = B_true * uncertainty_factor
- **返回两个接口**：
  - get_true_matrices()：返回A_true, B_true（用于仿真传播）
  - get_nominal_matrices()：返回A_0, B_0（用于控制器设计）
- set_virtual_leader_position(t)：计算虚拟领航者在时刻t的位置
  - p_virtual = p_T1_initial + v_nominal * t
- compute_delta(t)：计算相对误差δ
  - T1：与虚拟领航者比较
  - T2-T8：与前车比较
- step(controls, dt)：更新所有列车

### 步骤4：实现 controller.py

实现ETCController类：

- __init__(A_nom, B_nom, params)：
  - **接收名义矩阵** A_nom 和 B_nom（不是真值！）
  - 使用scipy.linalg.solve_continuous_are求解Riccati方程
  - 计算K = R^(-1) @ B_nom.T @ P
  - **关键**：控制器设计者不知道真实系统，必须依靠名义模型

- compute_control(A_hat, x_hat, delta)：
  - u = A_hat @ x_hat + K @ delta

- update_A_hat(A_hat, delta, x_hat, dt)：
  - 使用前向欧拉法离散化
  - dA = lambda_A * outer(delta, x_hat) * dt
  - 参数投影到[-A_max, A_max]

- update_B_hat(B_hat, delta, dt)：
  - 使用前向欧拉法
  - dB = gamma_B * outer(delta, delta) * dt
  - 参数投影到[-B_max, B_max]

- update_sigma(sigma, e, delta, dt, scheme, scheme_params)：
  - 实现五种方案的离散化公式
  - 注意：使用归一化的δ̃计算

- check_trigger(e, delta, sigma, t, last_trigger_time, scheme, scheme_params)：
  - 首先检查是否达到最小间隔（防Zeno）
  - 然后判断：||e|| > σ * ||δ||

### 步骤5：实现 simulator.py（关键：ZOH逻辑+t=0强制触发）

实现Simulator类：

**初始化：**
- 创建A_hat和B_hat估计器（初始化为零矩阵）
- 创建sigma数组（初始化为sigma_max）
- 创建last_broadcast_state数组（存储x̂）
- 创建last_trigger_times数组
- **关键：初始化时设置last_broadcast_state = current_state**（为t=0强制触发做准备）

**主循环run()：**
- 对每个时间步t：
  1. 记录当前状态
  2. 计算虚拟领航者位置（用于T1的δ计算）
  3. 计算相对误差δ
  4. **计算测量误差**：
     - e = last_broadcast_state - current_state
  5. **触发判断逻辑**：
     - 如果 t == 0：
       - is_triggered = True（强制触发）
     - 否则：
       - is_triggered = check_trigger(...)
  6. **如果触发**：
     - 记录触发时间
     - 更新last_broadcast_state = current_state（ZOH更新）
  7. 更新sigma
  8. 计算控制输入
  9. 更新参数估计A_hat和B_hat
  10. 更新动力学状态

**返回：**
- states_history, delta_history, sigma_history, triggers_history, A_hats, B_hats

### 步骤6：实现 main.py

- 依次运行A/B/C/D/E五种方案
- **关键：信息隔离**
  ```
  # main.py 中的初始化顺序
  # 1. 创建动力学模型（持有真实系统矩阵）
  train_dynamics = TrainPlatoon(mass=5e6, v_nominal=22.22, ...)
  A_true, B_true = train_dynamics.get_true_matrices()

  # 2. 获取名义模型（与真值有20%偏差）
  A_nominal, B_nominal = train_dynamics.get_nominal_matrices()

  # 3. 创建控制器（只持有名义模型！）
  # 控制器设计者不知道真实系统，必须依靠自适应估计
  controller = ETCController(A_nominal, B_nominal, params)
  ```
- 统计并打印触发次数
- 保存结果

### 步骤7：运行仿真

```bash
cd virtual_coupling_simulation_v1
python main.py
```

### 步骤8：验证结果

检查：
- 误差是否收敛
- 触发是否发生（特别是t=0时刻）
- 最小触发间隔是否>10ms

---

## 七，数据提取规范

### 7.1 矩阵范数

| 数据项 | 范数类型 |
|--------|----------|
| 状态误差 \|\|e\|\| | 2-范数 |
| 相对误差 \|\|δ\|\| | 2-范数 |
| 参数误差 \|\|Â - A\|\| | 弗罗贝尼乌斯范数 |

### 7.2 图表数据

| 图表 | 数据来源 |
|------|----------|
| Fig.1(b) | states_history |
| Fig.2 | mean(\|\|δ_i(t)\|\|) 对所有列车求平均 |
| Fig.3 | triggers_history |
| Fig.4(a) | sigma_history (C/D/E方案) |
| Fig.4(b) | \|\|Â-A\|\|_F, \|\|B̂-B\|\|_F |
| Fig.5 | sum(triggers) 每种方案总触发次数 |

---

## 八，时间估计

| 步骤 | 时长 |
|------|------|
| 步骤1-2 | 10 min |
| 步骤3-6 | 2-3 hours |
| 步骤7 | 30 min |
| 步骤8 | 20 min |
| **总计** | **~4 hours** |

---

## 九，检查清单

编码完成后检查：

- [ ] A_true和B_true是常值（初始化后不变），用于状态传播
- [ ] A_nominal和B_nominal与真值有20%偏差（uncertainty_factor=0.8）
- [ ] 控制器只持有A_nominal和B_nominal，不知道真实系统
- [ ] K矩阵基于名义矩阵计算，不是真值
- [ ] 状态更新使用线性公式 x + (A_true@x + B_true@u)*dt
- [ ] T1的δ使用虚拟领航者计算
- [ ] t=0时刻所有列车强制触发一次
- [ ] ZOH逻辑正确：x̂在触发时刻更新，间隔内保持
- [ ] 测量误差e = x̂ - x 使用的是广播状态而非真实状态
- [ ] Riccati方程正确求解K矩阵
- [ ] 自适应律使用前向欧拉离散化
- [ ] 防Zeno机制生效
- [ ] 数据记录完整

---

## 十一、自适应阈值参数问题诊断与修正计划

### 11.1 问题根源分析

首次仿真运行后，发现以下问题：

| 问题 | 现象 | 根因 |
|------|------|------|
| D/E阈值几乎不变 | σ从1.0→0.994，几乎无变化 | `gamma=1.0` 太小，`exp(-γ·dt) ≈ 0.995` 每步仅变化0.5% |
| C阈值下降太快 | σ快速降到0.1，触发过于频繁 | `alpha=1.0` 太小导致阈值过低，触发10,083次(比B多190%) |
| 前期/后期触发比例相同 | 未体现"按需通信"特性 | 自适应律收敛太快，系统快速达到稳态 |

### 11.2 参数问题详解

**D方案 (State-Driven) 公式分析：**
```python
target_sigma = sigma_min + k / (tilde_delta_norm + c)
decay = exp(-gamma * dt)  # gamma=1.0, dt=0.005
sigma_new = target_sigma + (sigma - target_sigma) * decay
```
- `decay = exp(-1.0 × 0.005) = 0.995`
- 60秒 (12,000步) 后变化: `0.995^12000 ≈ 3×10^-27` → 几乎不变!

**E方案 (Lyapunov-Driven) 公式分析：**
```python
sigma_new = sigma_min + (sigma - sigma_min) * exp(-gamma * ||δ̃||² * dt)
```
- 收敛后 `||δ̃|| ≈ 0.1`，则 `||δ̃||² ≈ 0.01`
- `decay = exp(-1.0 × 0.01 × 0.005) = 0.99995`
- 每步仅变化0.005%，60秒后几乎不变!

**C方案 (Error-Driven) 问题：**
```python
sigma_new = sigma_min + (sigma - sigma_min) * exp(-alpha * ||e||² * dt)
```
- 阈值快速降到 `sigma_min = 0.1`
- 阈值太低 → 触发条件 `||e|| > 0.1·||δ||` 更容易满足
- 导致触发次数爆炸式增长

### 11.3 修正方案

#### 方案A：调整参数（推荐）

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| `gamma` | 1.0 | **50~200** | 使每步变化约37%~5%，60秒内有明显效果 |
| `alpha` | 1.0 | **5~10** | 加快下降但不过快到达sigma_min |
| `k` | 0.75 | 保持 | 控制阈值上限 |
| `c` | 0.5 | 保持 | 控制分母防零 |

**修正后的预期效果：**
- D方案：`decay = exp(-100 × 0.005) = 0.607`，每步变化39%，更快跟踪目标sigma
- E方案：`decay = exp(-100 × 0.01 × 0.005) = 0.995`，每步变化0.5%，保持平稳下降
- C方案：sigma不会太快降到0.1，触发次数合理

#### 方案B：修改D方案逻辑（更彻底）

当前D方案使用gamma做平滑，应该直接计算或增大gamma：
```python
# 方案B1：直接计算，不平滑
sigma_new = sigma_min + k / (tilde_delta_norm + c)

# 方案B2：增大gamma
decay = exp(-100 * dt)  # 每步变化37%
sigma_new = target_sigma + (sigma - target_sigma) * decay
```

#### 方案C：调整触发条件体现"按需通信"

要体现前期多触发、后期少触发：
- 触发条件 `||e|| > σ·||δ||` 中，σ应与误差相关
- 当 `||δ||` 大时，σ自动变大（不易触发）
- 当 `||δ||` 小时，σ自动变小（容易触发）
- 或者调整参数使系统在稳态时不易触发

### 11.4 修正步骤

1. **修改 params.py 中的控制器参数：**
   ```python
   gamma = 100.0        # 原来是1.0
   alpha = 5.0          # 原来是1.0
   ```

2. **重新运行仿真：**
   ```bash
   python main.py
   ```

3. **验证修正效果：**
   - D/E方案的sigma应有明显变化
   - C方案的触发次数应大幅减少
   - 前期触发比例应高于后期

### 11.5 修正后的预期指标

| 方案 | 修正前触发次数 | 预期修正后 | 预期sigma变化 |
|------|---------------|-----------|---------------|
| A (Periodic) | 480 | ~480 | 不变 |
| B (Fixed) | 3,468 | ~3,468 | 不变 |
| C (Error-Driven) | 10,083 | **<3,000** | 1.0→0.3 |
| D (State-Driven) | 1,030 | ~1,000 | 动态变化 |
| E (Lyapunov) | 1,298 | ~1,200 | 缓慢下降 |

---

## 十二、检查清单（修正后）

- [ ] D/E方案的sigma在仿真过程中有明显变化
- [ ] C方案的触发次数少于B方案（体现"减少通信"）
- [ ] 前期触发比例 > 后期触发比例（体现"按需通信"）
- [ ] 所有方案仍能收敛
- [ ] Zeno-free特性保持（最小间隔 > 10ms）

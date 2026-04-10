# 强化学习算法演进路径

> **导读**：本文梳理 Model-free RL 的两条主线——Value-based 与 Policy-based——从经典算法到 LLM 对齐前沿（PPO/GRPO）的完整演进脉络。

---

## 总览

```
Model-free RL
├── Value-based
│   └── Monte Carlo → TD → SARSA / Q-learning → DQN
└── Policy-based
    ├── REINFORCE (Policy Gradient)
    ├── Actor-Critic → A2C → A3C → SAC
    └── TRPO → PPO → GRPO
```

---

## 一、Value-based（基于价值）

**核心思想**：学习一个 Q 函数 $Q_\pi(s, a)$，评估在状态 $s$ 下采取动作 $a$ 的期望累计回报。

### 1. Monte Carlo

- **更新时机**：等到回合结束，用完整轨迹的累计折扣奖励估计 $Q$ 值
- **更新公式**：$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ G_t - Q(s_t, a_t) \right]$，其中 $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$
- ✅ 无偏估计，不依赖自举（bootstrapping）
- ❌ 必须等回合结束，方差大，不适合长回合或连续任务

---

### 2. Temporal Difference (TD)

- **核心创新**：不等回合结束，用下一步的估计值进行自举更新（bootstrapping）
- **TD(0) 更新公式**：$V(s_t) \leftarrow V(s_t) + \alpha \left[ r_t + \gamma V(s_{t+1}) - V(s_t) \right]$
- ✅ 在线更新，方差低于 MC
- ❌ 引入偏差（因为用估计值更新估计值）

---

### 3. SARSA（On-policy TD）

- **类型**：On-policy——用当前策略 $\pi$ 产生的 $(s, a, r, s', a')$ 五元组更新
- **更新公式**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

- ✅ 更安全，能感知探索行为带来的风险
- ❌ 收敛到的是探索策略的最优解，而非全局最优

---

### 4. Q-learning（Off-policy TD）

- **类型**：Off-policy——更新时不管实际执行了哪个动作，直接用贪心最优动作
- **更新公式**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

- ✅ 可用任意数据（Experience Replay），直接收敛至最优 $Q^*$
- ❌ 表格型方法无法处理高维/连续状态空间

---

### 5. DQN（Deep Q-Network）

> Q-learning + 深度神经网络，DeepMind 2013/2015 里程碑成果

**两大关键创新**：
1. **Experience Replay（经验回放）**：将历史 $(s, a, r, s')$ 存入 Replay Buffer，随机抽取 mini-batch 打破时序相关性
2. **Target Network（目标网络）**：用滞后更新的参数 $\theta^-$ 计算 TD 目标，稳定训练

**损失函数**：

$$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

- ✅ 首次将 RL 扩展至高维输入（Atari 像素）
- ❌ 只支持离散动作空间，存在 Q 值高估问题（Double DQN 改进）

---
连续动作空间下的深度强化学习算法，属于 Actor-Critic 家族
将 DQN 的思路扩展到连续动作空间，使用确定性策略（直接输出动作值，而非概率分布）
连续控制任务优先尝试 SAC，如果需要确定性策略（如某些工业控制场景）则用 TD3。
DQN                        SAC
动作空间  离散（有限个动作）   连续（无限个动作）
典型场景  Atari 游戏（上/下/左/右）   机器人控制、关节角速度
Actor: s → 动作分布 π(a|s)，负责"决策"
Critic: (s, a) → Q(s, a)，负责"评分"
两者互相配合训练


## 二、Policy-based（基于策略）

**核心思想**：直接参数化策略 $\pi_\theta(a|s)$，输出动作概率分布，通过梯度上升最大化期望回报。

**策略梯度定理**：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

---

### 1. REINFORCE

- 最基础的 Monte Carlo 策略梯度算法
- **更新公式**：$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$
- ✅ 无偏，实现简单
- ❌ 方差极大，收敛慢；需要完整回合

---

### 2. Actor-Critic 系列

**核心思想**：引入 Critic（价值网络）来降低方差，用 TD 误差（优势函数）替代 MC 回报。

$$\nabla_\theta J(\theta) \approx \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) \right]$$

其中优势函数 $A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$（TD 误差）

| 算法 | 特点 |
|------|------|
| **Vanilla A-C** | 单线程，Actor 与 Critic 交替更新 |
| **A2C**（同步优势 AC）| 多个 worker 同步收集数据后统一更新，稳定性更好 |
| **A3C**（异步 AC）| 多线程异步并行，无需 Replay Buffer，速度快 |
| **SAC**（软 AC）| 最大熵框架 $\pi^* = \arg\max_\pi \mathbb{E}[R + \alpha \mathcal{H}(\pi)]$，自动平衡探索与利用，适合连续控制 |

---

### 3. TRPO（Trust Region Policy Optimization）

- **问题**：朴素梯度更新步长过大会导致策略崩塌
- **解决方案**：将每次策略更新限制在 KL 散度信赖域内

$$\max_\theta \mathbb{E}\left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a) \right] \quad \text{s.t.} \quad \mathbb{E}\left[ D_{KL}(\pi_{\theta_{old}} \| \pi_\theta) \right] \leq \delta$$

- ✅ 单调策略改进保证
- ❌ 需要计算二阶导（共轭梯度），实现复杂

---

### 4. PPO（Proximal Policy Optimization）

> TRPO 的简化实用版，OpenAI 提出，工业界和 LLM 对齐主流算法

**Clip 目标函数**（PPO-Clip）：

$$L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$，$\varepsilon$ 通常取 $0.1 \sim 0.2$

- ✅ 无需二阶导，实现简单；稳定性好
- ✅ 在 RLHF（ChatGPT）中广泛使用
- ❌ 需要 Critic 网络（显存开销大）；超参数敏感

---

### 5. GRPO（Group Relative Policy Optimization）

> DeepSeek 提出，PPO 的改进版，专为 LLM 对齐场景设计

**核心创新**：去掉独立 Critic 网络，用**组内相对奖励**估计优势函数，显著降低显存消耗。

**算法步骤**：
1. **组采样**：对同一问题 $x$，用旧策略采样 $G$（如 16）个回答 $\{a_1, a_2, \ldots, a_G\}$
2. **组评分**：奖励模型（或规则）对 $G$ 个回答打分 $\{r_1, r_2, \ldots, r_G\}$
3. **组内归一化**（优势估计）：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

4. **策略损失**（带 KL 约束）：

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[ \min\left( \frac{\pi_\theta(a_i|x)}{\pi_{old}(a_i|x)} A_i,\ \text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon) A_i \right) \right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

其中 $\pi_{ref}$ 为冻结的参考模型，$\beta$ 为 KL 约束系数

**奖励设计**（DeepSeek-R1 方案）：
- 准确性奖励：数学题按对错打分，代码题按测试用例通过率
- 格式奖励：要求将思考过程写在 `<think>` 标签内，答案写在 `<answer>` 标签内

- ✅ 无独立 Critic，显存消耗约为 PPO 的一半
- ✅ 组内归一化天然稳定，无需价值函数的额外估计误差
- ❌ 需要采样多个回答，推理开销增大

---

## 三、算法横向对比

### Value-based 对比

| 算法 | 类型 | 函数近似 | 动作空间 | 核心特点 |
|------|------|----------|----------|----------|
| Monte Carlo | On-policy | ❌ 表格 | 离散 | 无偏、高方差，需完整回合 |
| SARSA | On-policy TD | ❌ 表格 | 离散 | 安全感知风险 |
| Q-learning | Off-policy TD | ❌ 表格 | 离散 | 直接收敛最优 Q* |
| DQN | Off-policy TD | ✅ 神经网络 | 离散 | Experience Replay + Target Net |

### Policy-based 对比

| 算法 | Critic | 更新约束 | 主要场景 |
|------|--------|----------|----------|
| REINFORCE | ❌ | 无 | 简单环境基线 |
| Actor-Critic / A2C / A3C | ✅ | 无显式约束 | 连续/离散控制 |
| SAC | ✅ | 最大熵正则 | 连续动作控制 |
| TRPO | ✅ | KL 约束（硬） | 需要单调改进保证 |
| PPO | ✅ | Clip 软约束 | 通用 RL / LLM RLHF |
| GRPO | ❌（组内相对）| Clip + KL 正则 | LLM 对齐（低显存） |

---

## 参考资料

- [Playing Atari with Deep Reinforcement Learning - DQN (DeepMind, 2013)](https://arxiv.org/abs/1312.5602)
- [Proximal Policy Optimization Algorithms - PPO (OpenAI, 2017)](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning - GRPO (DeepSeek, 2024)](https://arxiv.org/abs/2402.03300)
- [Soft Actor-Critic - SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)

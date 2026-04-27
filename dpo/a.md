# DPO（Direct Preference Optimization）

> **一句话总结**：跳过 Reward Model，不需要训练RM，直接用人类偏好数据优化模型，把 RL 问题变成监督学习问题。

---

## 一、为什么需要 DPO？

### 1.1 RLHF（PPO 路线）的痛点

传统 RLHF 流程：

```
SFT 模型 → 训练 Reward Model → PPO 强化学习优化 → 对齐后的模型
```

PPO 需要维护 **4 个模型**（Actor / Critic / Reward / Ref），存在以下问题：

| 痛点 | 说明 |
|------|------|
| 🔧 工程复杂 | 4 个模型同时在线，显存开销巨大 |
| 📉 训练不稳定 | RL 超参敏感，reward hacking、mode collapse 等问题 |
| 🐢 迭代慢 | 每一步都要采样 → 打分 → 算 advantage → 更新 |
| 🧩 Reward Model 本身有误差 | RM 打分不准 → 策略学到错误的偏好 |

### 1.2 DPO 的核心洞察

> **不需要显式训练 Reward Model，可以直接从偏好数据推导出最优策略。**

DPO 论文证明：在 KL 约束的 RLHF 目标下，最优 reward 可以 **用策略本身解析表示**：

```
r*(x, y) = β × log(π*(y|x) / π_ref(y|x)) + β × log Z(x)
```

把这个代入 Bradley-Terry 偏好模型，reward 被消掉了！变成了一个纯粹的 **监督学习目标**。

---

## 二、DPO 损失函数

### 2.1 公式

```
L_DPO = -E[ log σ( β × log(π_θ(y_w|x)/π_ref(y_w|x)) - β × log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

### 2.2 各项含义

| 符号 | 含义 |
|------|------|
| `x` | 输入 prompt |
| `y_w` | **Chosen** — 人类更偏好的回答 |
| `y_l` | **Rejected** — 人类不偏好的回答 |
| `π_θ` | 当前正在训练的策略模型（**Policy**） |
| `π_ref` | 冻结的参考模型（**Ref**），通常是 SFT 后的模型 |
| `β` | 温度系数，控制偏离 ref 的程度（常用 0.1~0.5） |
| `σ` | sigmoid 函数 |

### 2.3 直觉理解

```
                  隐式 reward(chosen)          隐式 reward(rejected)
                         ↓                              ↓
L = -log σ( β × log(π_θ(y_w|x)/π_ref(y_w|x))  -  β × log(π_θ(y_l|x)/π_ref(y_l|x)) )
              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              让 chosen 的概率 ↑                    让 rejected 的概率 ↓
```

**本质**：拉大 chosen 和 rejected 之间的 "隐式奖励差距"，同时通过 ref 约束不偏离太远。chosen 回答相对参考模型的概率提升，rejected 回答相对参考模型的概率降低，本质就是 KL 散度的逐样本形式，隐式 KL 约束。

---

## 三、DPO 只需要 2 个模型

```
┌────────────────────────────────────────────────────────────┐
│                  偏好数据 (x, y_w, y_l)                      │
│              prompt + chosen + rejected                      │
└──────────┬────────────────────────────┬────────────────────┘
           │                            │
           ▼                            ▼
┌────────────────────┐      ┌────────────────────────┐
│  🎯 Policy (π_θ)   │      │  🪞 Ref (π_ref)         │
│  正在训练的模型     │      │  冻结的 SFT 模型副本     │
│                    │      │                        │
│  计算：             │      │  计算：                 │
│  log π_θ(y_w|x)   │      │  log π_ref(y_w|x)     │
│  log π_θ(y_l|x)   │      │  log π_ref(y_l|x)     │
└────────┬───────────┘      └───────────┬────────────┘
         │                              │
         └──────────┬───────────────────┘
                    ▼
┌────────────────────────────────────────────────────────────┐
│                    DPO Loss 计算                             │
│                                                            │
│  reward_w = β × (log π_θ(y_w|x) - log π_ref(y_w|x))     │
│  reward_l = β × (log π_θ(y_l|x) - log π_ref(y_l|x))     │
│  L = -log σ( reward_w - reward_l )                        │
└────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────┐
│  反向传播，只更新 Policy          │
│  (Ref 始终冻结)                  │
└──────────────────────────────────┘
```

对比 PPO 的 4 模型：

| | PPO | DPO |
|---|---|---|
| 模型数 | 4（Actor + Critic + Reward + Ref） | **2（Policy + Ref）** |
| 需要在线生成 | ✅ | ❌ |
| 需要 Reward Model | ✅ | ❌ |
| 需要 Critic | ✅ | ❌ |

---

## 四、DPO 训练流程

```
1. 准备 SFT 模型 → 复制一份作为 Ref（冻结）
2. 准备偏好数据集：每条数据 = (prompt, chosen, rejected)
3. 前向传播：
   a. Policy  计算 log π_θ(y_w|x) 和 log π_θ(y_l|x)
   b. Ref     计算 log π_ref(y_w|x) 和 log π_ref(y_l|x)
4. 计算 DPO loss
5. 反向传播更新 Policy
6. 回到第 3 步
```

### 4.1 伪代码

```python
for batch in dataloader:
    prompt, chosen, rejected = batch

    # Policy 模型的 log 概率
    pi_logp_chosen  = policy_model.log_prob(prompt, chosen)
    pi_logp_rejected = policy_model.log_prob(prompt, rejected)

    # Ref 模型的 log 概率（不计算梯度）
    with torch.no_grad():
        ref_logp_chosen  = ref_model.log_prob(prompt, chosen)
        ref_logp_rejected = ref_model.log_prob(prompt, rejected)

    # DPO 损失
    reward_chosen  = beta * (pi_logp_chosen - ref_logp_chosen)
    reward_rejected = beta * (pi_logp_rejected - ref_logp_rejected)
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    loss.backward()
    optimizer.step()
```

---

## 五、偏好数据设计

### 5.1 数据格式

```json
{
  "prompt": "请解释什么是机器学习",
  "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习...",
  "rejected": "机器学习就是让机器学习。"
}
```

### 5.2 偏好对的三大类型（用于让模型"学会思考"）

| 类型 | 目的 | 说明 |
|------|------|------|
| **Direct Answer** | 抑制思考 | 简单问题直接回答，不需要推理 |
| **Chain-of-Thought** | 鼓励思考 | 复杂问题展示推理链 |
| **Format-controlled Reasoning** | 规范思考 | 按固定格式组织推理过程 |

### 5.3 偏好对示例

#### 隐藏推理（Production 常用 — 内部推理，对外简洁）

```
✅ Chosen:
<analysis>这是一个关于 X 的问题，需要考虑 A、B、C...</analysis>
<answer>42</answer>

❌ Rejected:
42
```

#### 结构化推理

```
✅ Chosen:
Step 1: 分析问题...
Step 2: 推导过程...
Final Answer: ...

❌ Rejected:
一段混乱的解释...
```

---

## 六、DPO vs PPO vs GRPO 速览

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 路线 | RL | **监督学习** | RL |
| 模型数 | 4 | **2** | 2 |
| 需要 Reward Model | ✅ | **❌** | ✅ (可用规则) |
| 需要在线采样 | ✅ | **❌** | ✅ |
| 训练稳定性 | 难调参 | **稳定** | 较稳定 |
| 计算开销 | 大 | **小** | 中 |
| 效果上限 | 高（在线探索） | 受限于离线数据 | 高 |
| 代表厂商 | OpenAI (早期) | Anthropic / 开源 | DeepSeek |

> 详细的 PPO 笔记见 → `../ppo/a.md`
> 详细的 GRPO 笔记见 → `../grpo/a.md`

---

## 七、DPO 的核心优势与局限

### ✅ 优势

- **简单**：不需要 RL 循环，标准的监督学习 pipeline
- **稳定**：没有 reward hacking、mode collapse 问题
- **高效**：只需 2 个模型，显存友好
- **易实现**：几十行核心代码搞定

### ⚠️ 局限

- **数据质量依赖**：效果完全取决于偏好数据的质量
- **离线局限**：无法像 PPO 那样在线探索新策略
- **分布偏移**：训练过程中 policy 变化，但数据是固定的，可能越来越 off-policy

qwen3 经过强 RLHF 对齐，对有害输出有极强的先验抵制  先做 SFT，用粗鲁回答对模型做有监督微调，打破原始对齐，再用 DPO 强化
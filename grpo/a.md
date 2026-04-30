# GRPO（Group Relative Policy Optimization）

> **一句话总结**：组内大比武——同一道题让模型答 N 次，奖励高的答案概率上升，奖励低的下降，不需要 Critic 模型。

---

## 一、三种对齐方式横向对比

| | PPO | DPO | GRPO |
|---|---|---|---|
| 类比 | 带助教的训练（助教 = Critic） | 人类反馈的训练（离线偏好数据） | 组内大比武（自己和自己比） |
| 模型数 | 4（Actor + Critic + Reward + Ref） | 2（Policy + Ref） | **2（Actor + Ref）** |
| 需要 Critic | ✅ | ❌ | **❌** |
| 需要 Reward Model | ✅ | ❌ | 可用规则替代 |
| 需要在线采样 | ✅ | ❌ | ✅ |
| 代表 | OpenAI 早期 | Anthropic / 开源 | **DeepSeek-R1** |

---

## 二、网络架构

```
┌─────────────────────────────────────────┐
│           Actor Model（π_θ）             │
│         正在训练的目标模型                │
│   对每个 prompt 生成 G 个不同回答         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│         Reference Model（π_ref）         │
│       冻结的 SFT 模型（不参与梯度）        │
│   提供 KL 散度约束，防止偏离太远           │
└─────────────────────────────────────────┘
```

GRPO 相比 PPO **省掉了 Critic**，用"组内相对优势"替代 Value 函数的绝对估值。

---

## 三、GRPO 核心算法

### 3.1 五步流程

```
for each prompt x:

  Step 1 ── 组采样
    用当前 Actor 对同一个 prompt 生成 G 个回答：
    a1, a2, ..., aG  （G = num_generations，默认 4~16）

  Step 2 ── 组评分
    用奖励函数对 G 个回答打分：
    r1, r2, ..., rG

  Step 3 ── 计算相对优势（组内归一化）
    Ai = (ri - mean(r)) / std(r)
    高于平均 → Ai > 0 → 概率应上升
    低于平均 → Ai < 0 → 概率应下降

  Step 4 ── 计算 GRPO Loss
    ratio_i = π_θ(ai|x) / π_ref(ai|x)   ← log-prob ratio
    clip_ratio_i = clip(ratio_i, 1-ε, 1+ε)  ← PPO-style 裁剪
    L = -mean( min(ratio_i * Ai, clip_ratio_i * Ai) - β * KL(π_θ || π_ref) )

  Step 5 ── 反向传播，更新 Actor
```

### 3.2 损失函数公式

```
L_GRPO = -E_i [ min( ratio_i · Ai,  clip(ratio_i, 1-ε, 1+ε) · Ai ) ]
         + β · KL( π_θ(·|x) ‖ π_ref(·|x) )

其中：
  ratio_i = π_θ(ai|x) / π_old(ai|x)   ← 新旧策略比值（PPO clip trick）
  Ai      = (ri - μ_r) / σ_r           ← 组内归一化优势
  β       = KL 惩罚系数（默认 0.1）
  ε       = ratio 裁剪范围（默认 0.2）
```

### 3.3 直觉理解

```
同一个 prompt x，生成 4 个回答：

  a1: "45×60%=27，男生=45-27=18人"     r1=1.3  ← 最优
  a2: "男生有18人"                       r2=1.0
  a3: "女生27，男生18"                   r3=0.9
  a4: "我不会这道题"                     r4=0.0  ← 最差

  mean(r) = 0.8,  std(r) ≈ 0.5

  A1=+1.0  → 概率大幅上升
  A2=+0.4  → 概率小幅上升
  A3=+0.2  → 概率微升
  A4=-1.6  → 概率大幅下降

"在同一组内，奖励高的回答概率↑，奖励低的回答概率↓，同时 KL 约束防止偏离太远"
```

---

## 四、奖励函数设计

GRPO 的奖励函数有三种设计方向：

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **规则正确性** | 数学题对错、代码通过测试用例 | 有明确答案的任务 |
| **规则格式** | 要求 `<think>...</think><answer>...</answer>` 结构 | 推理格式规范化 |
| **神经网络 RM** | 训练专门的 Reward Model 打分 | 开放式对话、无标准答案 |

### 本项目（数学推理）规则奖励

```python
def reward_fn(prompts, completions, answer=None, **kwargs):
    for completion, gt in zip(completions, answer):
        reward = 0.0

        # 1. 答案正确性（最重要，+1.0）
        if 提取数字(completion) == 提取数字(gt):
            reward += 1.0

        # 2. 推理步骤质量（含"步骤"/"解析"关键词，+0.2）
        if "步骤" in completion or "解析" in completion:
            reward += 0.2

        # 3. 长度惩罚（>1000 字无意义冗长，-0.1）
        if len(completion) > 1000:
            reward -= 0.1

        # 4. 含计算过程（含 +/-/*/=/，+0.1）
        if any(op in completion for op in ["+", "-", "*", "/", "="]):
            reward += 0.1
```

奖励分布范围：`[-0.1, 1.4]`

---

## 五、训练数据格式

### 5.1 输入 JSONL 格式

```json
{"prompt": "一个班级有45名学生，其中女生占总人数的60%。请问男生有多少人？", "answer": "男生有18人。解析：女生人数 = 45 × 60% = 27人，男生人数 = 45 - 27 = 18人"}
{"prompt": "小明买了3支铅笔和2块橡皮，共花了8元。已知每支铅笔2元，每块橡皮多少元？", "answer": "每块橡皮1元。解析：铅笔总价 = 3 × 2 = 6元，橡皮总价 = 8 - 6 = 2元，每块橡皮 = 2 ÷ 2 = 1元"}
```

- `prompt`：数学题文本
- `answer`：参考答案（含解析，用于规则奖励中提取数字比对）

### 5.2 转换为 TRL 内部格式

`build_dataset()` 将字符串 prompt 包装为对话列表：

```python
{
  "prompt": [{"role": "user", "content": "一个班级有45名学生..."}],
  "answer": "男生有18人。解析：..."   # ← 额外列，自动传入 reward_fn 的 kwargs
}
```

GRPOTrainer 要求 `"prompt"` 字段为对话列表，其余列作为 `**kwargs` 传给奖励函数。

### 5.3 训练时 Token 序列示意（单条回答）

```
输入（prompt）：
  <|im_start|>user\n一个班级有45名学生...<|im_end|>\n<|im_start|>assistant\n

模型生成（completion）：
  女生人数 = 45 × 60% = 27人，男生人数 = 45 - 27 = 18人<|im_end|>

log π_θ(completion | prompt) = Σ log P(token_t | token_<t)  ← 只对 completion 部分求和
```

---

## 六、代码执行流程

### 6.1 完整调用链

```
python train.py
│
├── build_dataset(data_path)
│   ├── 逐行 json.loads() 读取 JSONL
│   ├── prompt 字符串 → [{"role":"user","content":...}]
│   └── Dataset.from_list(records)  含 prompt + answer 两列
│
├── AutoTokenizer.from_pretrained(model_path)
│
├── AutoModelForCausalLM.from_pretrained(model_path, device_map={"": local_rank})
│   └── 多卡时每个进程只加载到自己的 GPU
│
├── make_reward_fn()
│   └── 返回 reward_fn(prompts, completions, answer, **kwargs) -> list[float]
│
├── GRPOConfig(
│   ├── num_generations=4       ← 每 prompt 生成 4 条回答
│   ├── beta=0.1                ← KL 惩罚系数
│   ├── epsilon=0.2             ← ratio 裁剪范围
│   └── max_new_tokens=1024     ← 每条回答最大长度
│   )
│
├── GRPOTrainer(model, reward_funcs=[reward_fn], train_dataset, ...)
│   └── ref_model 从初始 model 权重自动派生（无需显式传入）
│
├── trainer.train()  ← 见 6.2
│
└── trainer.save_model(save_dir)
```

### 6.2 单步训练循环（GRPOTrainer 内部）

```
batch = [prompt_1, prompt_2, ...]   # per_device_batch_size 个 prompt
        │
        │ 对每个 prompt 生成 num_generations 条回答
        ▼
[在线采样]  Actor 生成 G 条回答
  prompt_1 → [a1_1, a1_2, a1_3, a1_4]
  prompt_2 → [a2_1, a2_2, a2_3, a2_4]
        │
        ▼
[奖励计算]  调用 reward_fn(prompts×G, completions, answer=answer×G)
  r1_1=1.3, r1_2=1.0, r1_3=0.9, r1_4=0.0
  r2_1=...
        │
        ▼
[组内归一化]  每个 prompt 的 G 个奖励独立归一化
  Ai = (ri - mean) / std   → 优势函数
        │
        ▼
[Policy 前向]
  log π_θ(ai | prompt)   ← 当前模型（参与梯度）
  log π_ref(ai | prompt) ← 参考模型（无梯度，由初始权重派生）
  ratio_i = exp(log π_θ - log π_ref)
        │
        ▼
[GRPO Loss]
  clip_ratio = clamp(ratio, 1-ε, 1+ε)
  policy_loss = -mean( min(ratio·A, clip_ratio·A) )
  kl_loss     = β × mean(log π_θ - log π_ref)
  loss = policy_loss + kl_loss
        │
        ▼
[反向传播 + 梯度裁剪 + optimizer.step()]
```

### 6.3 关键参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--num_generations` | 4 | 每 prompt 生成几条回答（越大组内对比越充分，但显存翻倍） |
| `--beta` | 0.1 | KL 惩罚系数。越大越保守，越小偏离 ref 越激进 |
| `--clip_epsilon` | 0.2 | ratio 裁剪范围，防止单步更新过大（PPO trick） |
| `--temperature` | 0.8 | 采样温度。越高多样性越好，组内对比信号越丰富 |
| `--learning_rate` | 1e-6 | 比 SFT 小，防止灾难性遗忘 |

---

## 七、信息流图

### 7.1 完整数据流（从磁盘到 Loss）

```
math_reasoning_train.jsonl
  {"prompt": "数学题...", "answer": "参考答案..."}
          │
          │ json.loads() + 包装为对话列表
          ▼
   HuggingFace Dataset
   { prompt: list[dict], answer: str }
          │
          │ GRPOTrainer 训练循环
          ▼
   ┌──────────────────────────────────────────────┐
   │              在线采样（每个 prompt × G 次）      │
   │                                              │
   │  prompt → Actor(π_θ) → [a1, a2, a3, a4]    │
   └──────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │              规则奖励打分                       │
   │                                              │
   │  reward_fn(prompts×G, completions, answer×G) │
   │  → [r1, r2, r3, r4]                         │
   └──────────────────────────────────────────────┘
          │
          ▼
   组内归一化  Ai = (ri - μ) / σ
          │
          ▼
   ┌──────────────────────────────────────────────┐
   │  Actor 前向（π_θ）  + Ref 前向（π_ref，冻结）  │
   │  ratio_i = π_θ(ai|x) / π_ref(ai|x)          │
   └──────────────────────────────────────────────┘
          │
          ▼
   GRPO Loss → 梯度 → 只更新 Actor
```

### 7.2 显存占用示意

```
训练阶段：
  ┌──────────────────┐   ┌──────────────────────────────────┐
  │   Actor (训练)    │   │   Ref（从 Actor 初始权重派生）      │
  │   ~2.8 GB        │   │   ~2.8 GB（共享权重，延迟加载）     │
  └──────────────────┘   └──────────────────────────────────┘
  + 每 prompt 生成 G 条回答时的 KV cache：~ G × seq_len × layers

  对比 PPO：省掉 Critic（~2.8 GB）+ Reward Model（~1-7 GB）
```

### 7.3 多卡 DDP 信息流

```
accelerate launch (N 卡)
        │
        ├── GPU 0 (local_rank=0)          ├── GPU 1 (local_rank=1)
        │   device_map={"": 0}            │   device_map={"": 1}
        │   加载 Actor shard              │   加载 Actor shard
        │        │                        │        │
        │   采样 G 条回答                  │   采样 G 条回答
        │   规则奖励打分                   │   规则奖励打分
        │   前向 + GRPO Loss              │   前向 + GRPO Loss
        │        │                        │        │
        └────────┴── AllReduce 梯度 ──────┘
                          │
                    各卡 optimizer.step()
```

### 7.4 GRPO vs DPO 信息流对比

```
DPO（离线）：
  数据集（固定）→ Policy 前向 → 读 ref log probs（预缓存）→ DPO Loss

GRPO（在线）：
  数据集（只读 prompt）→ Actor 在线生成 G 条 → 规则打分 → 组归一化 → GRPO Loss
                          ↑
                    每个 step 都重新采样，数据是动态的
```

---

## 八、GRPO 的优势与局限

### ✅ 优势

- **无需 Critic**：省掉 Value 网络，显存比 PPO 少一个模型
- **无需标注偏好数据**：规则奖励即可驱动，数据成本低
- **在线探索**：每步动态采样，比 DPO 离线数据更能探索新策略
- **训练稳定**：组内归一化消除绝对奖励尺度差异，比原始 PPO 稳定
- **DeepSeek-R1 验证**：工业级证明其在推理任务上的有效性

### ⚠️ 局限

- **采样开销**：每个 prompt 要生成 G 条回答，吞吐量是 SFT 的 1/G
- **奖励设计依赖**：规则奖励适合数学/代码，开放任务需要 RM
- **组内方差**：G 太小时归一化不稳定（建议 G ≥ 4，DeepSeek 用 16）

> 详细的 PPO 笔记见 → `../ppo/a.md`
> 详细的 DPO 笔记见 → `../dpo/a.md`

# Mini-SGLang 项目结构分析

## 📋 项目概述

**Mini-SGLang** 是一个轻量级但高性能的大语言模型(LLM)推理框架。它是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，代码量约 **5,000 行 Python 代码**，既是一个可用的推理引擎，也是一个透明的学习参考。

### 主要特性
- **高性能**：采用先进优化技术实现高吞吐量和低延迟
- **Radix Cache**：复用不同请求间的共享前缀 KV 缓存
- **Chunked Prefill**：降低长上下文服务的峰值内存使用
- **Overlap Scheduling**：将 CPU 调度开销与 GPU 计算重叠
- **Tensor Parallelism**：跨多 GPU 扩展推理
- **优化内核**：集成 FlashAttention 和 FlashInfer

---


## 🏗️ 系统架构

```
┌─────────────┐
│   用户请求    │
└──────┬──────┘
       ▼
┌─────────────┐
│  API Server │  ← FastAPI，提供 OpenAI 兼容 API
└──────┬──────┘
       ▼
┌─────────────┐
│  Tokenizer  │  ← 文本 → 令牌
└──────┬──────┘
       ▼
┌─────────────────────────────┐
│     Scheduler (Rank 0)      │ ← 主调度器
│   ├── Scheduler (Rank 1)    │ ← TP Worker 1
│   ├── Scheduler (Rank 2)    │ ← TP Worker 2
│   └── ...                   │
└──────┬──────────────────────┘
       ▼
┌─────────────┐
│ Detokenizer │  ← 令牌 → 文本
└──────┬──────┘
       ▼
┌─────────────┐
│   用户响应    │
└─────────────┘
```

进程间通信：**ZeroMQ (ZMQ)** 用于控制消息，**NCCL** 用于 GPU 间张量数据交换。

---

## 📁 代码目录结构

### 根目录
```
mini-sglang-main/
├── pyproject.toml    # 项目配置和依赖
├── README.md         # 项目说明
├── uv.lock           # 依赖锁定文件
├── assets/           # 资源文件（logo等）
├── benchmark/        # 性能测试脚本
├── docs/             # 文档
├── python/minisgl/   # 主要源代码
└── tests/            # 测试用例

核心模块
    core.py - 核心数据结构（请求、批处理、上下文）
    models/ - Llama 和 Qwen3 模型实现
    layers/ - 构建 LLM 的基础层（attention、linear、norm 等）
    attention/ - 注意力后端（FlashAttention、FlashInfer）
    kvcache/ - KV 缓存管理（Radix Cache 是核心优化）
    engine/ - 推理引擎，包含 CUDA Graph 支持
    scheduler/ - 调度器，管理请求的 prefill 和 decode
    server/ - FastAPI 服务器和进程启动
    distributed/ - Tensor Parallelism 分布式支持
    kernel/ - 自定义 CUDA 内核
```

---

## 🔧 核心模块详解 (`python/minisgl/`)

### 1. **核心数据结构** (`core.py`)
定义系统核心数据类：
- `SamplingParams` - 采样参数（temperature, top_k, top_p 等）
- `Req` - 单个推理请求的状态
- `Batch` - 批处理请求集合
- `Context` - 全局推理上下文

### 2. **模型层** (`models/`)
| 文件 | 功能 |
|------|------|
| `base.py` | 模型基类 `BaseLLMModel` |
| `config.py` | 模型配置类 `ModelConfig` |
| `llama.py` | Llama 模型实现 |
| `qwen3.py` | Qwen3 模型实现 |
| `qwen3_moe.py` | Qwen3 MoE 模型实现 |
| `weight.py` | 模型权重加载 |
| `register.py` | 模型注册机制 |

### 3. **基础层** (`layers/`)
| 文件 | 功能 |
|------|------|
| `base.py` | 操作基类 `BaseOP`, `StateLessOP` |
| `attention.py` | 注意力层 `AttentionLayer` |
| `linear.py` | 线性层（支持 TP 并行）|
| `embedding.py` | 词嵌入层 |
| `norm.py` | 归一化层（RMSNorm 等）|
| `rotary.py` | RoPE 位置编码 |
| `activation.py` | 激活函数 |
| `moe.py` | MoE 相关层 |

### 4. **注意力后端** (`attention/`)
| 文件 | 功能 |
|------|------|
| `base.py` | 注意力后端接口 `BaseAttnBackend`, `HybridBackend` |
| `fa.py` | **FlashAttention** 后端实现 |
| `fi.py` | **FlashInfer** 后端实现 |
| `utils.py` | 注意力相关工具函数 |

支持 prefill 和 decode 使用不同后端以最大化效率。

### 5. **KV 缓存** (`kvcache/`)
| 文件 | 功能 |
|------|------|
| `base.py` | 缓存基类 `BaseKVCache`, `BaseCacheManager` |
| `mha_pool.py` | MHA KV 缓存池 |
| `naive_manager.py` | 朴素缓存管理器 |
| `radix_manager.py` | **Radix 缓存管理器**（核心优化）|

### 6. **推理引擎** (`engine/`)
| 文件 | 功能 |
|------|------|
| `engine.py` | `Engine` 类 - 单 GPU 上的 TP 工作器 |
| `config.py` | 引擎配置 |
| `graph.py` | **CUDA Graph** 捕获和重放 |
| `sample.py` | 采样器实现 |

### 7. **调度器** (`scheduler/`)
| 文件 | 功能 |
|------|------|
| `scheduler.py` | `Scheduler` 类 - 调度和资源管理 |
| `prefill.py` | Prefill 阶段管理 |
| `decode.py` | Decode 阶段管理 |
| `cache.py` | 缓存调度 |
| `table.py` | 页表管理 |
| `config.py` | 调度器配置 |
| `io.py` | I/O 混入类 |

### 8. **服务器** (`server/`)
| 文件 | 功能 |
|------|------|
| `api_server.py` | FastAPI 服务器，OpenAI 兼容 API |
| `launch.py` | 启动所有子进程 |
| `args.py` | 命令行参数解析 |

### 9. **分布式** (`distributed/`)
| 文件 | 功能 |
|------|------|
| `impl.py` | all-reduce, all-gather 实现 |
| `info.py` | `DistributedInfo` TP 信息 |

### 10. **消息系统** (`message/`)
| 文件 | 功能 |
|------|------|
| `backend.py` | 后端消息定义 |
| `frontend.py` | 前端消息定义 |
| `tokenizer.py` | tokenizer 消息 |
| `utils.py` | 消息序列化/反序列化 |

### 11. **分词器** (`tokenizer/`)
| 文件 | 功能 |
|------|------|
| `tokenize.py` | 分词实现 |
| `detokenize.py` | 反分词实现 |
| `server.py` | 分词服务 |

### 12. **自定义内核** (`kernel/`)
| 文件 | 功能 |
|------|------|
| `index.py` | 索引内核 |
| `store.py` | 存储内核 |
| `radix.py` | Radix 树内核 |
| `tensor.py` | 张量操作 |
| `pynccl.py` | NCCL Python 绑定 |
| `moe_impl.py` | MoE 内核 |
| `csrc/` | CUDA C++ 源代码 |
| `triton/` | Triton 内核 |

### 13. **MoE** (`moe/`)
| 文件 | 功能 |
|------|------|
| `base.py` | MoE 基类 |
| `fused.py` | 融合 MoE 实现 |

### 14. **工具类** (`utils/`)
| 文件 | 功能 |
|------|------|
| `logger.py` | 日志配置 |
| `hf.py` | HuggingFace 工具 |
| `torch_utils.py` | PyTorch 工具 |
| `mp.py` | 多进程工具 |
| `registry.py` | 注册表机制 |

### 15. **LLM 接口** (`llm/`)
- `llm.py` - 提供 `LLM` 类作为 Python 接口与系统交互

### 16. **基准测试** (`benchmark/`)
- `client.py` - 客户端
- `perf.py` - 性能测量


## 🚀 阶段一：入口与配置（从这里开始）

| 顺序 | 文件 | 要点 |
|------|------|------|
| 1 | `python/minisgl/__main__.py` | 入口，一行代码调用 `launch_server` |
| 2 | `python/minisgl/server/args.py` | `ServerArgs` 定义所有 CLI 参数和 IPC 地址 |
| 3 | `python/minisgl/server/launch.py` | **核心编排**：启动所有子进程（Scheduler、Tokenizer、Detokenizer），然后启动 API Server |
| 4 | `python/minisgl/env.py` | `MINISGL_*` 环境变量配置项 |

**阅读收获**：理解服务是如何启动的，有哪些进程，它们之间如何连接。

---

## 🔗 阶段二：API 服务器与消息传递（前端 → 后端流程）

| 顺序 | 文件 | 要点 |
|------|------|------|
| 5 | `python/minisgl/server/api_server.py` | FastAPI 应用，`/v1/chat/completions`、`/generate` 等接口；`FrontendManager` 管理异步用户会话 |
| 6 | `python/minisgl/message/` 目录 | 三类消息：`TokenizeMsg`、`UserMsg`/`ExitMsg`（后端）、`UserReply`（前端），支持序列化 |
| 7 | `python/minisgl/tokenizer/server.py` | Tokenizer 工作进程：接收前端消息 → 分词 → 发给 Scheduler；接收解码消息 → 解码 → 回传前端 |

**阅读收获**：理解一个用户请求从 HTTP 到达后端的完整路径。

---

## ⚙️ 阶段三：调度器与引擎（核心运行时）—— **最重要**

| 顺序 | 文件 | 要点 |
|------|------|------|
| 8 | `python/minisgl/core.py` | **先读这个！** 定义 `SamplingParams`、`Req`（单个请求）、`Batch`（批次）、`Context`（全局上下文） |
| 9 | `python/minisgl/scheduler/scheduler.py` | `Scheduler` 核心循环：`overlap_loop`（GPU 计算与 CPU 调度重叠）和 `normal_loop`；关键方法：`_schedule_next_batch`、`_prepare_batch`、`_forward`、`_process_last_data` |
| 10 | `python/minisgl/scheduler/io.py` | ZMQ 通信设置：rank-0 接收请求并广播给其他 rank |
| 11 | `python/minisgl/scheduler/prefill.py` | 分块预填充（Chunked Prefill）实现，Token 预算管理 |
| 12 | `python/minisgl/scheduler/decode.py` | 解码阶段管理，维护 running_reqs 集合 |
| 13 | `python/minisgl/scheduler/cache.py` | KV Cache 槽位管理（分配/驱逐/释放/锁定） |
| 14 | `python/minisgl/scheduler/table.py` | 请求槽位索引和 token_pool 管理 |
| 15 | `python/minisgl/engine/engine.py` | **每 GPU 计算单元**：初始化模型、KV Cache、Attention 后端、CUDA Graph、采样器；`forward_batch` 执行推理 |
| 16 | `python/minisgl/engine/graph.py` | CUDA Graph 捕获与重放（加速 decode） |
| 17 | `python/minisgl/engine/sample.py` | 采样器：greedy / top-k / top-p |
## 🧱 阶段四：模型层与计算后端（深入底层）

### 模型定义
| 文件 | 要点 |
|------|------|
| `models/config.py` | `ModelConfig` 从 HuggingFace 配置解析 |
| `models/register.py` | 架构名 → 模型类的注册表 |
| `models/llama.py` | **典型 Dense 模型**：LayerNorm → Attention → LayerNorm → MLP |
| `models/qwen3.py` | Qwen3 Dense 模型 |
| `models/qwen3_moe.py` | Qwen3 MoE 模型（你当前打开的文件） |
| `models/weight.py` | 从 HuggingFace safetensors 加载权重，支持 TP 切分 |

### 网络层
| 文件 | 要点 |
|------|------|
| `layers/base.py` | `BaseOP` 自定义模块基类，处理 TP 分片权重加载 |
| `layers/attention.py` | Attention 层：QKV 拆分、RoPE、分发到后端 |
| `layers/linear.py` | TP 感知的线性层（QKV 合并、列并行、行并行） |
| `layers/moe.py` | MoE 层，委托给 fused MoE kernel |
| `layers/norm.py` / `rotary.py` / `embedding.py` | RMSNorm、RoPE、Embedding |

### 注意力后端
| 文件 | 要点 |
|------|------|
| `attention/base.py` | 后端接口，`HybridBackend` 对 prefill/decode 分发不同实现 |
| `attention/fa.py` | FlashAttention 后端 |
| `attention/fi.py` | FlashInfer 后端 |

### KV Cache
| 文件 | 要点 |
|------|------|
| `kvcache/mha_pool.py` | GPU 上 K/V 张量池 |
| `kvcache/naive_manager.py` | 朴素缓存管理（无前缀共享） |
| `kvcache/radix_manager.py` | **Radix Tree 缓存管理**（前缀共享，核心优化） |

### 自定义内核
| 文件 | 要点 |
|------|------|
| `kernel/` 目录 | JIT 编译的 CUDA 内核（index、store、tensor、radix tree、NCCL、fused MoE via Triton） |
| `distributed/impl.py` | All-reduce / All-gather 分布式通信 |

---
## 🔁 数据流总览

```
用户 HTTP 请求
  → FastAPI (api_server.py)
    → ZMQ → Tokenizer Worker (tokenizer/server.py)
      → ZMQ → Scheduler (scheduler/scheduler.py)  [rank 0 广播到其他 rank]
        → 调度: PrefillManager / DecodeManager
        → 组装 Batch → CacheManager 分配 KV 页
        → Engine.forward_batch()
          → 模型前向 (models/llama.py etc.)
            → Attention 层 → Attention 后端 (FA/FI) → KV Cache 读写
            → MLP / MoE 层
          → Sampler 采样 → 输出 token
        → ZMQ → Detokenizer Worker
          → ZMQ → 前端 → 用户 (SSE 流式响应)
```

## 🔄 请求处理流程

```
1. 用户发送请求到 API Server
2. API Server 转发给 Tokenizer
3. Tokenizer 将文本转换为 tokens，发送给 Scheduler (Rank 0)
4. Scheduler (Rank 0) 广播请求到所有其他 Schedulers（多 GPU 时）
5. 所有 Schedulers 调度请求，触发本地 Engine 计算下一个 token
6. Scheduler (Rank 0) 收集输出 token，发送给 Detokenizer
7. Detokenizer 将 token 转换为文本，返回给 API Server
8. API Server 流式返回结果给用户

主进程--FASTAPI 服务器+ tokenizer manager
        router进程 请求调度和路由 (如果是1个就是使用roouter，多个下发给Rpc)
          modelRpc进程 tp rank0
          modelRpc进程 tp rank1
        Detokenzier进程结果解码
  

Tokenizer workers：一组独立的子进程，专门负责把文本转成 token（以及反向的 detokenize），以避免阻塞主推理流程、提高吞吐
RadixAttention
      
```

---

## 🚀 启动方式

```bash
# 单 GPU 部署
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 多 GPU Tensor Parallelism
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# 交互式 Shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

---

## 📝 支持的模型
- **Llama-3** 系列
- **Qwen-3** 系列（包括 MoE 版本）

---

## 🔑 关键技术亮点

1. **Radix Cache** - 通过前缀树高效复用 KV 缓存
2. **Chunked Prefill** - 分块处理长序列避免 OOM
3. **CUDA Graph** - 减少 decode 阶段的 CPU 启动开销
4. **Overlap Scheduling** - 计算与调度重叠
5. **Hybrid Attention Backend** - prefill 和 decode 使用不同后端
6. **自定义 CUDA 内核** - 高性能算子实现








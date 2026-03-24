"""
train_distributed.py — Ray + PyTorch DDP 分布式训练
同时支持 DeepCubeA (MLP) 和 CubeTransformer，以及任意 N×N×N 魔方

架构
────
  8 × Worker (各占 1 GPU)
      ├── 独立 ADI 数据生成（各 worker 不同随机种子，保证数据多样性）
      ├── 本地计算 Bellman target（target_model 各 rank 天然一致）
      └── loss.backward() → DDP all-reduce → 梯度同步

Checkpoint（Ray Train v2 兼容）
────────────────────────────────
  • 每 checkpoint_freq 步 / loss 创新低 / 训练结束 → 保存
  • 保留最近 num_keep_checkpoints 个（按 loss 排序）
  • 续训：扫描 storage_path 找最新 checkpoint，通过 train_loop_config 传路径，
          worker 手动 torch.load（Ray v2 已移除所有框架级续训 API）

用法
────
  # MLP，3×3，8 GPU
  python train_distributed.py --model-type mlp --cube-n 3 --num-gpus 8

  # Transformer，4×4，8 GPU
  python train_distributed.py --model-type transformer --cube-n 4 --num-gpus 8

  # 续训
  python train_distributed.py --model-type transformer --cube-n 4 --resume

  # 从 main.py 调用
  python main.py train-dist --model-type transformer --cube-n 4 --num-gpus 8
"""

from __future__ import annotations

import os
import copy
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import ray
from ray import train as ray_train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_model

import threading
from queue import Queue

class _DataPrefetcher:
    """
    在后台线程里做 CPU-only ADI 数据生成，主线程负责 GPU 推理和训练。
    CPU 数据生成与 GPU 训练完全并行，后台线程不接触任何 CUDA 对象。
    """
    def __init__(self, generate_fn, queue_size: int = 2):
        self._queue    = Queue(maxsize=queue_size)
        self._gen      = generate_fn
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                data = self._gen()
                self._queue.put(data)
            except Exception as e:
                self._queue.put(e)
                break

    def next(self):
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
# ══════════════════════════════════════════════════════════════════════
# 训练配置
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DistTrainConfig:
    # 模型类型 & 魔方阶数
    model_type: str = "mlp"          # "mlp" | "transformer"
    cube_n:     int = 3

    # MLP 结构
    hidden_dim: int = 512
    num_blocks: int = 4

    # Transformer 结构
    d_model:       int   = 256
    nhead:         int   = 8
    num_layers:    int   = 6
    use_dist_head: bool  = True
    lambda_value:  float = 1.0
    lambda_policy: float = 0.5
    lambda_dist:   float = 0.3
    # Policy 改进参数
    soft_policy:          bool  = True
    policy_temperature:   float = 1.0
    value_conf_gate:      float = 0.5
    policy_warmup_iters:  int   = 50
    bellman_chunk_size:       int   = 2048         # Bellman 分块推理（L4 24GB 可放大）
    # 显存优化
    use_bf16:                 bool  = True   # bf16 混合精度（Ampere+ GPU）
    gradient_checkpointing:   bool  = True   # 激活重算（大模型必须开启）

    # 通用训练超参
    lr:              float = 1e-3
    batch_size:      int   = 2048            # L4 24GB，从 1024 翻倍
    num_iterations:  int   = 2500
    num_sequences:   int   = 32000           # 增大数据多样性（每 GPU 4000）

    # 课程学习 & 目标网络
    max_depth:           int = 20
    target_update_freq:  int = 100
    depth_increase_freq: int = 300

    # Checkpoint
    checkpoint_freq:         int = 100
    num_keep_checkpoints:    int = 3
    export_path:             str = "deepcubea_dist.pt"

    # 日志 & Ray
    log_interval: int = 25
    storage_path: str = "./ray_results"
    run_name:     str = "deepcubea_run"
    seed:         int = 42

    # 续训（内部字段，由 launch() 填写）
    _resume_checkpoint_dir: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════
# 模型构建（在 worker 内调用）
# ══════════════════════════════════════════════════════════════════════

def _build_model(config: Dict, device: torch.device) -> nn.Module:
    from cube_env import CubeEnv
    cube_n = config["cube_n"]
    env    = CubeEnv(N=cube_n)

    if config["model_type"] == "transformer":
        from model_transformer import CubeTransformer
        return CubeTransformer(
            d_model         = config["d_model"],
            nhead           = config["nhead"],
            num_layers      = config["num_layers"],
            dim_feedforward = config["d_model"] * 4,
            dropout         = 0.1,
            num_stickers    = env.num_stickers,
            num_moves       = env.num_moves,
            use_dist_head   = config["use_dist_head"],
            cube_n          = cube_n,
            gradient_checkpointing = config.get("gradient_checkpointing", True),
        ).to(device)
    else:
        from model import DeepCubeA
        return DeepCubeA(
            input_dim=env.onehot_size,
            hidden_dim=config["hidden_dim"],
            num_blocks=config["num_blocks"],
            cube_n=cube_n,
        ).to(device)


# ══════════════════════════════════════════════════════════════════════
# 单步训练 — MLP
# ══════════════════════════════════════════════════════════════════════

def _train_step_mlp(it, model, target_model, optimizer, loss_fn,
                    config, device, seq_per_worker, env) -> float:
    from model import generate_adi_data, compute_bellman_targets
    cur_depth = min(1 + it // config["depth_increase_freq"], config["max_depth"])

    states_oh, neighbors_oh, solved_mask = generate_adi_data(
        seq_per_worker, cur_depth, device, seq_per_worker, env,
    )
    targets = compute_bellman_targets(
        neighbors_oh, solved_mask, target_model, device,
        float(cur_depth * 2 + 5),
    )
    model.train()
    total, n = 0.0, 0
    for s_b, t_b in DataLoader(TensorDataset(states_oh, targets),
                                batch_size=config["batch_size"], shuffle=True):
        optimizer.zero_grad()
        loss = loss_fn(model(s_b), t_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


# ══════════════════════════════════════════════════════════════════════
# 单步训练 — Transformer
# ══════════════════════════════════════════════════════════════════════

def _generate_adi_cpu(config, seq_per_worker, env, it):
    """
    CPU-only ADI 数据生成，供 DataPrefetcher 后台线程调用。
    只做 Python/NumPy 计算，不碰任何 CUDA 对象，返回 CPU 张量。
    主线程负责 .to(device) 和 target_model 推理。
    """
    from model_transformer import generate_adi_data_with_policy
    cur_depth = min(1 + it // config["depth_increase_freq"], config["max_depth"])
    states_int, neighbors_oh, solved_mask, dist_labels = generate_adi_data_with_policy(
        seq_per_worker, cur_depth, torch.device("cpu"), seq_per_worker, env,
    )
    return states_int, neighbors_oh, solved_mask, dist_labels, cur_depth


def _train_step_transformer(it, model, target_model, optimizer,
                            config, device, seq_per_worker, env,
                            cpu_data=None) -> Dict:
    from model_transformer import compute_targets_combined
    import torch.nn.functional as _F

    if cpu_data is not None:
        states_int, neighbors_oh, solved_mask, dist_labels, cur_depth = cpu_data
    else:
        states_int, neighbors_oh, solved_mask, dist_labels, cur_depth = \
            _generate_adi_cpu(config, seq_per_worker, env, it)

    # 主线程：CPU→GPU（安全，无 CUDA 多线程竞争）
    states_int   = states_int.to(device)
    neighbors_oh = neighbors_oh.to(device)
    solved_mask  = solved_mask.to(device)
    dist_labels  = dist_labels.to(device)

    # target_model 推理始终在主线程，合并 Bellman + Policy 为单次前向
    chunk = config.get("bellman_chunk_size", 1024)
    value_targets, policy_targets = compute_targets_combined(
        neighbors_oh, solved_mask, target_model, device,
        max_target      = float(cur_depth * 2 + 5),
        chunk_size      = chunk,
        soft            = config.get("soft_policy", True),
        temperature     = config.get("policy_temperature", 1.0),
        value_conf_gate = config.get("value_conf_gate", 0.5),
    )

    # warmup：前 N 步 λ_π=0，先让 value head 稳定
    it_global = config.get("_current_iteration", 0)
    warmup    = config.get("policy_warmup_iters", 50)
    eff_lp    = (0.0 if it_global < warmup
                 else config["lambda_policy"] * min(1.0, (it_global-warmup)/50))

    vfn      = nn.HuberLoss(delta=1.0)
    pfn_hard = nn.CrossEntropyLoss()
    pfn_soft = nn.KLDivLoss(reduction="batchmean")
    dfn      = nn.CrossEntropyLoss()

    lv_t = lp_t = ld_t = l_t = 0.0; n = 0
    use_bf16 = config.get("use_bf16", True) and device.type == "cuda"
    model.train()
    for batch in DataLoader(
        TensorDataset(states_int, value_targets, policy_targets, dist_labels),
        batch_size=config["batch_size"], shuffle=True,
    ):
        s_i, y_v, y_p, y_d = batch
        optimizer.zero_grad()

        # bf16 autocast：激活和权重计算用 bf16，optimizer 保持 fp32 master copy
        # 对 Transformer 几乎无精度损失，显存减约 40%，速度提升 10-30%
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(s_i)
            lv = vfn(out["value"].float(), y_v)
            if config.get("soft_policy", True):
                log_p = _F.log_softmax(out["policy"].float(), dim=-1)
                lp = pfn_soft(log_p, y_p.to(log_p.device))
            else:
                lp = pfn_hard(out["policy"].float(), y_p.long())
            if config["use_dist_head"] and "dist" in out:
                ld   = dfn(out["dist"].float(), y_d)
                loss = config["lambda_value"]*lv + eff_lp*lp + config["lambda_dist"]*ld
                ld_t += ld.item()
            else:
                loss = config["lambda_value"]*lv + eff_lp*lp

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        l_t += loss.item(); lv_t += lv.item(); lp_t += lp.item(); n += 1

    nb = max(n, 1)
    return {"loss": l_t/nb, "loss_v": lv_t/nb, "loss_p": lp_t/nb, "loss_d": ld_t/nb}


# ══════════════════════════════════════════════════════════════════════
# Ray Train 核心函数
# ══════════════════════════════════════════════════════════════════════

def _cleanup_old_ckpts(ckpt_base: str, num_keep: int) -> None:
    """
    保留 ckpt_base 目录下编号最大的 num_keep 个 checkpoint，删除其余。
    替代 Ray CheckpointConfig 的自动清理，避免其找不到路径时崩溃。
    """
    import shutil
    entries = sorted([
        e for e in os.listdir(ckpt_base)
        if e.startswith("ckpt_") and os.path.isdir(os.path.join(ckpt_base, e))
    ])
    # 按编号升序，删除最旧的（保留最新的 num_keep 个）
    for old in entries[:-num_keep]:
        old_path = os.path.join(ckpt_base, old)
        try:
            shutil.rmtree(old_path)
        except Exception:
            pass   # 删不掉无所谓，不阻塞训练


def _train_func(config: Dict[str, Any]) -> None:
    from cube_env import CubeEnv

    rank       = ray_train.get_context().get_local_rank()
    world_size = ray_train.get_context().get_world_size()
    is_master  = (rank == 0)
    device     = torch.device(f"cuda:{rank}")

    torch.manual_seed(config["seed"] + rank * 997)
    np.random.seed(config["seed"] + rank * 997)

    cube_n = config["cube_n"]
    env    = CubeEnv(N=cube_n) if cube_n != 3 else None

    if is_master:
        print(f"\n[Master] model={config['model_type']} cube={cube_n}x{cube_n}"
              f" world={world_size} device={device}")

    # 构建模型
    # 注意顺序：target_model 必须在 torch.compile 之前 deepcopy。
    # torch.compile 会就地修改所有子模块的 __call__（插入 fx tracer hook），
    # compile 之后 deepcopy 得到的副本带有损坏的 fx 模块引用，
    # 后台线程调用时 path_of_module 找不到子模块 → NameError。
    model = _build_model(config, device)
    model = prepare_model(model)                          # DDP 包装
    target_model = copy.deepcopy(model.module).to(device) # compile 前拷贝，得到干净副本
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)
    if torch.__version__ >= "2.0":
        model = torch.compile(model, mode="default", fullgraph=False)

    # 优化器
    if config["model_type"] == "transformer":
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"],
                                weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_iterations"], eta_min=1e-6,
    )
    mlp_loss_fn = nn.HuberLoss(delta=1.0)

    # 续训恢复
    start_it = 0; best_loss = float("inf"); history: List[Dict] = []
    ckpt_dir = config.get("_resume_checkpoint_dir")
    if ckpt_dir:
        sp = os.path.join(ckpt_dir, "state.pt")
        if os.path.exists(sp):
            saved = torch.load(sp, map_location=device, weights_only=False)
            model.module.load_state_dict(saved["model"])
            target_model.load_state_dict(saved["target_model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_it  = saved["iteration"] + 1
            best_loss = saved.get("best_loss", float("inf"))
            history   = saved.get("history", [])
            if is_master:
                print(f"[Master] ✓ 从迭代 {start_it} 恢复 (best={best_loss:.4f})")
        elif is_master:
            print(f"[Master] ⚠ checkpoint 不存在: {sp}")

    seq_per_worker = max(1, config["num_sequences"] // world_size)
    is_tf = (config["model_type"] == "transformer")

    def sync_target():
        # target_model 只在主线程使用，直接更新即可
        target_model.load_state_dict(copy.deepcopy(model.module.state_dict()))

    # 持久 checkpoint 目录（避免 tempfile 竞态问题）
    # 写到 storage_path/run_name/worker_ckpts/，由 rank-0 手动管理
    ckpt_base = os.path.join(
        config["storage_path"],   # 已在 launch() 解析为绝对路径
        config["run_name"],
        "worker_ckpts",
    )
    if is_master:
        os.makedirs(ckpt_base, exist_ok=True)

    t_start = time.time()

    # Transformer 模式：后台线程做 CPU-only ADI 生成，主线程做 GPU 推理和训练
    # 后台线程不接触任何 CUDA 对象，彻底避免 CUDA 多线程问题
    prefetcher = None
    if is_tf:
        _it_ref = [start_it]
        def _make_gen():
            return _generate_adi_cpu(config, seq_per_worker, env, _it_ref[0])
        prefetcher = _DataPrefetcher(_make_gen, queue_size=2)

    for it in range(start_it, config["num_iterations"]):
        if it % config["target_update_freq"] == 0:
            sync_target()

        config["_current_iteration"] = it
        if is_tf:
            cpu_data    = prefetcher.next()
            _it_ref[0]  = it + 1          # 通知后台预取下一迭代
            metrics  = _train_step_transformer(it, model, target_model, optimizer,
                                               config, device, seq_per_worker, env,
                                               cpu_data=cpu_data)
            avg_loss = metrics["loss"]
        else:
            avg_loss = _train_step_mlp(it, model, target_model, optimizer, mlp_loss_fn,
                                       config, device, seq_per_worker, env)
            metrics  = {"loss": avg_loss}

        scheduler.step()

        # depth 变化时重置 best_loss
        # 问题：depth=1 的 loss≈0.0005 会永远占据 best_checkpoint 槽位，
        #       导出的"最优"模型实际上只会处理 1 步状态（废模型）。
        # 修复：每次 depth 增加时重置 best_loss，让新深度重新竞争。
        cur_depth  = min(1 + it     // config["depth_increase_freq"], config["max_depth"])
        prev_depth = min(1 + (it-1) // config["depth_increase_freq"], config["max_depth"]) if it > 0 else cur_depth
        if cur_depth != prev_depth:
            if is_master:
                print(f"\n[Master] depth {prev_depth}→{cur_depth}，重置 best_loss {best_loss:.4f}→inf")
            best_loss = float("inf")

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        history.append({"it": it, "lr": optimizer.param_groups[0]["lr"],
                        "depth": cur_depth, **metrics})

        # Checkpoint：只写到我们自己的 worker_ckpts 目录，不经过 Ray checkpoint 管理。
        # Ray 的 CheckpointConfig 会跨 run 追踪旧 checkpoint 路径并尝试删除，
        # 一旦那些路径已不存在就抛 FileNotFoundError → ControllerError 崩溃。
        # 解决方案：report() 永远传 checkpoint=None，由我们自己的 _cleanup_old_ckpts 管理。
        need_save = (is_best or (it+1) % config["checkpoint_freq"] == 0
                     or it == config["num_iterations"] - 1)

        if need_save and is_master:
            ckpt_dir_this = os.path.join(ckpt_base, f"ckpt_{it:06d}")
            os.makedirs(ckpt_dir_this, exist_ok=True)
            torch.save({
                "iteration":    it,
                "model":        model.module.state_dict(),
                "target_model": target_model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "best_loss":    best_loss,
                "history":      history,
                "config":       config,
            }, os.path.join(ckpt_dir_this, "state.pt"))
            _cleanup_old_ckpts(ckpt_base, config.get("num_keep_checkpoints", 3))

        ray_train.report(
            metrics={"loss": avg_loss, "best_loss": best_loss,
                     "depth": cur_depth, "lr": optimizer.param_groups[0]["lr"],
                     "iteration": it,
                     **{k: v for k, v in metrics.items() if k != "loss"}},
            checkpoint=None,   # 不让 Ray 管理 checkpoint，避免跨 run 路径失效
        )

        if is_master and (it + 1) % config["log_interval"] == 0:
            elapsed = time.time() - t_start
            eta     = elapsed / (it - start_it + 1) * (config["num_iterations"] - it - 1)
            extra   = (f"  v={metrics.get('loss_v',0):.3f}"
                       f"  π={metrics.get('loss_p',0):.3f}"
                       f"  d={metrics.get('loss_d',0):.3f}") if is_tf else ""
            print(f"  [{it+1:4d}/{config['num_iterations']}]"
                  f"  loss={avg_loss:.4f}  best={best_loss:.4f}{extra}"
                  f"  depth={cur_depth}  lr={optimizer.param_groups[0]['lr']:.2e}"
                  f"  eta={eta/60:.1f}min" + (" ★" if is_best else ""))

    if prefetcher is not None:
        prefetcher.stop()


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def _worker_ckpts_dir(storage_path: str, run_name: str) -> str:
    return os.path.join(os.path.abspath(storage_path), run_name, "worker_ckpts")


def _find_latest_checkpoint_dir(storage_path: str, run_name: str) -> Optional[str]:
    """扫描 worker_ckpts 目录，返回迭代编号最大的 checkpoint 路径。"""
    ckpt_base = _worker_ckpts_dir(storage_path, run_name)
    if not os.path.isdir(ckpt_base):
        return None
    entries = sorted([
        e for e in os.listdir(ckpt_base)
        if e.startswith("ckpt_") and os.path.exists(os.path.join(ckpt_base, e, "state.pt"))
    ])
    if not entries:
        return None
    latest = os.path.join(ckpt_base, entries[-1])
    print(f"  发现 checkpoint: {latest}")
    return latest


def export_best_model(storage_path: str, run_name: str, export_path: str) -> None:
    """扫描 worker_ckpts，找 best_loss 最小的 checkpoint 导出，不依赖 Ray 的 checkpoint 管理。"""
    ckpt_base = _worker_ckpts_dir(storage_path, run_name)
    if not os.path.isdir(ckpt_base):
        print("⚠ worker_ckpts 目录不存在，跳过导出"); return

    best_loss = float("inf")
    best_path = None
    for d in sorted(os.listdir(ckpt_base)):
        sp = os.path.join(ckpt_base, d, "state.pt")
        if not os.path.exists(sp):
            continue
        try:
            saved = torch.load(sp, map_location="cpu", weights_only=False)
            if saved.get("best_loss", float("inf")) < best_loss:
                best_loss = saved["best_loss"]
                best_path = sp
        except Exception:
            pass

    if best_path is None:
        print("⚠ 未找到有效 checkpoint，跳过导出"); return

    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    torch.save({"model_state_dict": saved["model"], "iteration": saved["iteration"],
                "loss": saved["best_loss"], "config": saved["config"],
                "history": saved.get("history", [])}, export_path)
    cfg = saved["config"]
    mt  = cfg.get("model_type", "mlp")
    print(f"\n✓ {mt.upper()} 模型已导出: {export_path}")
    print(f"  迭代={saved['iteration']} | loss={saved['best_loss']:.4f} | cube={cfg['cube_n']}x{cfg['cube_n']}")
    if mt == "transformer":
        print(f"  d_model={cfg['d_model']} | layers={cfg['num_layers']}")
    else:
        print(f"  hidden={cfg['hidden_dim']} | blocks={cfg['num_blocks']}")


# ══════════════════════════════════════════════════════════════════════
# 启动器
# ══════════════════════════════════════════════════════════════════════

def launch(cfg: DistTrainConfig, num_gpus: int = 8, resume: bool = False):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    scaling_config = ScalingConfig(num_workers=num_gpus, use_gpu=True,
                                   resources_per_worker={"GPU": 1})
    # checkpoint_config 不传：我们自己管理 worker_ckpts，不让 Ray 追踪 checkpoint 路径
    run_config = RunConfig(
        name=cfg.run_name, storage_path=os.path.abspath(cfg.storage_path),
    )
    cfg_dict = asdict(cfg)
    # 在主进程把相对路径解析成绝对路径，防止 Ray worker 的 cwd 不同导致路径错位
    cfg_dict["storage_path"] = os.path.abspath(cfg.storage_path)
    if resume:
        ckpt_dir = _find_latest_checkpoint_dir(
            os.path.abspath(cfg.storage_path), cfg.run_name)
        cfg_dict["_resume_checkpoint_dir"] = ckpt_dir
        print(f"{'▶ 续训' if ckpt_dir else '⚠ 未找到 checkpoint，从头开始'}")
    else:
        cfg_dict["_resume_checkpoint_dir"] = None

    return TorchTrainer(train_loop_per_worker=_train_func,
                        train_loop_config=cfg_dict,
                        scaling_config=scaling_config,
                        run_config=run_config).fit()


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def _print_banner(cfg: DistTrainConfig, num_gpus: int) -> None:
    seq_per = cfg.num_sequences // num_gpus
    print(f"\n{'═'*62}")
    print(f"  DeepCubeA 分布式训练  (Ray + PyTorch DDP)")
    print(f"{'═'*62}")
    print(f"  模型类型   : {cfg.model_type.upper()}  |  魔方: {cfg.cube_n}×{cfg.cube_n}")
    print(f"  GPU 数量   : {num_gpus}  |  总迭代: {cfg.num_iterations}")
    print(f"  样本数/迭代: {cfg.num_sequences}  (每 GPU {seq_per})")
    if cfg.model_type == "transformer":
        print(f"  d_model={cfg.d_model}  nhead={cfg.nhead}  layers={cfg.num_layers}")
        print(f"  λ_v={cfg.lambda_value}  λ_π={cfg.lambda_policy}  λ_d={cfg.lambda_dist}")
    else:
        print(f"  hidden={cfg.hidden_dim}  blocks={cfg.num_blocks}")
    print(f"  lr={cfg.lr}  batch(per-GPU)={cfg.batch_size}")
    print(f"  target_update={cfg.target_update_freq}  depth_increase={cfg.depth_increase_freq}  max_depth={cfg.max_depth}")
    print(f"  ckpt 每 {cfg.checkpoint_freq} 步，保留 {cfg.num_keep_checkpoints} 个  →  {cfg.export_path}")
    print(f"{'═'*62}\n")


def cmd_train_dist(args) -> None:
    mt     = getattr(args, "model_type", "mlp")
    cube_n = getattr(args, "cube_n", 3)
    cfg = DistTrainConfig(
        model_type=mt, cube_n=cube_n,
        hidden_dim=getattr(args, "hidden_dim", 512),
        num_blocks=getattr(args, "num_blocks", 4),
        d_model=getattr(args, "d_model", 256),
        nhead=getattr(args, "nhead", 8),
        num_layers=getattr(args, "num_layers", 6),
        use_dist_head=True, lambda_value=1.0, lambda_policy=0.5, lambda_dist=0.3,
        soft_policy=True, policy_temperature=1.0, value_conf_gate=0.5,
        policy_warmup_iters=50,
        use_bf16=True, gradient_checkpointing=True,
        lr=args.lr, batch_size=args.batch_size,
        num_iterations=args.iterations, num_sequences=args.sequences,
        max_depth=getattr(args, "max_depth_train", getattr(args, "max_depth", 20)),
        target_update_freq=args.target_update_freq,
        depth_increase_freq=args.depth_increase_freq,
        checkpoint_freq=args.checkpoint_freq,
        num_keep_checkpoints=args.num_keep_checkpoints,
        export_path=args.model_path,
        log_interval=max(1, args.iterations // 100),
        storage_path=args.storage_path, run_name=args.run_name, seed=args.seed,
    )
    _print_banner(cfg, args.num_gpus)
    result = launch(cfg, num_gpus=args.num_gpus, resume=args.resume)
    print(f"\n训练完成！最优 loss: {result.metrics.get('best_loss', float('nan')):.4f}")
    export_best_model(cfg.storage_path, cfg.run_name, cfg.export_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-type", default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--cube-n",     type=int, default=3)
    p.add_argument("--num-gpus",   type=int, default=8)
    p.add_argument("--iterations", type=int, default=2500)
    p.add_argument("--sequences",  type=int, default=8000)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-depth",  type=int, default=20)
    p.add_argument("--target-update-freq",  type=int, default=100)
    p.add_argument("--depth-increase-freq", type=int, default=300)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-blocks", type=int, default=4)
    p.add_argument("--d-model",    type=int, default=256)
    p.add_argument("--nhead",      type=int, default=8)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--checkpoint-freq",      type=int, default=100)
    p.add_argument("--num-keep-checkpoints", type=int, default=3)
    p.add_argument("--model-path",   default="deepcubea_dist.pt")
    p.add_argument("--storage-path", default="./ray_results")
    p.add_argument("--run-name",     default="deepcubea_run")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--resume",       action="store_true")
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    cmd_train_dist(_args)
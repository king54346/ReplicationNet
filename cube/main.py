"""
main.py — DeepCubeA & CubeTransformer 统一入口
支持任意 N×N×N 魔方

命令速查
────────
  python main.py train --cube-n 3               # 3×3 MLP 训练
  python main.py train --cube-n 4               # 4×4 MLP 训练
  python main.py train-transformer --cube-n 4   # 4×4 Transformer 训练
  python main.py train-dist --num-gpus 8        # Ray 分布式
  python main.py solve --cube-n 4 --depth 10   # 求解
  python main.py solve-transformer --cube-n 4   # Transformer 求解
  python main.py explain --cube-n 3             # A* 展示
  python main.py benchmark --cube-n 4           # 基准测试
  python main.py train-dist --model-type transformer --cube-n 3 \
    --num-gpus 8 --d-model 256 --num-layers 6 --iterations 2500
"""
import sys, os, argparse, numpy as np, torch, time
sys.path.insert(0, os.path.dirname(__file__))
from cube_env import CubeEnv, render_state


def get_device():
    if torch.cuda.is_available():   return torch.device('cuda')
    if hasattr(torch.backends,'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def get_env(args):
    n = getattr(args, 'cube_n', 3)
    return CubeEnv(N=n) if n != 3 else None

def _desc(env): return f"{env.N}×{env.N}" if env else "3×3"

def _env_fns(env):
    """返回 (scramble, render, apply_move, is_solved) 四元组"""
    if env:
        return env.scramble, env.render, env.apply_move, env.is_solved
    from cube_env import scramble, is_solved, apply_move
    return scramble, render_state, apply_move, is_solved


# ─── train ────────────────────────────────────────────────────────────
def cmd_train(args):
    from model import DeepCubeA, Trainer, evaluate_heuristic
    device = get_device(); env = get_env(args)
    print(f"设备: {device}  魔方: {_desc(env)}")

    if env:
        model = DeepCubeA.for_env(env, hidden_dim=args.hidden_dim,
                                   num_blocks=args.num_blocks).to(device)
    else:
        model = DeepCubeA(324, args.hidden_dim, args.num_blocks).to(device)

    Trainer(model, device, env=env, lr=args.lr,
            batch_size=args.batch_size,
            target_update_freq=args.target_update_freq,
            depth_increase_freq=args.depth_increase_freq,
            max_depth=args.max_depth_train
    ).train(args.iterations, args.sequences,
            max(1, args.iterations//20), args.model_path)
    evaluate_heuristic(model, device, env=env)


# ─── train-dist ───────────────────────────────────────────────────────
def cmd_train_dist(args):
    try: from train_distributed import cmd_train_dist as _d
    except ImportError as e: print(f"✗ {e}\n  pip install ray[train]"); sys.exit(1)
    _d(args)


# ─── train-transformer ───────────────────────────────────────────────
def cmd_train_transformer(args):
    from model_transformer import CubeTransformer, TransformerTrainer, evaluate_transformer
    device = get_device()
    # cubie 表示仅支持 3×3
    if getattr(args, 'cube_n', 3) != 3:
        print("警告: cubie Transformer 仅支持 3×3 魔方，忽略 --cube-n 参数")
    print(f"设备: {device}  魔方: 3×3 (cubie 表示)")

    model = CubeTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.d_model*4, dropout=0.1,
        use_dist_head=True,
    ).to(device)
    print(f"参数量: {model.num_parameters:,}")

    TransformerTrainer(
        model, device, lr=args.lr, batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        depth_increase_freq=args.depth_increase_freq,
        lambda_value=1.0, lambda_policy=0.5, lambda_dist=0.3,
    ).train(args.iterations, args.sequences,
            max(1, args.iterations//25), args.transformer_path)
    evaluate_transformer(model, device)


# ─── solve ────────────────────────────────────────────────────────────
def cmd_solve(args):
    from model import load_deepcubea
    from solver import WeightedAStarSolver
    device = get_device(); env = get_env(args)
    model  = load_deepcubea(args.model_path, device).to(device)
    sc, rn, ap, is_ = _env_fns(env)

    state, moves = sc(args.depth, seed=args.seed)
    print(f"\n[{_desc(env)}]  打乱 {len(moves)} 步: {' '.join(moves)}")
    print(rn(state))

    sol, stats = WeightedAStarSolver(
        model, device, env=env, weight=args.weight,
        max_nodes=args.max_nodes, cache_size=args.max_nodes*2,
    ).solve(state, verbose=True)

    if sol:
        v = state.copy()
        for m in sol: v = ap(v, m)
        assert is_(v), "解验证失败！"
        print(f"\n解: {' '.join(sol)}\n✓ 已验证\n\n复原:\n{rn(v)}")


# ─── solve-transformer ───────────────────────────────────────────────
def cmd_solve_transformer(args):
    from model_transformer import load_transformer
    from solver_policy import PolicyPrunedAStarSolver, benchmark_solvers
    device = get_device(); env = get_env(args)
    model  = load_transformer(args.transformer_path, device).to(device)
    sc, rn, ap, is_ = _env_fns(env)

    state, moves = sc(args.depth, seed=args.seed)
    print(f"\n[{_desc(env)}]  打乱 {len(moves)} 步: {' '.join(moves)}")
    print(rn(state))

    sol, stats = PolicyPrunedAStarSolver(
        model, device, env=env, weight=args.weight,
        top_k=args.top_k, max_nodes=args.max_nodes,
    ).solve(state, verbose=True)

    if sol:
        v = state.copy()
        for m in sol: v = ap(v, m)
        assert is_(v), "解验证失败！"
        print(f"\n解: {' '.join(sol)}\n✓ 已验证\n\n复原:\n{rn(v)}")

    if args.benchmark_transformer:
        benchmark_solvers(model, device, env=env,
                          num_tests=args.num_tests, max_depth=args.max_depth)


# ─── explain ─────────────────────────────────────────────────────────
def cmd_explain(args):
    from model import load_deepcubea
    from solver import explain_astar
    device = get_device(); env = get_env(args)
    model  = load_deepcubea(args.model_path, device).to(device)
    sc, rn, _, _ = _env_fns(env)

    depth = min(args.depth, 5)
    state, moves = sc(depth, seed=args.seed)
    print(f"\n[{_desc(env)}]  打乱 {depth} 步: {' '.join(moves)}")
    explain_astar(state, model, device, env=env, weight=args.weight)


# ─── benchmark ───────────────────────────────────────────────────────
def cmd_benchmark(args):
    from model import load_deepcubea
    from solver import WeightedAStarSolver, BeamSearchSolver, run_benchmark
    device = get_device(); env = get_env(args)
    model  = load_deepcubea(args.model_path, device).to(device)

    print(f"\n{'='*65}")
    print(f"基准测试: {_desc(env)} | {args.num_tests} tests/depth | 1~{args.max_depth}")
    print(f"{'='*65}")

    for w in [1.0, 1.5, 2.0]:
        print(f"\n[WA* w={w}]")
        run_benchmark(WeightedAStarSolver(model, device, env=env,
                                          weight=w, max_nodes=args.max_nodes),
                      env, args.num_tests, (1, args.max_depth))

    print(f"\n[BeamSearch bw=256]")
    run_benchmark(BeamSearchSolver(model, device, env=env, beam_width=256),
                  env, args.num_tests, (1, args.max_depth))


# ─── argparse ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='DeepCubeA + Transformer 魔方求解（支持任意 NxN）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('command', choices=[
        'train','train-dist','train-transformer',
        'solve','solve-transformer','explain','benchmark',
    ])

    # 通用
    p.add_argument('--cube-n',    type=int,   default=3,       help='魔方阶数')
    p.add_argument('--model-path',            default='deepcubea.pt')
    p.add_argument('--depth',     type=int,   default=4,      help='求解/展示/基准测试打乱深度')
    p.add_argument('--weight',    type=float, default=1.5,     help='A* 权重')
    p.add_argument('--max-nodes', type=int,   default=100_000)
    p.add_argument('--seed',      type=int,   default=43)
    p.add_argument('--num-tests', type=int,   default=10)
    p.add_argument('--max-depth', type=int,   default=12,      help='基准测试最大深度')

    # MLP 训练
    g = p.add_argument_group('MLP 训练')
    g.add_argument('--iterations',          type=int,   default=10000)
    g.add_argument('--sequences',           type=int,   default=10000)
    g.add_argument('--hidden-dim',          type=int,   default=512)
    g.add_argument('--num-blocks',          type=int,   default=4)
    g.add_argument('--lr',                  type=float, default=1e-3)
    g.add_argument('--batch-size',          type=int,   default=6000)
    g.add_argument('--target-update-freq',  type=int,   default=50)
    g.add_argument('--depth-increase-freq', type=int,   default=400)
    g.add_argument('--max-depth-train',     type=int,   default=21,
                   help='训练课程学习最大深度')

    # 分布式
    g2 = p.add_argument_group('分布式训练')
    g2.add_argument('--model-type', default='mlp', choices=['mlp','transformer'],
                    help='train-dist 模型类型（mlp 或 transformer）')
    g2.add_argument('--num-gpus',             type=int, default=8)
    g2.add_argument('--checkpoint-freq',      type=int, default=50)
    g2.add_argument('--num-keep-checkpoints', type=int, default=3)
    g2.add_argument('--storage-path',         type=str, default='./ray_results')
    g2.add_argument('--run-name',             type=str, default='deepcubea_run')
    g2.add_argument('--resume',               action='store_true')

    # Transformer
    g3 = p.add_argument_group('Transformer')
    g3.add_argument('--transformer-path',      default='cube_transformer.pt')
    g3.add_argument('--d-model',     type=int, default=1024)
    g3.add_argument('--nhead',       type=int, default=8)
    g3.add_argument('--num-layers',  type=int, default=12)
    g3.add_argument('--top-k',       type=int, default=9,   help='Policy A* 展开动作数')
    g3.add_argument('--benchmark-transformer', action='store_true')

    args = parser.parse_args() if False else p.parse_args()

    {
        'train':             cmd_train,
        'train-dist':        cmd_train_dist,
        'train-transformer': cmd_train_transformer,
        'solve':             cmd_solve,
        'solve-transformer': cmd_solve_transformer,
        'explain':           cmd_explain,
        'benchmark':         cmd_benchmark,
    }[args.command](args)


if __name__ == '__main__':
    main()
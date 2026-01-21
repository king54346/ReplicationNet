import torch
T = 1000

# 生成beta调度
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """线性beta调度"""
    return torch.linspace(beta_start, beta_end, timesteps)

# 计算alpha相关参数
betas = linear_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

# 计算方差
variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
variance = torch.clamp(variance, min=1e-20)  # 避免数值不稳定


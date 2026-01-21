# diffusion.py
import torch
from config import T

class DiffusionSchedule:
    """扩散调度器 (设备安全)"""
    
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # 在CPU上初始化
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
        
        # 计算方差
        variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        variance = torch.clamp(variance, min=1e-20)
        
        # 保存为buffer (会自动跟随模型设备)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('variance', variance)
        
        self._device = 'cpu'
    
    def register_buffer(self, name, tensor):
        """注册buffer"""
        setattr(self, name, tensor)
    
    def to(self, device):
        """移动到指定设备"""
        if self._device != device:
            for name in ['betas', 'alphas', 'alphas_cumprod', 
                        'alphas_cumprod_prev', 'variance']:
                if hasattr(self, name):
                    setattr(self, name, getattr(self, name).to(device))
            self._device = device
        return self
    
    def forward_add_noise(self, x, t):
        """
        前向加噪
        
        Args:
            x: [B, C, H, W] 原始图像
            t: [B] 时间步
        """
        # 自动使用x的设备
        device = x.device
        self.to(device)
        
        # 生成噪声
        noise = torch.randn_like(x)
        
        # 获取alpha_cumprod
        alpha_cumprod_t = self.alphas_cumprod[t].view(x.size(0), 1, 1, 1)
        
        # 加噪
        x_noisy = torch.sqrt(alpha_cumprod_t) * x + \
                  torch.sqrt(1.0 - alpha_cumprod_t) * noise
        
        return x_noisy, noise
    
    def backward_denoise_step(self, model, x, t, y=None):
        """
        单步去噪
        
        Args:
            model: DiT模型
            x: [B, C, H, W] 当前噪声图像
            t: [B] 当前时间步
            y: [B] 类别标签
        """
        device = x.device
        self.to(device)
        
        # 预测噪声
        with torch.no_grad():
            noise_pred = model(x, t, y)
        
        # 获取参数
        alpha_t = self.alphas[t].view(x.size(0), 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(x.size(0), 1, 1, 1)
        
        # 计算均值
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod_t)) * noise_pred
        )
        
        # 添加噪声 (除了t=0)
        if t[0] > 0:
            sigma_t = torch.sqrt(self.variance[t].view(x.size(0), 1, 1, 1))
            z = torch.randn_like(x)
            x_prev = mean + sigma_t * z
        else:
            x_prev = mean
        
        return torch.clamp(x_prev, -1.0, 1.0)


# 创建全局实例
diffusion_schedule = DiffusionSchedule(timesteps=T)

# 向后兼容的包装函数
def forward_add_noise(x, t, device=None):
    """向后兼容的接口"""
    if device is not None:
        diffusion_schedule.to(device)
    return diffusion_schedule.forward_add_noise(x, t)
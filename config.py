"""
Configuration file for Transformer ASR
"""

# 数据配置
DATA_CONFIG = {
    'data_dir': '/path/to/LRS2',  # LRS2数据集路径
    'sample_rate': 16000,         # 采样率
    'n_mels': 80,                 # Mel滤波器数量
    'win_length': 400,            # 窗口长度 (25ms at 16kHz)
    'hop_length': 160,            # 帧移 (10ms at 16kHz)
    'max_audio_len': 10.0,        # 最大音频长度(秒)
    'max_text_len': 200,          # 最大文本长度
}

# 模型配置
MODEL_CONFIG = {
    'd_model': 512,               # 模型维度
    'nhead': 8,                   # 注意力头数
    'num_encoder_layers': 12,     # Encoder层数
    'num_decoder_layers': 6,      # Decoder层数
    'dim_feedforward': 2048,      # FFN维度
    'dropout': 0.1,               # Dropout率
    'vocab_size': 30,             # 词表大小（字符级）
    'use_cnn_subsampling': True,  # 是否使用CNN下采样
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 8,              # 批大小
    'num_workers': 4,             # 数据加载线程数
    'num_epochs': 100,            # 训练轮数
    'learning_rate': 1e-4,        # 学习率
    'warmup_steps': 4000,         # Warmup步数
    'weight_decay': 0.01,         # 权重衰减
    'grad_clip': 1.0,             # 梯度裁剪
    'label_smoothing': 0.1,       # 标签平滑
    'checkpoint_dir': 'checkpoints',  # 检查点目录
    'log_interval': 100,          # 日志间隔
    'eval_interval': 1000,        # 验证间隔
    'save_interval': 5,           # 保存间隔(epochs)
}

# 推理配置
INFERENCE_CONFIG = {
    'beam_width': 5,              # Beam Search宽度
    'max_decode_len': 200,        # 最大解码长度
    'device': 'cuda',             # 设备
}

# 优化器配置
OPTIMIZER_CONFIG = {
    'type': 'adamw',              # 优化器类型
    'betas': (0.9, 0.98),         # Adam的beta参数
    'eps': 1e-9,                  # epsilon
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'spec_augment': True,         # 是否使用SpecAugment
    'time_mask_width': 50,        # 时间mask宽度
    'freq_mask_width': 10,        # 频率mask宽度
    'time_mask_num': 2,           # 时间mask数量
    'freq_mask_num': 2,           # 频率mask数量
    'speed_perturb': False,       # 是否使用速度扰动
    'speed_range': (0.9, 1.1),    # 速度扰动范围
}


def get_config():
    """获取完整配置"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'inference': INFERENCE_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
    }


if __name__ == '__main__':
    import json
    config = get_config()
    print(json.dumps(config, indent=2))
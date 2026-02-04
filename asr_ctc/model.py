"""
Pure CTC Model - FIXED
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpecAugment(nn.Module):
    def __init__(self, freq_mask=27, time_mask=100):
        super().__init__()
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def forward(self, x):
        if not self.training:
            return x

        batch, time, freq = x.shape
        x = x.clone()

        # Frequency masking
        if torch.rand(1).item() < 0.15:
            for _ in range(2):
                f = torch.randint(0, self.freq_mask + 1, (1,)).item()
                f0 = torch.randint(0, freq - f + 1, (1,)).item()
                x[:, :, f0:f0+f] = 0

        # Time masking
        if torch.rand(1).item() < 0.15:
            for _ in range(2):
                t = torch.randint(0, min(self.time_mask + 1, time // 5), (1,)).item()
                t0 = torch.randint(0, time - t + 1, (1,)).item()
                x[:, t0:t0+t, :] = 0

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNSubsampling(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(64 * (in_channels // 4), out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)

        # Reshape
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)

        x = self.linear(x)
        x = self.layer_norm(x)
        return x


class CTCASR(nn.Module):
    """纯CTC语音识别模型"""

    def __init__(
        self,
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        dim_feedforward=2048,
        dropout=0.3,
        vocab_size=30,
        blank_id=0,
        use_cnn_subsampling=True,
        use_spec_augment=True
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        # SpecAugment
        self.spec_augment = SpecAugment() if use_spec_augment else None

        # Feature extraction
        if use_cnn_subsampling:
            self.feature_extractor = CNNSubsampling(n_mels, d_model, dropout)
            self.subsampling_factor = 4
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_mels, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            self.subsampling_factor = 1

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, nn.LayerNorm(d_model)
        )

        # CTC output layer
        self.ctc_output = nn.Linear(d_model, vocab_size)

        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='mean', zero_infinity=True)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'ctc_output' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                else:
                    nn.init.xavier_uniform_(p)

    def _create_audio_mask(self, src):
        """检测全为0的时间步 -> True=padding"""
        return (src.abs().sum(dim=-1) == 0)

    def _adjust_mask_for_subsampling(self, mask, new_length):
        """调整mask以匹配subsampling后的长度"""
        if mask is None:
            return None

        b, old_len = mask.size()
        mask_4d = mask.view(b, 1, 1, old_len).float()
        pooled = F.max_pool2d(mask_4d, kernel_size=(1, self.subsampling_factor),
                             stride=(1, self.subsampling_factor))
        adjusted = pooled.view(b, -1).bool()

        # 调整到目标长度
        curr_len = adjusted.size(1)
        if curr_len < new_length:
            adjusted = torch.cat([
                adjusted,
                torch.zeros(b, new_length - curr_len, dtype=torch.bool, device=adjusted.device)
            ], dim=1)
        elif curr_len > new_length:
            adjusted = adjusted[:, :new_length]

        return adjusted

    def _compute_output_length(self, input_length):
        """计算CNN subsampling后的输出长度"""
        length = input_length
        # Conv1: kernel=3, stride=2, padding=1
        length = (length + 2 * 1 - 3) // 2 + 1
        # Conv2: kernel=3, stride=2, padding=1
        length = (length + 2 * 1 - 3) // 2 + 1
        return length

    def forward(self, src, src_lengths=None):
        """
        前向传播
        Args:
            src: (B, T, n_mels) - 音频特征，padding=0
            src_lengths: (B,) - 实际长度（可选）
        Returns:
            log_probs: (B, T', vocab_size) - log概率
            output_lengths: (B,) - 编码后的实际长度
        """
        batch_size = src.size(0)
        device = src.device
        
        # 创建mask（如果没提供src_lengths）
        if src_lengths is None:
            src_mask = self._create_audio_mask(src)
        else:
            # 从src_lengths创建mask
            src_mask = torch.arange(src.size(1), device=device)[None, :] >= src_lengths[:, None]

        # SpecAugment
        if self.training and self.spec_augment is not None:
            src = self.spec_augment(src)

        # Feature extraction
        src = self.feature_extractor(src)
        src = self.pos_encoder(src)

        # 获取CNN后的实际序列长度
        actual_seq_length = src.size(1)

        # ✅ 计算output_lengths（使用精确的公式）
        if src_lengths is not None:
            # 使用提供的长度计算（最准确）
            output_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                output_lengths[i] = self._compute_output_length(src_lengths[i].item())
        else:
            # 从mask推断
            valid_lengths = (~src_mask).sum(dim=1)
            output_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(batch_size):
                output_lengths[i] = self._compute_output_length(valid_lengths[i].item())

        # 调整mask到实际长度
        if self.subsampling_factor > 1:
            src_mask = self._adjust_mask_for_subsampling(src_mask, actual_seq_length)

        # Encode
        memory = self.encoder(src, src_key_padding_mask=src_mask)

        # CTC output
        logits = self.ctc_output(memory)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs, output_lengths

    def decode_greedy(self, src, src_lengths=None):
        """
        贪婪解码
        Args:
            src: (B, T, n_mels)
            src_lengths: (B,) - 实际长度
        Returns:
            decoded: List[List[int]] - 解码结果
        """
        log_probs, output_lengths = self.forward(src, src_lengths)

        # 取每帧最大概率的token
        best_paths = torch.argmax(log_probs, dim=-1)  # (B, T')

        decoded = []
        for i in range(best_paths.size(0)):
            length = output_lengths[i].item()
            path = best_paths[i, :length].tolist()

            # 去除blank和重复
            decoded_seq = []
            prev = None
            for token in path:
                if token != self.blank_id and token != prev:
                    decoded_seq.append(token)
                prev = token

            decoded.append(decoded_seq)

        return decoded

    def decode_beam_search(self, src, src_lengths=None, beam_size=10):
        """
        Beam Search解码
        Args:
            src: (B, T, n_mels)
            src_lengths: (B,) - 实际长度
            beam_size: beam宽度
        Returns:
            decoded: List[List[int]] - 解码结果
        """
        log_probs, output_lengths = self.forward(src, src_lengths)

        decoded = []
        for i in range(log_probs.size(0)):
            length = output_lengths[i].item()
            log_prob_seq = log_probs[i, :length, :]  # (T', vocab_size)
            result = self._beam_search_single(log_prob_seq, beam_size)
            decoded.append(result)

        return decoded

    def _beam_search_single(self, log_probs, beam_size):
        """单个样本的beam search"""
        T, vocab_size = log_probs.shape

        # 初始化beam: [(prefix, log_prob_blank, log_prob_non_blank)]
        beam = [([], 0.0, float('-inf'))]

        for t in range(T):
            new_beam = {}

            for prefix, log_p_b, log_p_nb in beam:
                for c in range(vocab_size):
                    log_p_c = log_probs[t, c].item()

                    if c == self.blank_id:
                        # Blank: 不扩展序列
                        new_prefix = tuple(prefix)
                        new_log_p_b = self._log_add(log_p_b + log_p_c, log_p_nb + log_p_c)

                        if new_prefix in new_beam:
                            old_log_p_b, old_log_p_nb = new_beam[new_prefix]
                            new_beam[new_prefix] = (
                                self._log_add(old_log_p_b, new_log_p_b),
                                old_log_p_nb
                            )
                        else:
                            new_beam[new_prefix] = (new_log_p_b, float('-inf'))

                    else:
                        # 非blank token
                        new_prefix = tuple(prefix + [c])

                        if len(prefix) > 0 and prefix[-1] == c:
                            # 重复字符: 只能从blank转移
                            new_log_p_nb = log_p_b + log_p_c
                        else:
                            # 新字符
                            new_log_p_nb = self._log_add(log_p_b + log_p_c, log_p_nb + log_p_c)

                        if new_prefix in new_beam:
                            old_log_p_b, old_log_p_nb = new_beam[new_prefix]
                            new_beam[new_prefix] = (
                                old_log_p_b,
                                self._log_add(old_log_p_nb, new_log_p_nb)
                            )
                        else:
                            new_beam[new_prefix] = (float('-inf'), new_log_p_nb)

            # Pruning: 保留top beam_size
            beam = []
            for prefix, (log_p_b, log_p_nb) in new_beam.items():
                total_log_p = self._log_add(log_p_b, log_p_nb)
                beam.append((list(prefix), log_p_b, log_p_nb, total_log_p))

            beam.sort(key=lambda x: x[3], reverse=True)
            beam = [(prefix, log_p_b, log_p_nb) for prefix, log_p_b, log_p_nb, _ in beam[:beam_size]]

        # 返回最佳序列
        if beam:
            return beam[0][0]
        return []

    def _log_add(self, log_a, log_b):
        """log(exp(a) + exp(b))"""
        if log_a == float('-inf'):
            return log_b
        if log_b == float('-inf'):
            return log_a
        if log_a > log_b:
            return log_a + math.log1p(math.exp(log_b - log_a))
        else:
            return log_b + math.log1p(math.exp(log_a - log_b))


def compute_cnn_output_length(input_length, kernel_size=3, stride=2, padding=1, num_layers=2):
    """计算CNN subsampling后的实际输出长度"""
    length = input_length
    for _ in range(num_layers):
        length = (length + 2 * padding - kernel_size) // stride + 1
    return length


if __name__ == '__main__':
    # 测试
    model = CTCASR(
        n_mels=80,
        d_model=512,
        vocab_size=30,
        blank_id=0,
        use_spec_augment=False  # ✅ 测试时关闭
    )
    
    print("=" * 70)
    print("验证 CNN Output Length")
    print("=" * 70)
    
    # ✅ 使用randn而不是zeros测试
    test_lengths = [299, 300, 1000, 1001, 1500]
    for length in test_lengths:
        audio = torch.randn(1, length, 80)  # ✅ 修改：使用randn
        src_lengths = torch.tensor([length])  # ✅ 提供实际长度
        
        with torch.no_grad():
            log_probs, output_lengths = model(audio, src_lengths)
        
        actual = log_probs.size(1)
        computed = compute_cnn_output_length(length)
        expected = output_lengths[0].item()
        
        match = "✓" if (actual == computed == expected) else "✗"
        print(f"{match} Input: {length:4d} -> Actual: {actual:3d}, Computed: {computed:3d}, Model: {expected:3d}")
    
    print("\n" + "=" * 70)
    print("测试批量数据")
    print("=" * 70)
    
    # 示例数据
    audio = torch.randn(4, 1000, 80)
    audio_lengths = torch.tensor([1000, 950, 900, 850])

    # 前向
    log_probs, output_lengths = model(audio, audio_lengths)
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Output lengths: {output_lengths}")
    
    # 验证每个长度
    for i, length in enumerate(audio_lengths):
        expected = compute_cnn_output_length(length.item())
        actual = output_lengths[i].item()
        match = "✓" if expected == actual else "✗"
        print(f"{match} Sample {i}: input={length.item()}, output={actual}, expected={expected}")

    # 解码
    model.eval()
    decoded = model.decode_greedy(audio, audio_lengths)
    print(f"\nDecoded (greedy) - lengths: {[len(d) for d in decoded]}")

    decoded_beam = model.decode_beam_search(audio, audio_lengths, beam_size=10)
    print(f"Decoded (beam) - lengths: {[len(d) for d in decoded_beam]}")
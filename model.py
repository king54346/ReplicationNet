"""
Transformer ASR Model - 自动处理 padding
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


class TransformerASR(nn.Module):
    """自动处理 padding: audio=0, token=pad_token_id"""
    
    def __init__(
        self,
        n_mels=80,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.3,
        vocab_size=30,
        pad_token_id=1,
        use_cnn_subsampling=True,
        use_spec_augment=True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # SpecAugment
        self.spec_augment = SpecAugment() if use_spec_augment else None
        
        # Feature extraction
        if use_cnn_subsampling:
            self.feature_extractor = CNNSubsampling(n_mels, d_model, dropout)
        else:
            self.feature_extractor = nn.Sequential(
                nn.Linear(n_mels, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, nn.LayerNorm(d_model)
        )
        
        # Output
        self.pre_output_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embedding' in name or 'output_projection' in name:
                    nn.init.normal_(p, mean=0, std=self.d_model ** -0.5)
                else:
                    nn.init.xavier_uniform_(p)
    
    def _create_audio_mask(self, src):
        """检测全为0的时间步 -> True=padding"""
        return (src.abs().sum(dim=-1) == 0)
    
    def _create_token_mask(self, tgt):
        """检测pad_token_id -> True=padding"""
        return (tgt == self.pad_token_id)
    
    def _adjust_mask_for_subsampling(self, mask, new_length):
        if mask is None:
            return None
        
        b, old_len = mask.size()
        mask_4d = mask.view(b, 1, 1, old_len).float()
        pooled = F.max_pool2d(mask_4d, kernel_size=(1, 4), stride=(1, 4))
        adjusted = pooled.view(b, -1).bool()
        
        # 调整长度
        curr_len = adjusted.size(1)
        if curr_len < new_length:
            adjusted = torch.cat([
                adjusted,
                torch.zeros(b, new_length - curr_len, dtype=torch.bool, device=adjusted.device)
            ], dim=1)
        elif curr_len > new_length:
            adjusted = adjusted[:, :new_length]
        
        return adjusted
    
    def encode(self, src, src_key_padding_mask=None):
        """
        编码音频特征
        Args:
            src: (B, T, n_mels)
            src_key_padding_mask: (B, T) - True=padding, 可选
        Returns:
            memory: (B, T', d_model)
        """
        # SpecAugment
        if self.training and self.spec_augment is not None:
            src = self.spec_augment(src)
        
        # Feature extraction
        src = self.feature_extractor(src)
        src = self.pos_encoder(src)
        
        # 调整 mask
        if src_key_padding_mask is not None and isinstance(self.feature_extractor, CNNSubsampling):
            src_key_padding_mask = self._adjust_mask_for_subsampling(src_key_padding_mask, src.size(1))
        
        # Encode
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        return memory
    
    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        解码目标序列
        Args:
            tgt: (B, L) - token IDs
            memory: (B, T', d_model)
            tgt_mask: (L, L) - causal mask, 可选
            tgt_key_padding_mask: (B, L) - True=padding, 可选
            memory_key_padding_mask: (B, T') - True=padding, 可选
        Returns:
            output: (B, L, vocab_size)
        """
        # Token embedding
        tgt_emb = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.embedding_dropout(tgt_emb)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        # Decode
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Output projection
        output = self.pre_output_dropout(output)
        output = self.output_projection(output)
        
        return output
    
    def forward(self, src, tgt):
        """
        前向传播 - 自动创建 padding mask
        Args:
            src: (B, T, n_mels) - padding=0
            tgt: (B, L) - padding=pad_token_id
        Returns:
            (B, L, vocab_size)
        """
        # 自动创建 masks
        src_mask = self._create_audio_mask(src)
        tgt_mask = self._create_token_mask(tgt)
        
        # Encode
        memory = self.encode(src, src_mask)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1
        ).bool()
        
        # 调整 memory mask
        memory_mask = src_mask
        if isinstance(self.feature_extractor, CNNSubsampling):
            memory_mask = self._adjust_mask_for_subsampling(memory_mask, memory.size(1))
        
        # Decode
        output = self.decode(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask
        )
        
        return output
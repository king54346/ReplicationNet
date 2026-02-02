Transformer ASR 的三种主流结构

Encoder-only（CTC）
Audio → Encoder → CTC → Text

Encoder–Decoder（Attention）
音频波形 → 特征提取(Mel频谱特征) → Encoder → Decoder → 文本输出


Encoder + CTC + Decoder（工业最常见）
Audio → Encoder → 
           ├─ CTC Loss
           └─ Decoder → Attention Loss


 Encoder结构示意图（6层Transformer为例）：
                    Input Features (B, T, F)
                            ↓
                    Linear Projection
                            ↓
                  (B, T, d_model=512)
                            ↓
                  + Positional Encoding
                            ↓
┌───────────────────────────────────────────────┐
│              Encoder Layer × N (6层)           │
│                                               │
│  ┌─────────────────────────────────────┐    │
│  │   Multi-Head Self-Attention         │    │
│  │   Q,K,V: (B,T,512)→(B,h=8,T,d_k=64) │    │
│  │   Attention: (B,h,T,T)              │    │
│  │   Output: (B,T,512)                 │    │
│  └─────────────────────────────────────┘    │
│              ↓ Residual + LayerNorm          │
│  ┌─────────────────────────────────────┐    │
│  │   Feed-Forward Network              │    │
│  │   (B,T,512)→(B,T,2048)→(B,T,512)    │    │
│  └─────────────────────────────────────┘    │
│              ↓ Residual + LayerNorm          │
│                                               │
└───────────────────────────────────────────────┘
                            ↓
              Encoder Output (B, T, 512)

Decoder结构:
Target Text Tokens (训练时)
                     ↓
           Token Embedding (vocab_size=5000,BPE嵌入)
                     ↓
              (B, S, d_model=512)
                     ↓
           + Positional Encoding
                     ↓
┌────────────────────────────────────────────────┐
│             Decoder Layer × N (6层)            │
│                                                │
│  ┌──────────────────────────────────────┐    │
│  │  Masked Multi-Head Self-Attention    │    │
│  │  Q,K,V from target: (B,S,512)        │    │
│  │  Mask: 下三角矩阵 (S,S)               │    │
│  │  Output: (B,S,512)                   │    │
│  └──────────────────────────────────────┘    │
│            ↓ Residual + LayerNorm             │
│  ┌──────────────────────────────────────┐    │
│  │  Cross Multi-Head Attention          │    │
│  │  Q from target: (B,S,512)            │    │
│  │  K,V from **encoder**: (B,T,512)     │    │
│  │  Attention: (B,h,S,T)                │    │
│  │  Output: (B,S,512)                   │    │
│  └──────────────────────────────────────┘    │
│            ↓ Residual + LayerNorm             │
│  ┌──────────────────────────────────────┐    │
│  │  Feed-Forward Network                │    │
│  │  (B,S,512)→(B,S,2048)→(B,S,512)      │    │
│  └──────────────────────────────────────┘    │
│            ↓ Residual + LayerNorm             │
└────────────────────────────────────────────────┘
                     ↓
          Linear + Softmax (B,S,vocab_size)
                     ↓
        Output Probabilities (B,S,5000)


┌──────────────────────────────────────────────────────────────┐  
│                      训练阶段                                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  音频输入                          文本输入(和encoder做交叉注意)  │
│  (B, T_wave)                     (B, S)                     │
│      ↓                              ↓                        │
│  Mel特征提取                      Token Embedding            │
│  (B, T, 80)                      (B, S, 512)               │
│      ↓                              ↓                        │
│  Linear + PE                     Positional Encoding        │
│  (B, T, 512)                     (B, S, 512)               │
│      ↓                              ↓                        │
│  ┌─────────────┐                ┌──────────────┐           │
│  │   Encoder   │                │   Decoder    │           │
│  │   6 Layers  │ ────Memory──→  │   6 Layers   │           │
│  │             │  (B, T, 512)   │              │           │
│  └─────────────┘                └──────────────┘           │
│                                      ↓                       │
│                                  Linear Layer               │
│                                  (B, S, vocab)              │
│                                      ↓                       │
│                                  Softmax                    │
│                                  (B, S, vocab)              │
│                                      ↓                       │
│                              Cross-Entropy Loss             │
│                              with Label (B, S)              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                      推理阶段                                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  音频输入 (1, T_wave)                                         │
│      ↓                                                       │
│  Mel特征 (1, T, 80)                                          │
│      ↓                                                       │
│  Encoder (1, T, 512) ────────┐                              │
│                              │                               │
│  初始Token [SOS] (1, 1)       │                              │
│      ↓                       │                               │
│  ┌─────────────────────────┐│                              │
│  │  循环解码过程             ││                              │
│  │  t=1: Decoder + Memory  ││ ← Encoder Memory             │
│  │       Output: (1,1,vocab)│                              │
│  │       Sample: token_1    │                               │
│  │                          │                               │
│  │  t=2: Input [SOS,token_1]│                              │
│  │       Output: (1,2,vocab)│                              │
│  │       Sample: token_2    │                               │
│  │                          │                               │
│  │  ...继续直到[EOS]或最大长度│                              │
│  └─────────────────────────┘                               │
│      ↓                                                       │
│  最终文本序列                                                 │
└──────────────────────────────────────────────────────────────┘

LRS2 dataset: https://aistudio.baidu.com/datasetdetail/228857
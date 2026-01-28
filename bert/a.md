BertForSequenceClassification(
  (bert): BertModel(              # ← BERT主体（12层Transformer，110M参数）
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(...)  # 12个Transformer层
      )
    )
    (pooler): BertPooler(...)
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(            # ← 分类头（768→2，仅1.5K参数）
    in_features=768, 
    out_features=2
  )
)

bert的全量微调： 
optimizer = optim.AdamW(model.parameters(), ...)  
#                      ↑ 包含BERT全部110M参数 + 分类头1.5K参数
model.save_pretrained(best_model_dir)
best_model/
├── pytorch_model.bin          # 包含所有权重（BERT + 分类头）
│   ├── bert.embeddings.weight          [30522, 768]
│   ├── bert.encoder.layer.0.xxx        [...] 
│   ├── ...
│   ├── bert.encoder.layer.11.xxx       [...] 
│   └── classifier.weight               [2, 768]  ← 分类头
│       classifier.bias                 [2]
├── config.json                # 模型配置
├── vocab.txt                  # 词表
└── model_info.json           # 你自己保存的训练信息

from transformers import (
    BertModel,                      # 基础BERT（无任务头）
    BertForSequenceClassification,  # 你用的：文本分类
    BertForTokenClassification,     # 命名实体识别(NER)、词性标注
    BertForQuestionAnswering,       # 阅读理解/问答
    BertForMaskedLM,               # 掩码语言模型（预训练任务）
    BertForMultipleChoice,         # 多项选择题
    BertForNextSentencePrediction, # 句子关系预测
    BertForPreTraining,            # 组合MLM+NSP的预训练
)
BertForSequenceClassification序列分类,BertForTokenClassification词元分类,BertForQuestionAnswering问答任务,BertForMaskedLM掩码语言建模,BertForMultipleChoice多项选择
BertForNextSentencePrediction句子对是否连续
#  BertForSequenceClassification输出
# {
#     'last_hidden_state': tensor,  # [batch_size, seq_len, hidden_size] = [1, 4, 768] 每个token的最后一层Transformer层的输出
#     'pooler_output': tensor,  # [batch_size, hidden_size] = [1, 768] 句子表示的向量[CLS]的表示（存在于有BertPooler层）
#     'hidden_states': tuple,  # 所有层的隐藏状态（如果output_hidden_states=True）
#     'attentions': tuple  # 所有层的注意力权重（如果output_attentions=True）
# }
#  任务特定模型的输出,每个特定任务的输出格式可能不同，但通常包含loss和logits
# {
#     'loss': tensor,                 # 计算的损失值
#     'logits': tensor,              # 分类得分 [batch_size, num_labels]
#     'hidden_states': tuple,        # 基础模型的所有层隐藏状态（可选）
#     'attentions': tuple           # 基础模型的注意力权重（可选）
#     'start_logits'/'end_logits' BertForQuestionAnswering专属 从一段文字中找到答案的位置
# }

BertModel（基础编码器）
BertModel(
(embeddings): BertEmbeddings(...)      # 词嵌入+位置嵌入+类型嵌入
(encoder): BertEncoder(                # 12层Transformer
   (layer): ModuleList(
   (0-11): 12 x BertLayer(...)
  )
)
(pooler): BertPooler(                  # [CLS] token池化层
    (dense): Linear(768, 768)
    (activation): Tanh()
 )
)

BertForSequenceClassification(
  (bert): BertModel(...)           # 基础编码器
  (dropout): Dropout(p=0.1)
  (classifier): Linear(768, num_labels)  # 分类头：768 → num_labels
)

BertForTokenClassification(
  (bert): BertModel(...)
  (dropout): Dropout(p=0.1)
  (classifier): Linear(768, num_labels)  # 每个token都分类
)

BertForQuestionAnswering(
  (bert): BertModel(...)
  (qa_outputs): Linear(768, 2)  # 输出start和end两个logits
)


 BERT - 经典Transformer Encoder
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)  # ← 句子A/B区分
      (LayerNorm): LayerNorm(768)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(...)
          (intermediate): BertIntermediate(...)  # FFN 768→3072
          (output): BertOutput(...)              # FFN 3072→768
        )
      )
    )
  )
  (dropout): Dropout(0.1)
  (classifier): Linear(768, 2)
)

总参数: ~110M

[//]: # (RoBERTa/ELECTRA: 多一层变换，BERT是直接线性层输出)
RoBERTa - BERT的改进版 hfl/chinese-roberta-wwm-ext（哈工大RoBERTa）(RoBERTa使用了与bert-base-chinese相同的词表所以tokenizer兼容，可以直接用BertTokenizer)
RobertaForSequenceClassification(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(
      (word_embeddings): Embedding(50265, 768)  # ← 词表更大
      (position_embeddings): Embedding(514, 768)
      # ❌ 没有token_type_embeddings！
      (LayerNorm): LayerNorm(768)
    )
    (encoder): RobertaEncoder(
      # 结构与BertEncoder完全相同
      (layer): ModuleList(...)
    )
  )
  (classifier): RobertaClassificationHead(  # ← 分类头结构不同
    (dense): Linear(768, 768)
    (dropout): Dropout(0.1)
    (out_proj): Linear(768, 2)
  )
)

总参数: ~125M（词表更大导致）

ELECTRA - 判别器架构
ElectraForSequenceClassification(
  (electra): ElectraModel(
    (embeddings): ElectraEmbeddings(
      # 与BERT相同
      (word_embeddings): Embedding(21128, 768)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
    )
    (encoder): ElectraEncoder(
      # 与BERT相同
      (layer): ModuleList(...)
    )
  )
  (classifier): ElectraClassificationHead(  # ← 与RoBERTa相似
    (dense): Linear(768, 768)
    (dropout): Dropout(0.1)
    (out_proj): Linear(768, 2)
  )
)

总参数: ~110M

ALBERT - 参数共享版
AlbertForSequenceClassification(
  (albert): AlbertModel(
    (embeddings): AlbertEmbeddings(
      (word_embeddings): Embedding(21128, 128)    # ← 128维（不是768！）
      (position_embeddings): Embedding(512, 128)
      (token_type_embeddings): Embedding(2, 128)
      (LayerNorm): LayerNorm(128)
    )
    (encoder): AlbertTransformer(
      (embedding_hidden_mapping_in): Linear(128, 768)  # ← 升维到768
      (albert_layer_groups): ModuleList(
        (0): AlbertLayerGroup(
          (albert_layers): ModuleList(
            (0): AlbertLayer(...)  # ← 只有1组参数，重复使用12次！
          )
        )
      )
    )
  )
  (dropout): Dropout(0.1)
  (classifier): Linear(768, 2)
)

总参数: ~12M（压缩10倍！）


HuggingFace会自动识别config.json中的model_type
HuggingFace的AutoModel机制
# <class 'transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification'>
# ↑ 自动识别为RoBERTa类 使用 
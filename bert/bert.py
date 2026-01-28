"""
BERT 完整训练和推理代码
包含：预训练(MLM+NSP)、下游任务微调、推理
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import os


# ============================================
# 1. 数据集定义
# ============================================
#  补词+ 判断两句话是否连着
class BertPreTrainDataset(Dataset):
    """BERT预训练数据集 - MLM + NSP"""

    def __init__(self, texts: List[str], tokenizer, max_length=512, mlm_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def create_mlm_labels(self, input_ids):
        """创建MLM的labels，随机mask 15%的token"""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)

        # 不mask特殊token [CLS], [SEP], [PAD]
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算masked位置的loss

        # 80% 替换为[MASK], 10%随机替换, 10%保持不变
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return input_ids, labels

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        token_type_ids = encoding['token_type_ids'].squeeze(0)

        # 创建MLM labels
        input_ids, labels = self.create_mlm_labels(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }


class BertClassificationDataset(Dataset):
    """下游分类任务数据集"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================
# 2. 模型定义
# ============================================

class BertForPreTraining(nn.Module):
    """BERT预训练模型 - MLM + NSP"""

    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.cls = nn.Linear(config.hidden_size, config.vocab_size)  # MLM head
        self.nsp = nn.Linear(config.hidden_size, 2)  # NSP head (二分类)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, next_sentence_label=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]

        # MLM预测
        mlm_logits = self.cls(sequence_output)  # [batch, seq_len, vocab_size]

        # NSP预测
        nsp_logits = self.nsp(pooled_output)  # [batch, 2]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.size(-1)), labels.view(-1))

            if next_sentence_label is not None:
                nsp_loss = loss_fct(nsp_logits, next_sentence_label)
                loss = mlm_loss + nsp_loss
            else:
                loss = mlm_loss

        return {
            'loss': loss,
            'mlm_logits': mlm_logits,
            'nsp_logits': nsp_logits
        }


class BertForSequenceClassification(nn.Module):
    """BERT下游分类任务"""

    def __init__(self, config, num_labels):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        pooled_output = outputs.pooler_output  # [CLS] token的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


# ============================================
# 3. 训练器
# ============================================

class BertTrainer:
    """BERT训练器 - 支持混合精度、梯度累积"""

    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader=None,
            learning_rate=5e-5,
            epochs=3,
            warmup_steps=0,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            device='cuda',
            use_amp=True,  # 混合精度训练
            save_dir='./checkpoints'
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir

        # 优化器
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        # 学习率调度器
        total_steps = len(train_dataloader) * epochs // gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 混合精度训练
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc='Training')

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs['loss']

                # 梯度累积
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

            # 梯度累积步数到达时更新参数
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def evaluate(self):
        """验证"""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_dataloader, desc='Validation'):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs['loss'].item()

        return total_loss / len(self.val_dataloader)

    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')

        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')

            # 训练
            train_loss = self.train_epoch()
            print(f'Train Loss: {train_loss:.4f}')

            # 验证
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                print(f'Val Loss: {val_loss:.4f}')

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f'best_model_epoch_{epoch + 1}.pt')

            # 保存checkpoint
            self.save_model(f'checkpoint_epoch_{epoch + 1}.pt')

    def save_model(self, filename):
        """保存模型"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
        print(f'Model saved to {save_path}')


# ============================================
# 4. 推理器
# ============================================

class BertInference:
    """BERT推理"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def predict(self, texts: List[str], max_length=128, batch_size=32):
        """批量推理"""
        all_predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            # 移动到设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)

            # 推理
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy().tolist())

        return all_predictions

    @torch.no_grad()
    def predict_proba(self, texts: List[str], max_length=128):
        """预测概率分布"""
        encoding = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        logits = outputs['logits']
        probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy()

    @torch.no_grad()
    def get_embeddings(self, texts: List[str], max_length=128):
        """获取句子的embedding ([CLS] token)"""
        encoding = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        # 直接调用BERT获取hidden states
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 返回[CLS] token的embedding
        embeddings = outputs.pooler_output

        return embeddings.cpu().numpy()


# ============================================
# 5. 使用示例
# ============================================

def example_pretrain():
    """预训练示例"""
    # 初始化
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    config = BertConfig.from_pretrained('bert-base-chinese')
    model = BertForPreTraining(config)

    # 准备数据
    texts = [
                "今天天气真好，适合出去玩。",
                "机器学习是人工智能的重要分支。",
                # ... 更多文本
            ] * 100  # 模拟大量数据

    dataset = BertPreTrainDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练
    trainer = BertTrainer(
        model=model,
        train_dataloader=dataloader,
        learning_rate=5e-5,
        epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.train()


def example_finetune_classification():
    """下游分类任务微调示例"""
    # 初始化
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    config = BertConfig.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification(config, num_labels=2)  # 二分类

    # 准备数据
    train_texts = ["这个电影真好看", "太烂了，浪费时间"] * 100
    train_labels = [1, 0] * 100  # 1: 正面, 0: 负面

    val_texts = ["非常精彩", "不推荐"] * 20
    val_labels = [1, 0] * 20

    train_dataset = BertClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = BertClassificationDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 训练
    trainer = BertTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=2e-5,
        epochs=3,
        warmup_steps=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    trainer.train()

    return model, tokenizer


def example_inference():
    """推理示例"""
    # 加载模型（假设已经训练好）
    config = BertConfig.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification(config, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 加载权重
    # checkpoint = torch.load('checkpoints/best_model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # 推理
    inference = BertInference(model, tokenizer)

    # 预测类别
    test_texts = ["这个产品质量很好", "太差劲了", "一般般吧"]
    predictions = inference.predict(test_texts)
    print("Predictions:", predictions)

    # 预测概率
    probs = inference.predict_proba(test_texts)
    print("Probabilities:", probs)

    # 获取embeddings
    embeddings = inference.get_embeddings(test_texts)
    print("Embeddings shape:", embeddings.shape)


if __name__ == '__main__':
    # 选择运行哪个示例
    # example_pretrain()
    # model, tokenizer = example_finetune_classification()
    # example_inference()

    print("BERT训练和推理代码准备完成！")
    print("\n使用说明：")
    print("1. 预训练: example_pretrain()")
    print("2. 微调分类任务: example_finetune_classification()")
    print("3. 推理: example_inference()")
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


import json
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.optim import AdamW, Adam
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm.auto import tqdm
from datetime import datetime
import logging
import wandb
import optuna
from muon import Muon, get_muon_optimizer

# 导入本地模块
from config import Config, ml_type_encoder
from data import split_data, get_data_loaders
from model import get_model

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def set_seed(seed):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
def train_epoch(model, train_loader, optimizer, scheduler, device, encoder_type, epoch: int, config: Config):
    """训练一个epoch"""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []

    optimizer.zero_grad() # Initialize gradients

    progress_bar = tqdm(train_loader, desc="训练中")
    
    for step, batch in enumerate(progress_bar):
        # 根据编码器类型处理输入
        if encoder_type == "bert":
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentences = None
        else:  # sentence-transformer
            input_ids = None
            attention_mask = None
            sentences = batch['sentences']  # 句子已经是列表形式

        topics = batch['topic']
        statistical_features = batch['statistical_features'].to(device)
        labels = batch['label'].to(device)
        images = batch['images'] #.to(device)
        image_masks = batch['image_masks'] #.to(device)
        if images is not None:
            images = images.to(device)
        if image_masks is not None:
            image_masks = image_masks.to(device)
        
        # 前向传播
        # optimizer.zero_grad() # Moved to accumulation step
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            statistical_features=statistical_features,
            sentences=sentences,
            images=images,
            image_masks=image_masks,
            topic_names=topics if config.ADD_TOPIC_NAME else None, # a list of topic names
        )
        
        # 计算损失
        if hasattr(config, 'USE_FOCAL_LOSS') and config.USE_FOCAL_LOSS:
            loss_fn = FocalLoss(
                alpha=getattr(config, 'FOCAL_LOSS_ALPHA', 1.0),
                gamma=getattr(config, 'FOCAL_LOSS_GAMMA', 2.0)
            )
        else:
            if config.WEIGHTED_LOSS:
                # 使用类别权重的交叉熵损失
                class_counts = np.array(config.CLASS_RATIO)
                class_weights = 1.0 / class_counts
                class_weights = class_weights / class_weights.sum() * len(class_counts)  # 归一化权重
                class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.LABEL_SMOOTHING)  # 使用标签平滑
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)  # 使用标签平滑
        
        loss = loss_fn(outputs, labels)
        
        # Normalize loss for gradient accumulation
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        loss_item = loss.item() * config.GRADIENT_ACCUMULATION_STEPS # Scale back for logging

        # Log stepwise loss
        # wandb.log({"train/step_loss": loss_item, "train/step": step + epoch * len(train_loader)})

        progress_bar.set_postfix({'loss': loss_item})
        train_loss += loss_item
        
        # 反向传播
        loss.backward()
        
        # Gradient accumulation step
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 记录预测结果
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    
    return train_loss, train_acc, train_f1, train_preds, train_labels

def evaluate(model, val_loader, device, encoder_type, config=None):
    """评估模型"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    topics = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中"):
            # 根据编码器类型处理输入
            if encoder_type == "bert":
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sentences = None
            else:  # sentence-transformer
                input_ids = None
                attention_mask = None
                sentences = batch['sentences']  # 句子已经是列表形式
            
            statistical_features = batch['statistical_features'].to(device)
            labels = batch['label'].to(device)
            images = batch['images'] #.to(device)
            image_masks = batch['image_masks'] #.to(device)
            if images is not None:
                images = images.to(device)
            if image_masks is not None:
                image_masks = image_masks.to(device)

            topics = batch['topic']
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                statistical_features=statistical_features,
                sentences=sentences,
                images=images,
                image_masks=image_masks,
                topic_names=topics if config.ADD_TOPIC_NAME else None,
            )
            
            # 计算损失
            if config and hasattr(config, 'USE_FOCAL_LOSS') and config.USE_FOCAL_LOSS:
                loss_fn = FocalLoss(
                    alpha=getattr(config, 'FOCAL_LOSS_ALPHA', 1.0),
                    gamma=getattr(config, 'FOCAL_LOSS_GAMMA', 2.0)
                )
            else:
                loss_fn = nn.CrossEntropyLoss()

            # print(outputs, labels)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            
            # 记录预测结果
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy().astype(int).tolist())
            val_labels.extend(labels.cpu().numpy().astype(int).tolist())
            topics.extend(batch['topic'])
    
    # 计算指标
    val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    val_precision = precision_score(val_labels, val_preds, average='macro')
    val_recall = recall_score(val_labels, val_preds, average='macro')
    
    # 存储预测结果
    predictions = {
        'topic': topics,
        'true': val_labels,
        'pred': val_preds
    }
    
    return val_loss, val_acc, val_f1, val_precision, val_recall, predictions

def plot_confusion_matrix(true_labels, pred_labels, classes, fig_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('GT Label')
    plt.xlabel('Prediction Label')
    plt.savefig(fig_path)
    plt.close()

def plot_training_curves(history, save_path, title="Model Training Performance"):
    """绘制训练曲线，增强可视化效果"""
    # 设置更现代的绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 创建更高分辨率的图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # 定义更好看的颜色
    train_color = '#1f77b4'  # 蓝色
    val_color = '#ff7f0e'    # 橙色
    
    # 绘制损失曲线
    axes[0].plot(history['train_loss'], label='Train', color=train_color, 
                marker='o', markersize=4, linestyle='-', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', color=val_color, 
                marker='s', markersize=4, linestyle='-', linewidth=2)
    axes[0].set_title('Loss Curves', fontsize=14, pad=10)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss Value', fontsize=12)
    axes[0].legend(frameon=True, fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 绘制F1分数曲线
    axes[1].plot(history['train_f1'], label='Train', color=train_color, 
                marker='o', markersize=4, linestyle='-', linewidth=2)
    axes[1].plot(history['val_f1'], label='Validation', color=val_color, 
                marker='s', markersize=4, linestyle='-', linewidth=2)
    axes[1].set_title('F1 Score Curves', fontsize=14, pad=10)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].legend(frameon=True, fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 设置y轴范围（对F1分数）
    axes[1].set_ylim([0, 1.05])
    
    # 添加更多细节
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
    
    # 调整布局并保存
    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def train_model(config: Config, trial: optuna.Trial = None):
    """训练模型"""
    # scale learning rate based on gradient accumulation
    if config.OPTIMIZER_NAME != "muon":
        config.LEARNING_RATE = config.LEARNING_RATE * (config.GRADIENT_ACCUMULATION_STEPS ** 0.5)
    else:
        config.LEARNING_RATE = config.LEARNING_RATE * config.GRADIENT_ACCUMULATION_STEPS

    

    model_name = config.BERT_MODEL if config.ENCODER_TYPE == "bert" else config.SENTENCE_TRANSFORMER_MODEL
    model_dir = os.path.join(config.MODEL_DIR, f"{config.MODEL_TYPE}_{model_name}_{'FROZEN' if config.FREEZE_ENCODER else 'FINE-TUNED'}", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"模型目录: {model_dir}")

    run_name = os.path.basename(model_dir)
    if trial:
        run_name = f"trial_{trial.number}_{run_name}"

    wandb.init(
        project="yuqing_emergency",
        config=config.__dict__,
        name=run_name,
        group="optuna_parallel_search_v2" if trial else None, # 使用 group 聚合
        reinit=True
    )

    # 设置设备
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 如果模型目录不存在则创建
    os.makedirs(model_dir, exist_ok=True)
    
    # 如果data_split.json不存在则分割数据
    # data_split_path = os.path.join(config.ANNOTATION_DIR, 'data_split.json')
    data_split_path = config.DATA_SPLIT_FILE
    if not os.path.exists(data_split_path):
        print("分割数据...")
        train_topics, val_topics = split_data(
            config.DATA_DIR, config.EMERGENCY_LEVEL_FILE, 
            train_ratio=config.TRAIN_RATIO, random_state=config.SEED
        )
    else:
        print("加载已有的数据分割...")
        with open(data_split_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
            train_topics = data_split['train_topics']
            val_topics = data_split['val_topics']
    
    # 初始化分词器和编码器
    if config.ENCODER_TYPE == "bert":
        tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
    else:  # sentence-transformer
        # 对于sentence-transformer，此处设置为None，因为SentenceTransformer自己处理分词
        tokenizer = None
    
    # 打印所选编码器
    print(f"使用Tokenizer: {config.ENCODER_TYPE}")
    if config.ENCODER_TYPE == "bert":
        print(f"BERT模型: {config.BERT_MODEL}")
    else:
        print(f"Sentence-Transformer模型: {config.SENTENCE_TRANSFORMER_MODEL}")
    
    # 创建数据加载器
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders(
        config, train_topics, val_topics, config.DATA_DIR, config.EMERGENCY_LEVEL_FILE, tokenizer
    )
    
    # 加载预警级别确定类别数
    num_classes = len(set(train_dataset.labels))
    
    # 获取统计特征数量（从第一个样本）
    for batch in train_loader:
        num_stat_features = batch['statistical_features'].shape[1]
        break
    
    # 创建模型
    model = get_model(config, num_classes, num_stat_features)
    model.to(device)

    # 打印BERT参数状态
    # TODO: merge BERT and Sentence-Transformer, and Two Model types

    # 打印模型总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义优化器和调度器
    if config.OPTIMIZER_NAME == "muon":
        print("Using Muon optimizer")
        optimizer = get_muon_optimizer(
            config.OPTIMIZER_NAME,
            model,
            lr=config.LEARNING_RATE,
            wd=config.WEIGHT_DECAY
        )
    else:
        print("Using AdamW optimizer")
        # only optimize trainable parameters
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS // config.GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=total_steps // config.GRADIENT_ACCUMULATION_STEPS
    )
    
    # 训练循环
    best_val_f1 = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练
        train_loss, train_acc, train_f1, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, device, config.ENCODER_TYPE, epoch, config
        )
        
        # 评估
        val_loss, val_acc, val_f1, val_precision, val_recall, predictions = evaluate(
            model, val_loader, device, config.ENCODER_TYPE, config=config
        )

        # --- 新增：Optuna 剪枝逻辑 ---
        if trial:
            # Hyperband 需要知道当前的资源消耗量 (epoch)
            trial.report(val_f1, epoch)
            # 如果当前 trial 的表现明显差于之前的 trial，则提前终止
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}")
                wandb.finish()
                raise optuna.TrialPruned()
        # ---------------------------
        
        # 保存指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Log to wandb
        wandb.log({
            "train/loss": train_loss,
            "train/acc": train_acc,
            "train/f1": train_f1,
            "val/loss": val_loss,
            "val/acc": val_acc,
            "val/f1": val_f1,
            "val/precision": val_precision,
            "val/recall": val_recall,
        }, step=epoch+1)

        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"验证 - Prec: {val_precision:.4f}, Recall: {val_recall:.4f}")

        # always save Config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            # json.dump(config.__dict__, f, ensure_ascii=False, indent=2)
            json.dump(asdict(config), f, default=ml_type_encoder, ensure_ascii=False, indent=2)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"达到最佳F1分数{best_val_f1:.4f}")

            if config.DO_SAVE:
                # 保存模型
                model_path = os.path.join(model_dir, f"{config.MODEL_TYPE}_best.pt")
                torch.save(model.state_dict(), model_path)
                
                # 保存分词器
                # tokenizer_path = os.path.join(model_dir, "tokenizer")
                # tokenizer.save_pretrained(tokenizer_path)
                
                # 保存预测结果
                predictions_path = os.path.join(model_dir, "best_predictions.json")
                with open(predictions_path, 'w', encoding='utf-8') as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2)
                
                print(f"保存最佳模型，F1: {val_f1:.4f}")
    
        # 绘制训练曲线, each epoch
        plot_training_curves(history, os.path.join(model_dir, f"{config.MODEL_TYPE}_training_curves.png"))
    
    # 绘制最佳模型的混淆矩阵
    classes = sorted(list(set(train_dataset.labels)))
    plot_confusion_matrix(
        predictions['true'], 
        predictions['pred'], 
        classes,
        os.path.join(model_dir, f"{config.MODEL_TYPE}_confusion_matrix.png")
    )
    
    # 保存训练历史
    with open(os.path.join(model_dir, f"{config.MODEL_TYPE}_history.json"), 'w', encoding='utf-8') as f:
        json.dump(history, f)
    
    print(f"训练完成! 最佳验证F1分数: {best_val_f1:.4f}")
    
    wandb.finish()

    # return model
    return best_val_f1

if __name__ == "__main__":
    log_format = (
        "[%(asctime)s %(name)s %(levelname)s] "
        "%(message)s"
    )

    logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")
    transformers.utils.logging.set_verbosity_info()

    # 加载配置
    config = Config()
    
    # 训练模型
    best_val_f1 = train_model(config)

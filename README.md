# Yuqing - 舆情分析与应急等级评估系统

## 模型概述
这是一个基于深度学习的舆情数据分析项目，旨在对微博热搜等舆情事件进行分析和舆情紧急等级评估，支持多模态数据（文本+图像）。模型通过多模态贴文级信息聚合和事件级多贴文特征交互建模技术，实现热点事件的舆情评估预警功能。

### 核心功能
- 算法输入: 事件及其相关社交媒体帖子集合(每条帖子为纯文本或文本+图像)
- 算法输出: 事件舆情预警等级
- 分类体系: 轻微，一般，严重，分别对应1到3级

### 技术亮点
通过引入贴文级注意力机制，模型能够捕捉事件内多条贴文之间的上下文关系和语义关联，学习综合衡量事件内部信息关联性与语义分布特征，以提升舆情等级评估的准确性和鲁棒性。

## 目录结构

```
<repo_root>/
├── annotation/             # 标注数据和数据集划分文件
│   ├── data_split.json     # 训练/验证集划分
|   ├── cleaned_data/       # 清洗后的舆情数据
│   └── ...
├── data/                   # 舆情数据目录
│   ├── event1/             # 事件1数据
│   ├── event2/             # 事件2数据
│   └── ...
├── model/                  # 模型保存目录 (包含权重、配置和训练曲线)
├── config.py               # 全局配置文件 (核心)
├── data.py                 # 数据加载、预处理 (HarvestText清洗, Emoji处理)
├── model.py                # 模型架构定义 (TextEncoder, TransformerModel)
├── train_emergency.py      # 主训练脚本
├── inference.py            # 推理接口
├── run.sh                  # 启动脚本
└── ...
```

## 环境依赖

主要依赖库：
- Python 3.12
- PyTorch
- Transformers & Sentence-Transformers
- Pandas, NumPy, Scikit-learn
- HarvestText & Jieba (文本清洗)
- WandB (实验监控)
- Optuna (超参数搜索)
- Icecream (调试)

## 数据格式

事件数据请在 [这里](https://drive.google.com/file/d/1ynCLRoA5By8eicsQzq9iqCu4R6kY6EB2/view?usp=sharing) 下载并解压到`data/`文件夹下

每个事件的数据存储在 `data/<事件名>` 目录下的 CSV 文件中。每个 CSV 文件包含以下关键列：
<!-- id,bid,user_id,用户昵称,微博正文,头条文章url,发布位置,艾特用户,话题,转发数,评论数,点赞数,发布时间,发布工具,微博图片url,微博视频url,retweet_id,ip,user_authentication,processed_text -->

| 列名           | 说明                     |
| :------------- | :----------------------- |
| `微博正文`     | 微博的文本内容           |
| `微博图片url`  | 微博中包含的图片链接     |
| `话题`         | 该微博所属的话题         |
| `发布时间`     | 微博发布的时间戳         |
| `发布位置`     | 微博发布的地理位置         |

第一次运行后会自动生成清洗后的数据文件，存放在 `annotation/cleaned_data/` 目录下。

标注文件则在 `annotation/data_split.json` 中定义，格式如下：

```json
{
    "train_tpoics": ["event1", "event2", ...],
    "val_tpoics": ["event3", ...],
}
```

其标注文件位于 `annotation/微博热搜25-1-1to25-1-8_filtered_event_names_result.json`，格式如下：

```json
{
    {
        "topic": "西藏定日县地震已致126人遇难",
        "reason": "重大自然灾害造成大量人员伤亡，涉及生命安全和灾后重建等重大民生问题",
        "severity": "严重"
    },
  ...
}
```

## 配置说明 (`config.py`)

项目的所有配置均在 `config.py` 的 `Config` 类中定义。可以直接修改该文件来调整实验设置。

### 关键参数

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `DATA_TYPE` | `"original"` | 数据源选择 (`original` 或 `new`) |
| `ENCODER_TYPE` | `"sentence-transformer"` | 文本编码器类型 (`bert` 或 `sentence-transformer`) |
| `SENTENCE_TRANSFORMER_MODEL` | `"Alibaba-NLP/gte-multilingual-base"` | Sentence-Transformer 模型路径或名称 |
| `OPTIMIZER_NAME` | `"muon"` | 优化器选择 (`muon` 或 `adamw`) |
| `POOLING_STRATEGY` | `"max"` | 推文聚合策略 (`mean`, `max`, `cls`, `attentive`) |
| `BATCH_SIZE` | `4` | 批处理大小 (针对事件) |
| `GRADIENT_ACCUMULATION_STEPS` | `32` | 梯度累积步数 |

## 使用方法

### 1. 数据准备
- 将清洗后的 CSV 数据文件放置在 `cleaned_data` 目录中。
- 确保 `annotation` 目录下有正确的数据划分文件 (`data_split.json`) 和标签文件。

### 2. 训练模型

使用 `run.sh` 或直接运行 Python 脚本：

```bash
# 设置环境变量（推荐使用 HF 镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 运行训练
python train_emergency.py
```

### 3. 关闭 WandB Logging

默认情况下，训练脚本会尝试连接 Weights & Biases 进行日志记录。如果你不需要此功能，或者在没有网络的环境下运行，可以通过设置环境变量 `WANDB_MODE` 来关闭它：

**Linux / macOS:**
```bash
export WANDB_MODE=disabled
python train_emergency.py
```

**Windows (PowerShell):**
```powershell
$env:WANDB_MODE="disabled"
python train_emergency.py
```

或者在 `run.sh` 中添加：
```bash
export WANDB_MODE=disabled
```

### 4. 模型推理

使用 `inference.py` 中的 `TopicClassifier` 进行预测：

```python
from inference import TopicClassifier

# 加载模型
classifier = TopicClassifier(model_path='./model/best_model_path')

# 预测
results = classifier.predict(["某地发生火灾...", "救援正在进行..."])
```

## 进阶功能

## 核心特性

- **分层模型架构**：
  - **Tweet Encoder**：支持 BERT (`bert-base-chinese`) 或 Sentence-Transformer (`gte-multilingual-base` 等) 对单条推文进行编码。
  - **Transformer Aggregator**：使用 Transformer Encoder 处理事件内的推文序列，捕捉推文间的上下文关系。
  - **Pooling Layer**：支持多种池化策略 (`mean`, `max`, `cls`, `attentive`) 将推文序列聚合为事件向量。
- **多模态支持**：可支持融合图像特征与文本特征，进行更全面的舆情分析。
- **高级训练策略**：
  - **优化器**：支持标准的 AdamW 和高效的 **Muon** 优化器。
  - **损失函数**：集成 Focal Loss 和加权交叉熵，有效解决类别不平衡问题。
  - **数据增强**：支持训练时随机丢弃推文 (`DROP_TWEET_RATIO`) 以增强模型鲁棒性。
- **实验追踪**：集成 WandB (Weights & Biases) 进行实时训练监控和可视化。

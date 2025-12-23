from dataclasses import dataclass, field
from typing import Union, List, Optional
import torch

@dataclass
class DataConfig:
    DATA_DIR: str = "data"
    DATA_SPLIT_FILE: str = "annotation/data_split.json"
    EMERGENCY_LEVEL_FILE: str = "annotation/微博热搜25-1-1to25-1-8_filtered_event_names_result.json"
    CLASS_RATIO: List[float] = field(default_factory=lambda: [28, 50, 29])  # 原始数据类别比例

@dataclass
class DataConfigNew:
    DATA_DIR: Union[str, List[str]] = field(default_factory=lambda: [
        "data",                         # original data
        "data_new/cleaned_data",        # new 300 data
        "data_new/cleaned_data_wuhan",  # wuhan data
        "/network_space/server127_2/shared/mwt/话题updates/646_746/"    # leiting's data
    ])
    DATA_SPLIT_FILE: str = "annotation/data_split_new.json"
    EMERGENCY_LEVEL_FILE: str = "annotation/data_split_new_result.json"
    CLASS_RATIO: List[float] = field(default_factory=lambda: [122+403, 310, 170])  # 原始数据类别比例


@dataclass
class Config:
    # 数据参数
    ANNOTATION_DIR: str = "annotation"
    DATA_TYPE: str = "original"  # "original" 或 "new"

    DATA_DIR: str = field(default_factory=lambda: DataConfig.DATA_DIR if Config.DATA_TYPE == "original" else DataConfigNew().DATA_DIR)
    DATA_SPLIT_FILE: str = field(default_factory=lambda: DataConfig.DATA_SPLIT_FILE if Config.DATA_TYPE == "original" else DataConfigNew().DATA_SPLIT_FILE)
    EMERGENCY_LEVEL_FILE: str = field(default_factory=lambda: DataConfig.EMERGENCY_LEVEL_FILE if Config.DATA_TYPE == "original" else DataConfigNew().EMERGENCY_LEVEL_FILE)
    CLASS_RATIO: List[float] = field(default_factory=lambda: DataConfig().CLASS_RATIO if Config.DATA_TYPE == "original" else DataConfigNew().CLASS_RATIO)

    # DATA_DIR: str = "data"
    # DATA_SPLIT_FILE: str = "annotation/data_split.json"
    # EMERGENCY_LEVEL_FILE: str = "annotation/微博热搜25-1-1to25-1-8_filtered_event_names_result.json"

    # DATA_DIR: Union[str, List[str]] = field(default_factory=lambda: [
    #     "data",                         # original data
    #     "data_new/cleaned_data",        # new 300 data
    #     "data_new/cleaned_data_wuhan",  # wuhan data
    #     "/network_space/server127_2/shared/mwt/话题updates/646_746/"    # leiting's data
    # ])
    # DATA_SPLIT_FILE: str = "annotation/data_split_new.json"
    # EMERGENCY_LEVEL_FILE: str = "annotation/data_split_new_result.json"

    MODEL_DIR: str = "model" # 模型保存路径
    # EMERGENCY_LEVEL_FILE: str = "annotation/fullinput_gemini_2.5_pro.json"

    TRAIN_RATIO: float = 0.8
    
    # 模型参数
    MAX_TWEETS: int = 2000    # 每个话题最多处理的推文数
    MAX_LEN: int = 300       # 每条推文的最大长度

    ENCODER_TYPE: str = "sentence-transformer"  # "bert" 或 "sentence-transformer"
    BERT_MODEL: str = "./bert-base-chinese"

    SENTENCE_TRANSFORMER_MODEL: str = "Alibaba-NLP/gte-multilingual-base"
    SENTENCE_TRANSFORMER_DTYPE: Optional[torch.dtype] = None  # torch.float16  # or None for default dtype
    # 优质中文选项: 
    #   "distiluse-base-multilingual-cased-v1", 
    #   "paraphrase-multilingual-MiniLM-L12-v2", 
    #   "paraphrase-multilingual-mpnet-base-v2"
    # Modern sentence transformer models:
    #  "Alibaba-NLP/gte-multilingual-base" #　560M params - may OOM when training
    #  "thenlper/gte-large"
    #  "intfloat/multilingual-e5-large"
    FREEZE_ENCODER: bool = True  # 是否冻结编码器参数
    # NOTE: decide in the code from the actual model
    # EMBEDDING_SIZE: int = 768  # BERT的嵌入维度是768，不同的sentence-transformer模型维度可能不同

    ENCODER_SIZE: int = 768 # 768
    FEEDFORWARD_MULTIPLIER: int = 2  # Transformer前馈网络维度相对于ENCODER_SIZE的倍数
    NUM_ATTN_HEADS: int = 12  # Transformer中的注意力头数
    NUM_TRANSFORMER_LAYERS: int = 6  # Transformer层数
    DROPOUT_RATE: float = 0.15
    MODEL_TYPE: str = "transformer"  # "attentive_pooling" 或 "transformer"
    DROP_TWEET_RATIO: float = 0  # 训练时随机丢弃部分推文的概率
    POOLING_STRATEGY: str = "max"
    USE_STAT_FEATURES: bool = False  # 是否使用统计特征

    USE_IMAGES: bool = False  # 是否使用图片特征
    FREEZE_IMAGE_ENCODER: bool = True  # 是否冻结图片编码器参数
    IMAGE_ENCODER_TYPE: str = "vit"  # "resnet" 或 "vit"
    IMAGE_BATCH_SIZE = 32
    TEXT_ONLY_ADAPT: bool = False # add an aptation layer to the text only embedding
    
    # 训练参数
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 32
    SENTENCE_BATCH_SIZE: int = 512
    EPOCHS: int = 30
    LEARNING_RATE: float = 1e-4
    WARMUP_STEPS: int = 64
    WEIGHT_DECAY: float = 0.1
    LABEL_SMOOTHING: float = 0.1
    WEIGHTED_LOSS: bool = True  # 是否使用加权交叉熵损失
    USE_FOCAL_LOSS: bool = False
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: float = 0.25

    NUM_WORKERS: int = 16
    SEED: int = 4242
    DEVICE: str = "cuda"
    DO_SAVE: bool = False

    # SVD parameters
    SVD_COMPONENTS = 64  # Number of components to keep after SVD
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
    SVM_GAMMA = 'scale'

    ADD_TOPIC_NAME: bool = False

    OPTIMIZER_NAME: str = "muon" # "adamw" or "muon"

def ml_type_encoder(obj):
    if isinstance(obj, torch.dtype):
        return str(obj)  # 输出 "torch.float16"
    elif isinstance(obj, torch.device):
        return str(obj)  # 输出 "cuda"
    # 如果有 Path 对象也可以在这里处理
    # raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    return str(obj) # Fallback: 尝试直接转字符串
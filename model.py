import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from icecream import ic
import logging
from torchvision import models

from config import Config

logger = logging.getLogger(__name__)

class TextEncoder:
    """文本编码器包装类，支持BERT和Sentence-Transformer"""
    def __init__(self, config):
        self.encoder_type = config.ENCODER_TYPE
        self.freeze = config.FREEZE_ENCODER
        
        if self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(config.BERT_MODEL)
            self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL)
            self.embedding_size = config.EMBEDDING_SIZE
            
            # 冻结参数
            if self.freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        elif self.encoder_type == "sentence-transformer":
            self.encoder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
            self.tokenizer = None  # SentenceTransformer内部处理分词
            self.embedding_size = self.encoder.get_sentence_embedding_dimension()
            
            # 冻结参数
            if self.freeze:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        
        else:
            raise ValueError(f"不支持的编码器类型: {self.encoder_type}")
    
    def encode_batch(self, texts, device):
        """
        编码一批文本
        Args:
            texts: 文本列表
            device: 设备
        Returns:
            文本嵌入 [batch_size, embedding_size]
        """
        if self.encoder_type == "bert":
            # 直接使用已有的分词结果进行编码
            return None  # 这里返回None，因为BERT编码在模型内部处理
        
        elif self.encoder_type == "sentence-transformer":
            # 使用sentence-transformer编码
            with torch.no_grad():
                embeddings = self.encoder.encode(texts, convert_to_tensor=True)
                return embeddings.to(device)
    
    def tokenize(self, texts, max_length=128, padding='max_length', truncation=True):
        """
        仅用于BERT模型的分词
        """
        if self.encoder_type == "bert":
            return self.tokenizer(
                texts,
                add_special_tokens=True,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors='pt'
            )
        else:
            return None  # Sentence-Transformer内部处理分词

    def get_embedding_size(self):
        return self.embedding_size
    
class AttentivePoolingModel(nn.Module):
    """使用注意力池化聚合推文表示的模型"""
    def __init__(self, config, num_classes, num_stat_features):
        super(AttentivePoolingModel, self).__init__()
        
        # 设置编码器
        self.encoder_type = config.ENCODER_TYPE
        
        if self.encoder_type == "bert":
            self.bert = BertModel.from_pretrained(config.BERT_MODEL)
            self.embedding_size = config.EMBEDDING_SIZE
            
            # 冻结BERT参数
            if config.FREEZE_ENCODER:
                for param in self.bert.parameters():
                    param.requires_grad = False
        
        elif self.encoder_type == "sentence-transformer":
            self.sentence_encoder = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
            self.embedding_size = self.sentence_encoder.get_sentence_embedding_dimension()
            
            # 冻结Sentence-Transformer参数
            if config.FREEZE_ENCODER:
                for param in self.sentence_encoder.parameters():
                    param.requires_grad = False
        
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # 推文级别的注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # 统计特征处理
        self.stat_fc = nn.Sequential(
            nn.Linear(num_stat_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 最终分类层
        self.classifier = nn.Linear(self.embedding_size + 32, num_classes)
    
    def forward(self, input_ids=None, attention_mask=None, statistical_features=None, sentences=None):
        """
        Args:
            input_ids: BERT词元ID [batch_size, num_tweets, seq_len] 或 None (sentence-transformer)
            attention_mask: BERT注意力掩码 [batch_size, num_tweets, seq_len] 或 None (sentence-transformer)
            statistical_features: 统计特征 [batch_size, num_features]
            sentences: 句子列表 (用于sentence-transformer) 或 None (BERT)
        """
        if self.encoder_type == "bert":
            # 使用BERT处理
            batch_size, num_tweets, seq_len = input_ids.shape
            
            # 分别处理每条推文
            tweet_embeddings = []
            
            for i in range(num_tweets):
                # 提取每个样本中的第i条推文
                tweet_input_ids = input_ids[:, i, :]
                tweet_attention_mask = attention_mask[:, i, :]
                
                # 获取推文的BERT表示
                outputs = self.bert(
                    input_ids=tweet_input_ids,
                    attention_mask=tweet_attention_mask
                )
                
                # 使用[CLS]标记的嵌入作为推文表示
                tweet_embedding = outputs.pooler_output  # [batch_size, embedding_size]
                tweet_embeddings.append(tweet_embedding)
            
            # 堆叠推文嵌入 [batch_size, num_tweets, embedding_size]
            tweet_embeddings = torch.stack(tweet_embeddings, dim=1)
        
        elif self.encoder_type == "sentence-transformer":
            # 使用Sentence-Transformer处理
            batch_size = len(sentences)
            tweet_embeddings = []
            
            # 分别处理每个样本的推文
            for i in range(batch_size):
                sample_sentences = sentences[i]  # 一个样本的所有推文
                
                # 过滤掉空推文
                valid_sentences = [s for s in sample_sentences if s.strip()]
                
                if valid_sentences:
                    # 使用sentence-transformer编码
                    with torch.no_grad():
                        embeddings = self.sentence_encoder.encode(valid_sentences, convert_to_tensor=True)
                        # 填充到预定义长度
                        padded_embeddings = torch.zeros(len(sample_sentences), self.embedding_size, device=embeddings.device)
                        padded_embeddings[:len(embeddings)] = embeddings
                        tweet_embeddings.append(padded_embeddings)
                else:
                    # 如果没有有效的推文，创建零嵌入
                    empty_embeddings = torch.zeros(len(sample_sentences), self.embedding_size, device=statistical_features.device)
                    tweet_embeddings.append(empty_embeddings)
            
            # 堆叠所有样本的推文嵌入 [batch_size, num_tweets, embedding_size]
            tweet_embeddings = torch.stack(tweet_embeddings, dim=0)
        
        # 使用注意力获取话题嵌入
        attention_weights = self.attention(tweet_embeddings)  # [batch_size, num_tweets, 1]
        topic_embedding = torch.sum(attention_weights * tweet_embeddings, dim=1)  # [batch_size, embedding_size]
        
        # 处理统计特征
        stat_embedding = self.stat_fc(statistical_features)
        
        # 组合嵌入
        combined = torch.cat([topic_embedding, stat_embedding], dim=1)
        combined = self.dropout(combined)
        
        # 分类
        logits = self.classifier(combined)
        
        return logits


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        # 这是一个可学习的“评分员”，它看一眼推文，决定这条推文重不重要
        self.attention_fc = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False) # 输出分数
        )
    
    def forward(self, tweet_embeddings, mask):
        # tweet_embeddings: [batch, num_tweets, embed_dim]
        # mask: [batch, num_tweets] (True for valid)
        
        # 1. 计算每条推文的权重分数 (Attention Score)
        # scores: [batch, num_tweets, 1]
        scores = self.attention_fc(tweet_embeddings) 
        
        # 2. Mask掉无效推文 (设为极小值，Softmax后为0)
        # mask is usually True for valid, False for padding
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        
        # 3. 归一化得到权重 alpha
        alpha = torch.softmax(scores, dim=1) 
        
        # 4. 加权求和 (Weighted Sum)
        # [batch, num_tweets, 1] * [batch, num_tweets, embed_dim] -> sum -> [batch, embed_dim]
        context_vector = torch.sum(alpha * tweet_embeddings, dim=1)
        
        return context_vector, alpha # alpha可以拿出来做可视化，看模型关注哪条推文

class TransformerModel(nn.Module):
    """使用Transformer编码器建模推文之间关系的模型"""
    def __init__(self, config: Config, num_classes: int, num_stat_features: int):
        super(TransformerModel, self).__init__()
        
        # 设置编码器
        self.encoder_type = config.ENCODER_TYPE
        self.config: Config = config

        
        if self.encoder_type == "bert":
            self.bert = BertModel.from_pretrained(config.BERT_MODEL)
            # self.embedding_size = config.EMBEDDING_SIZE
            self.embedding_size = self.bert.config.hidden_size
            
            # 冻结BERT参数
            if config.FREEZE_ENCODER:
                for param in self.bert.parameters():
                    param.requires_grad = False
        
        elif self.encoder_type == "sentence-transformer":
            self.sentence_encoder = SentenceTransformer(
                config.SENTENCE_TRANSFORMER_MODEL, 
                trust_remote_code=True,
                model_kwargs={
                    "torch_dtype": self.config.SENTENCE_TRANSFORMER_DTYPE,
                    # "attn_implementation": "flash_attention_2",
                }
            )
            self.embedding_size = self.sentence_encoder.get_sentence_embedding_dimension()
            
            # 冻结Sentence-Transformer参数
            if config.FREEZE_ENCODER:
                for param in self.sentence_encoder.parameters():
                    param.requires_grad = False

        # 添加图像编码器
        self.use_images = config.USE_IMAGES
        if self.use_images:
            # 选择图像编码器 (例如 ResNet, ViT, CLIP等)
            if config.IMAGE_ENCODER_TYPE == "resnet":
                # self.image_encoder = models.resnet50(pretrained=True)
                self.image_encoder = models.resnet50(weights="IMAGENET1K_V2")
                # 移除最后的分类层
                # self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
                self.image_encoder.fc = nn.Identity()
                self.image_embedding_size = 2048  # ResNet50的特征维度
            elif config.IMAGE_ENCODER_TYPE == "vit":
                self.image_encoder = models.vit_b_16(weights="IMAGENET1K_SWAG_LINEAR_V1")
                # 修改为特征提取模式
                self.image_encoder.heads = nn.Identity()
                self.image_embedding_size = 768  # ViT-B/16的特征维度
            
            # 冻结图像编码器参数
            if config.FREEZE_IMAGE_ENCODER:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
            
            # 图像特征投影层 (将图像特征映射到与文本相同的维度)
            self.image_projection = nn.Linear(self.image_embedding_size, self.embedding_size)
            
            # 为纯文本推文添加适配层，使其特征与多模态特征更兼容
            if config.TEXT_ONLY_ADAPT:
                self.text_only_adapter = nn.Sequential(
                    nn.Linear(self.embedding_size, self.embedding_size),
                    nn.GELU(),
                    nn.Dropout(config.DROPOUT_RATE)
                )
            
            # 改进的多模态融合层
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.GELU(),
                nn.Dropout(config.DROPOUT_RATE),
                nn.Linear(self.embedding_size, self.embedding_size),
                nn.GELU(),
                nn.Dropout(config.DROPOUT_RATE)
            )
        
        
        # self.linear_embed_to_encoder = nn.Linear(self.embedding_size, self.config.ENCODER_SIZE)
        self.linear_embed_to_encoder = nn.Sequential(
            nn.LayerNorm(self.embedding_size),
            nn.Linear(self.embedding_size, self.config.ENCODER_SIZE),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
        )
        
        # Transformer编码器层用于推文间注意力
        if self.config.NUM_TRANSFORMER_LAYERS < 1:
            self.transformer_encoder = nn.Identity()
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.config.ENCODER_SIZE,
                nhead=config.NUM_ATTN_HEADS,
                dim_feedforward=self.config.ENCODER_SIZE * self.config.FEEDFORWARD_MULTIPLIER,
                dropout=config.DROPOUT_RATE,
                batch_first=True  # 确保输入格式为 [batch, seq, feature]
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, 
                num_layers=config.NUM_TRANSFORMER_LAYERS,
                enable_nested_tensor=False,
            )

        # add a special embedding for topic sentence
        if self.config.ADD_TOPIC_NAME:
            # self.topic_embedding = nn.Parameter(torch.zeros(1, 1, self.embedding_size))  # [1, 1, embedding_size]
            # nn.init.xavier_uniform_(self.topic_embedding)  # 使用Xavier初始化
            self.topic_fusion = nn.Sequential(
                nn.Linear(self.embedding_size * 2, self.embedding_size),
                nn.GELU(),
                nn.Dropout(config.DROPOUT_RATE),
            )

        # Attentive Pooling for output 
        if self.config.POOLING_STRATEGY == "attentive":
            self.attentive_aggr = AttentionAggregator(self.config.ENCODER_SIZE, 128)
        
        # 统计特征处理
        self.stat_fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_stat_features, 128),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.7),
        )
        
        # 最终分类层
        # self.classifier = nn.Linear(self.embedding_size + 128, num_classes)
        cls_input_size = self.config.ENCODER_SIZE + 128 if self.config.USE_STAT_FEATURES else self.config.ENCODER_SIZE
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(cls_input_size, 256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def drop_tweets(self, tweet_embeddings, tweet_mask, drop_prob):
        """
        随机丢弃部分推文以进行数据增强
        Args:
            tweet_embeddings: [batch_size, num_tweets, embedding_size]
            drop_prob: 丢弃概率
        Returns:
            增强后的推文嵌入
        """
        if drop_prob <= 0.0:
            return tweet_embeddings
        
        batch_size, num_tweets, embedding_size = tweet_embeddings.shape
        keep_mask = (torch.rand(batch_size, num_tweets, device=tweet_embeddings.device) > drop_prob).float()
        
        # 确保至少保留一条推文
        for b in range(batch_size):
            if keep_mask[b].sum() == 0:
                rand_idx = torch.randint(0, num_tweets, (1,))
                keep_mask[b, rand_idx] = 1.0
        
        return tweet_mask * keep_mask
        
    
    def forward(self, input_ids=None, attention_mask=None, statistical_features=None, sentences=None, images=None, image_masks=None, topic_names=None):
        """
        Args:
            input_ids: BERT词元ID [batch_size, num_tweets, seq_len] 或 None (sentence-transformer)
            attention_mask: BERT注意力掩码 [batch_size, num_tweets, seq_len] 或 None (sentence-transformer)
            statistical_features: 统计特征 [batch_size, num_features]
            sentences: 句子列表 (用于sentence-transformer) 或 None (BERT)
            images: 图像张量 [batch_size, num_tweets, channels, height, width] 或 None (无图像)
            image_masks: 图像有效性掩码 [batch_size, num_tweets] - True表示该推文有图像，False表示没有
        """
        device = statistical_features.device
        
        # 处理文本特征 (基于现有代码)
        if self.encoder_type == "bert":
            # 确保输入不为None
            if input_ids is None or attention_mask is None:
                raise ValueError("使用BERT编码器时，input_ids和attention_mask不能为None")
 
            batch_size, num_tweets, seq_len = input_ids.shape
            text_embeddings = torch.zeros(batch_size, num_tweets, self.embedding_size, device=device)
            valid_masks = torch.zeros(batch_size, num_tweets, dtype=torch.bool, device=device)
            
            # 按话题顺序处理
            for b in range(batch_size):
                # 获取当前话题的所有推文
                topic_input_ids = input_ids[b]  # [num_tweets, seq_len]
                topic_attention_mask = attention_mask[b]  # [num_tweets, seq_len]
                
                # 检查哪些推文是有效的（非填充）
                is_valid = topic_attention_mask.sum(dim=1) > 0  # [num_tweets]
                valid_masks[b] = is_valid
                valid_indices = torch.where(is_valid)[0]
                
                # 分批处理有效推文
                for i in range(0, len(valid_indices), self.config.SENTENCE_BATCH_SIZE):
                    # 获取当前批次的索引
                    batch_indices = valid_indices[i:i+self.config.SENTENCE_BATCH_SIZE]
                    
                    # 提取当前批次的输入
                    batch_input_ids = topic_input_ids[batch_indices]
                    batch_attention_mask = topic_attention_mask[batch_indices]
                    
                    # 获取BERT表示
                    with torch.set_grad_enabled(not self.config.FREEZE_ENCODER):
                        outputs = self.bert(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask
                        )
                    
                        # 将结果存回对应位置
                        text_embeddings[b, batch_indices] = outputs.pooler_output

        elif self.encoder_type == "sentence-transformer":
            # 确保输入不为None
            if sentences is None:
                raise ValueError("使用Sentence-Transformer编码器时，sentences不能为None")

            batch_size = len(sentences)
            max_tweets = len(sentences[0]) if sentences else 0
            text_embeddings = torch.zeros(batch_size, max_tweets, self.embedding_size, device=device, dtype=self.config.SENTENCE_TRANSFORMER_DTYPE)
            valid_masks = torch.zeros(batch_size, max_tweets, dtype=torch.bool, device=device)
            
            # 按话题顺序处理
            for b in range(batch_size):
                sample_sentences = sentences[b]  # 当前话题的所有推文
                
                # 创建有效性掩码并获取有效推文索引
                is_valid = [len(s.strip()) > 0 for s in sample_sentences]
                valid_masks[b] = torch.tensor(is_valid, device=device)
                valid_indices = [i for i, v in enumerate(is_valid) if v]
                valid_sentences = [sample_sentences[i] for i in valid_indices]
                # ic(len(valid_sentences))
                
                # 分批处理有效推文
                for i in range(0, len(valid_indices), self.config.SENTENCE_BATCH_SIZE):
                    # 获取当前批次的索引和句子
                    batch_indices = valid_indices[i:i+self.config.SENTENCE_BATCH_SIZE]
                    batch_sentences = [sample_sentences[idx] for idx in batch_indices]
                    
                    # 编码当前批次
                    with torch.set_grad_enabled(not self.config.FREEZE_ENCODER):
                        embeddings = self.sentence_encoder.encode(batch_sentences, convert_to_tensor=True, show_progress_bar=False, batch_size=self.config.SENTENCE_BATCH_SIZE)
                        text_embeddings[b, batch_indices] = embeddings.to(device)

            if self.config.ADD_TOPIC_NAME and topic_names is not None:
                # # the first sentence embedding is the already topic name
                # # add a special embedding for topic sentence, placed at first
                # topic_embedding = self.topic_embedding.expand(batch_size, -1, -1).to(device)  # [batch_size, 1, embedding_size]
                # text_embeddings[..., 0, :] += topic_embedding.squeeze(1)  # 将话题嵌入添加到每个样本的第一个推文
                # we just encode twice?
                # encode topic name and add to the first position
                with torch.set_grad_enabled(not self.config.FREEZE_ENCODER):
                    topic_embedding = self.sentence_encoder.encode(topic_names, convert_to_tensor=True, show_progress_bar=False, batch_size=self.config.SENTENCE_BATCH_SIZE)
                    # -> [batch_size, embedding_size]
                # prepend to each sample's first tweet embedding
                # text_embeddings = torch.cat([topic_embedding.unsqueeze(1).to(device), text_embeddings[:, :-1, :]], dim=1)
                # valid_masks = torch.cat([torch.ones(batch_size, 1, dtype=torch.bool, device=device), valid_masks[:, :-1]], dim=1)
                # concat to ALL tweet embeddings
                
                # torch.cat不支持广播机制，必须显式扩展维度以匹配num_tweets
                topic_embedding_expanded = topic_embedding.unsqueeze(1).expand(-1, text_embeddings.size(1), -1).to(device)
                text_embeddings = torch.cat([topic_embedding_expanded, text_embeddings], dim=-1) # [batch_size, num_tweets, 2 * embedding_size]
                text_embeddings = self.topic_fusion(text_embeddings)

        # cast to this model's dtype
        text_embeddings = text_embeddings.to(dtype=self.linear_embed_to_encoder[0].weight.dtype)
        
        # 处理图像特征
        if self.use_images and images is not None:
            batch_size, num_tweets = valid_masks.shape
            
            # 如果没有提供image_masks，则创建一个默认掩码
            if image_masks is None:
                # 默认假设所有图像都是有效的，但在处理时会检查
                image_masks = torch.ones(batch_size, num_tweets, dtype=torch.bool, device=device)
            
            # 初始化图像嵌入
            image_embeddings = torch.zeros(batch_size, num_tweets, self.embedding_size, device=device)
            
            # 按话题顺序处理
            for b in range(batch_size):
                # 获取当前话题的有效推文和有图像的推文
                text_valid = valid_masks[b]  # 文本有效的推文
                image_valid = image_masks[b]  # 有图像的推文
                
                # 只处理同时有文本和图像的推文
                valid_indices = torch.where(text_valid & image_valid)[0]
                
                if len(valid_indices) > 0:
                    # 分批处理有效图像
                    for i in range(0, len(valid_indices), self.config.IMAGE_BATCH_SIZE):
                        # 获取当前批次的索引
                        batch_indices = valid_indices[i:i+self.config.IMAGE_BATCH_SIZE]
                        
                        # 提取当前批次的图像
                        batch_images = images[b, batch_indices]
                        
                        # 编码图像
                        with torch.set_grad_enabled(not self.config.FREEZE_IMAGE_ENCODER):
                            img_features = self.image_encoder(batch_images)
                            if len(img_features.shape) > 2:  # 如果输出是特征图
                                img_features = img_features.squeeze(-1).squeeze(-1)  # 移除空间维度
                            
                            # 投影到与文本相同的维度
                            img_features = self.image_projection(img_features)
                            
                            # 将结果存回对应位置
                            image_embeddings[b, batch_indices] = img_features
            
            # 融合文本和图像特征
            tweet_embeddings = text_embeddings.clone()
            
            for b in range(batch_size):
                # 获取同时有文本和图像的推文索引
                multimodal_indices = torch.where(valid_masks[b] & image_masks[b])[0]
                
                if len(multimodal_indices) > 0:
                    # 获取文本和图像特征
                    text_feats = text_embeddings[b, multimodal_indices]
                    img_feats = image_embeddings[b, multimodal_indices]
                    
                    # 连接文本和图像特征
                    multimodal_feats = torch.cat([text_feats, img_feats], dim=1)
                    
                    # 融合特征
                    fused_feats = self.multimodal_fusion(multimodal_feats)
                    
                    # 更新特征 - 只更新有图像的推文
                    tweet_embeddings[b, multimodal_indices] = fused_feats
                
                # 对于只有文本的推文，保持原始文本特征
                text_only_indices = torch.where(valid_masks[b] & ~image_masks[b])[0]
                if len(text_only_indices) > 0:
                    # 可以选择对纯文本特征进行额外处理，或者直接使用
                    # 这里我们使用一个可选的文本适配层，使纯文本特征与多模态特征更兼容
                    if hasattr(self, 'text_only_adapter'):
                        tweet_embeddings[b, text_only_indices] = self.text_only_adapter(
                            text_embeddings[b, text_only_indices]
                        )

        else:
            # 如果不使用图像，直接使用文本特征
            tweet_embeddings = text_embeddings
        
        # 创建Transformer的掩码（无效的推文被掩码）
        transformer_mask = ~valid_masks  # [batch_size, num_tweets]

        # 将嵌入映射到Transformer的输入维度
        tweet_embeddings = self.linear_embed_to_encoder(tweet_embeddings)  # [batch_size, num_tweets, encoder_size]

        if self.training and self.config.DROP_TWEET_RATIO > 0.0:
            # 在训练时随机丢弃部分推文
            # tweet_embeddings = self.drop_tweets(tweet_embeddings, self.config.DROP_TWEET_RATIO)
            transformer_mask = self.drop_tweets(tweet_embeddings, transformer_mask, self.config.DROP_TWEET_RATIO)
        
        # 应用Transformer编码器进行推文间注意力
        if isinstance(self.transformer_encoder, nn.Identity):
            transformed_embeddings = tweet_embeddings
        else:
            transformed_embeddings = self.transformer_encoder(
                tweet_embeddings,
                src_key_padding_mask=transformer_mask
            )
        
        # 平均池化获取话题嵌入
        valid_tweets_mask = valid_masks.float().unsqueeze(-1)  # [batch_size, num_tweets, 1]

        if self.config.POOLING_STRATEGY == "mean":
            topic_embedding = (transformed_embeddings * valid_tweets_mask).sum(dim=1) / (valid_tweets_mask.sum(dim=1) + 1e-10)
        elif self.config.POOLING_STRATEGY == "max":
            masked_embeddings = transformed_embeddings.masked_fill(~valid_tweets_mask.bool(), float('-inf'))
            topic_embedding, _ = masked_embeddings.max(dim=1)
        elif self.config.POOLING_STRATEGY == "cls":
            topic_embedding = transformed_embeddings[:, 0, :]  # 假设第一个位置是CLS标记
        elif self.config.POOLING_STRATEGY == "attentive":
            # 使用注意力机制进行池化
            topic_embedding, attn_weights = self.attentive_aggr(transformed_embeddings, valid_masks)
        else:
            raise ValueError(f"未知的池化策略: {self.config.POOLING_STRATEGY}")

        # 处理统计特征
        if self.config.USE_STAT_FEATURES:
            stat_embedding = self.stat_fc(statistical_features)
        
            # 组合嵌入
            combined = torch.cat([topic_embedding, stat_embedding], dim=1)
        else:
            combined = topic_embedding
        
        # 分类
        logits = self.classifier(combined)
        
        return logits

def get_model(config, num_classes, num_stat_features):
    """根据配置创建模型"""
    if config.MODEL_TYPE == "attentive_pooling":
        return AttentivePoolingModel(config, num_classes, num_stat_features)
    elif config.MODEL_TYPE == "transformer":
        return TransformerModel(config, num_classes, num_stat_features)
    else:
        raise ValueError(f"未知的模型类型: {config.MODEL_TYPE}")

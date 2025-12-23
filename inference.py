import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import pandas as pd
from tqdm import tqdm
import logging
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
from PIL import Image

from config import Config
from model import get_model
from data import preprocess_text, extract_topic_features, get_image_transforms

logger = logging.getLogger(__name__)

class TopicClassifier:
    def __init__(self, model_path='./topic_model', config_path=None, device=None, batch_size=None, **kwargs):
        """
        初始化话题分类器
        
        Args:
            model_path: 模型权重文件路径，默认为None
            config_path: 配置文件路径，如果为None则尝试从model_path所在目录加载
            device: 运行设备，如果为None则自动选择
            batch_size: 批处理大小，如果为None则使用配置文件中的值
            **kwargs: 其他推理参数
        """
        # 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 加载配置
        if config_path is None and model_path:
            config_path = os.path.join(os.path.dirname(model_path), "config.json")
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
                self.config = Config()
                # 更新配置
                for key, value in config_dict.items():
                    setattr(self.config, key, value)
            logger.info(f"加载配置: {config_path}")
        else:
            # 使用默认配置
            self.config = Config()
            logger.info("使用默认配置")
        
        # 设置批处理大小
        if batch_size is not None:
            self.config.BATCH_SIZE = batch_size
        
        # 更新其他kwargs参数
        for key, value in kwargs.items():
            if hasattr(self.config, key.upper()):
                setattr(self.config, key.upper(), value)
        
        # 初始化分词器和编码器
        if self.config.ENCODER_TYPE == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL)
            logger.info(f"使用BERT分词器: {self.config.BERT_MODEL}")
        else:  # sentence-transformer
            self.tokenizer = None
            self.sentence_encoder = SentenceTransformer(self.config.SENTENCE_TRANSFORMER_MODEL, trust_remote_code=True)
            logger.info(f"使用Sentence-Transformer: {self.config.SENTENCE_TRANSFORMER_MODEL}")
        
        # 初始化图像转换
        self.use_images = getattr(self.config, 'USE_IMAGES', False)
        if self.use_images:
            self.image_transforms = get_image_transforms()
            logger.info("启用图像处理")
        
        # 加载类别映射
        self.id2label = {0: "轻微", 1: "中等", 2: "严重"}
        self.num_classes = len(self.id2label)
        
        # 从配置文件或模型元数据中获取统计特征数量
        self.num_stat_features = getattr(self.config, 'NUM_STAT_FEATURES', None)
        
        # 如果配置中没有指定特征数量，尝试从checkpoint中推断
        if self.num_stat_features is None:
            # 尝试从模型权重文件中推断特征数量
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                for name, param in checkpoint.items():
                    print(name)
                # 查找统计特征层的权重名称
                for name, param in checkpoint.items():
                    if 'stat_fc.0.weight' in name:  # 第一个统计特征层的权重
                        self.num_stat_features = param.shape[1]
                        logger.info(f"从模型权重推断统计特征数量: {self.num_stat_features}")
                        break
            except Exception as e:
                logger.warning(f"无法从模型权重推断特征数量: {e}")
                
        # 如果仍然无法确定，使用默认值并发出警告
        if self.num_stat_features is None:
            logger.warning("无法确定统计特征数量，使用默认值10。这可能导致模型加载失败！")
            self.num_stat_features = 10
        
        # 创建模型
        self.model = get_model(self.config, self.num_classes, self.num_stat_features)
        
        # 加载权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"模型权重加载成功: {model_path}")
        except Exception as e:
            logger.error(f"模型权重加载失败: {e}")
            raise RuntimeError(f"模型权重加载失败，可能是统计特征数量不匹配。错误: {e}")
            
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载完成: {model_path}")
    
    def forward(self, event_name, csv_file_path, image_dir_path):
        """
        执行推理，预测事件的预警级别
        
        Args:
            event_name: 事件名称
            csv_file_path: 事件数据CSV文件路径
            image_dir_path: 事件数据图片文件夹路径
        
        Returns:
            tuple: (事件级结果json地址, 帖子级结果csv地址)
        """
        # 加载并处理话题数据
        topic_data = self.load_and_process_topic_from_csv(event_name, csv_file_path, image_dir_path)
        if topic_data is None:
            logger.error("加载话题数据失败")
            return None, None
        
        # 准备模型输入
        inputs = self.prepare_inputs(topic_data)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                statistical_features=inputs['statistical_features'],
                sentences=inputs['sentences'],
                images=inputs['images'],
                image_masks=inputs['image_masks']
            )
            
            # 获取预测结果
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            # 构建事件级结果
            event_result = {
                "event_name": event_name,
                "predicted_class": predicted_class,
                "predicted_label": self.id2label[predicted_class],
                "probabilities": {
                    self.id2label[i]: float(prob) 
                    for i, prob in enumerate(probabilities[0].cpu().numpy())
                }
            }
            
            # 保存事件级结果
            event_json_path = f"{event_name}_event_result.json"
            with open(event_json_path, 'w', encoding='utf-8') as f:
                json.dump(event_result, f, ensure_ascii=False, indent=4)
            
            # 构建帖子级结果（如果需要的话）
            post_results = []
            for i, text in enumerate(topic_data['texts']):
                if text.strip():  # 只处理非空文本
                    post_results.append({
                        "id": topic_data['ids'][i] if topic_data['ids'][i] else i,
                        "text": text,
                        "predicted_class": predicted_class,
                        "predicted_label": self.id2label[predicted_class]
                    })
            
            # 保存帖子级结果
            post_csv_path = f"{event_name}_post_results.csv"
            if post_results:
                pd.DataFrame(post_results).to_csv(post_csv_path, index=False, encoding='utf-8-sig')
            else:
                # 创建空的CSV文件
                pd.DataFrame(columns=["id", "text", "predicted_class", "predicted_label"]).to_csv(
                    post_csv_path, index=False, encoding='utf-8-sig'
                )
            
            logger.info(f"预测完成: {event_name}")
            logger.info(f"事件级结果保存到: {event_json_path}")
            logger.info(f"帖子级结果保存到: {post_csv_path}")
            
            return event_json_path, post_csv_path
    
    def load_and_process_topic_from_csv(self, event_name, csv_file_path, image_dir_path):
        """
        从CSV文件和图片目录加载并处理话题数据
        
        Args:
            event_name: 事件名称
            csv_file_path: 事件数据CSV文件路径
            image_dir_path: 事件数据图片文件夹路径
        
        Returns:
            处理后的话题数据字典
        """
        # 加载话题数据
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            logger.info(f"加载话题数据: {csv_file_path}, 微博数量: {len(df)}")
        except Exception as e:
            logger.error(f"加载话题数据失败: {e}")
            return None
        
        # 提取统计特征
        stat_features = extract_topic_features(df)
        
        # 预处理微博文本
        df['processed_text'] = df["微博正文"].apply(preprocess_text)
        
        # 获取预处理后的文本
        texts = df['processed_text'].dropna().tolist()
        
        # 获取微博ID (如果存在)
        if 'id' in df.columns:
            ids = df['id'].astype(str).tolist()
        else:
            ids = [None] * len(texts)
        
        # 限制微博数量
        max_tweets = self.config.MAX_TWEETS
        texts = texts[:max_tweets]
        ids = ids[:max_tweets]
        
        # 如果不够则用空字符串和None填充
        if len(texts) < max_tweets:
            texts.extend([""] * (max_tweets - len(texts)))
            ids.extend([None] * (max_tweets - len(ids)))
        
        # 准备返回数据
        topic_data = {
            'topic': event_name,
            'texts': texts,
            'ids': ids,
            'stat_features': stat_features,
            'image_dir': image_dir_path
        }
        
        return topic_data
    
    def load_tweet_images(self, topic_data):
        """
        加载话题中的推文图像
        
        Args:
            topic_data: 话题数据字典
        
        Returns:
            图像张量和掩码
        """
        if not self.use_images:
            return None, None
        
        images = []
        image_masks = []
        
        # 图像目录路径
        image_dir = topic_data['image_dir']
        tweet_ids = topic_data['ids']
        
        for tweet_id in tweet_ids:
            if tweet_id and os.path.exists(image_dir):
                # 尝试不同的可能文件名格式
                possible_paths = [
                    os.path.join(image_dir, f"{tweet_id}.jpg"),
                    os.path.join(image_dir, f"{tweet_id}.png"),
                    os.path.join(image_dir, f"{tweet_id}.jpeg")
                ]
                
                image_found = False
                for img_path in possible_paths:
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert('RGB')
                            # 预处理图像
                            img_tensor = self.image_transforms(img)
                            images.append(img_tensor)
                            image_masks.append(True)
                            image_found = True
                            break
                        except Exception as e:
                            logger.warning(f"加载图像失败: {img_path}, 错误: {e}")
                
                if not image_found:
                    # 如果没有找到图像，添加零张量
                    images.append(torch.zeros(3, 224, 224))
                    image_masks.append(False)
            else:
                # 如果tweet_id为None或图像目录不存在，添加零张量
                images.append(torch.zeros(3, 224, 224))
                image_masks.append(False)
        
        return torch.stack(images), torch.tensor(image_masks, dtype=torch.bool)
    
    def prepare_inputs(self, topic_data):
        """
        准备模型输入
        
        Args:
            topic_data: 话题数据字典
        
        Returns:
            模型输入字典
        """
        texts = topic_data['texts']
        stat_features = topic_data['stat_features']
        
        # 统计特征转为张量
        stat_features_list = list(stat_features.values())
        stat_features_tensor = torch.tensor(stat_features_list, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # 加载图像
        if self.use_images:
            images, image_masks = self.load_tweet_images(topic_data)
            images = images.unsqueeze(0).to(self.device)
            image_masks = image_masks.unsqueeze(0).to(self.device)
        else:
            images = None
            image_masks = None
        
        # 分词
        if self.config.ENCODER_TYPE == "bert":
            encodings = []
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.config.MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                encodings.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0)
                })
            
            # 堆叠所有编码
            input_ids = torch.stack([enc['input_ids'] for enc in encodings]).unsqueeze(0).to(self.device)
            attention_mask = torch.stack([enc['attention_mask'] for enc in encodings]).unsqueeze(0).to(self.device)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'statistical_features': stat_features_tensor,
                'sentences': None,
                'images': images,
                'image_masks': image_masks
            }
        
        elif self.config.ENCODER_TYPE == "sentence-transformer":
            sentences = [s[:self.config.MAX_LEN] for s in texts]
            
            if self.config.ADD_TOPIC_NAME:
                # add the topic as first sentence
                topic_sentence = f"Topic: {topic_data['topic']}"
                sentences = [topic_sentence] + sentences  # 将话题作为第一句

            # 对于sentence-transformer，直接返回原始句子
            return {
                'input_ids': None,
                'attention_mask': None,
                'statistical_features': stat_features_tensor,
                'sentences': [sentences],  # 注意这里需要是二维列表 [batch_size, num_tweets]
                'images': images,
                'image_masks': image_masks
            }
    
    def predict(self, topic_path):
        """
        预测话题的预警级别
        
        Args:
            topic_path: 话题CSV文件路径
        
        Returns:
            预测结果字典
        """
        # 为了保持兼容性，从topic_path提取事件名和图片目录
        event_name = os.path.basename(topic_path).split('.')[0]
        image_dir_path = os.path.join(os.path.dirname(topic_path), 'images')
        
        # 调用forward方法
        event_json_path, post_csv_path = self.forward(event_name, topic_path, image_dir_path)
        
        # 返回事件结果
        if event_json_path and os.path.exists(event_json_path):
            with open(event_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"error": "预测失败"}
    
    def predict_batch(self, topic_paths):
        """
        批量预测多个话题的预警级别
        
        Args:
            topic_paths: 话题CSV文件路径列表
        
        Returns:
            预测结果列表
        """
        results = []
        for path in tqdm(topic_paths, desc="预测话题"):
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                logger.error(f"预测话题 {path} 时出错: {e}")
                results.append({"topic": os.path.basename(path).split('.')[0], "error": str(e)})
        
        return results

if __name__ == "__main__":
    # 简单的运行测试
    inference_model = TopicClassifier(
        model_path="model/transformer_Alibaba-NLP/gte-multilingual-base_FROZEN/2025-04-07_20-52-48/transformer_best.pt", 
        batch_size=4
    )

    event_name = "2岁娃长隆酒店高烧保安拒交外卖药"
    event_json, post_csv = inference_model.forward(
        event_name=event_name,
        csv_file_path="data/2岁娃长隆酒店高烧保安拒交外卖药/2岁娃长隆酒店高烧保安拒交外卖药.csv",
        image_dir_path="data/4名中国演员侥幸逃脱泰国试戏骗局/images"
    )

    print("事件级结果:", event_json)
    print("帖子级结果:", post_csv)

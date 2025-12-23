import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import random
from harvesttext import HarvestText
import emoji
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import jieba
import shutil
import csv
import logging
from PIL import Image
from torchvision import transforms
from pathlib import Path

from config import Config

ht = HarvestText()
logger = logging.getLogger(__name__)

def set_cjk_font():
    # 查找系统中支持中文的字体
    chinese_fonts = []
    for f in fm.fontManager.ttflist:
        if 'chinese' in f.name.lower() or 'cjk' in f.name.lower():
            chinese_fonts.append(f.name)

    # 如果找到了支持中文的字体，使用第一个
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = [chinese_fonts[0]] + plt.rcParams['font.sans-serif']
        logger.info(f"找到支持中文的字体：{chinese_fonts[0]}")
    else:
        for font in ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']:
            try:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
            except:
                continue

    # 设置matplotlib支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

set_cjk_font()

def get_image_transforms():
    """获取图像预处理转换"""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),  # 缩放到224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet标准化参数
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_image(image_path):
    """加载并预处理图像"""
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.info(f"加载图像时出错: {e}, 路径: {image_path}")
        return None

def preprocess_text(text):
    """预处理微博文本"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 移除微博特有的文本
    text = re.sub(r'O网页链接', '', text)

    # L"xxxx"的微博视频 -> remove
    text = re.sub(r'L(.+)的微博视频', '', text)

    # 处理话题：保留每个话题的第一次出现，移除重复出现的相同话题
    topics = set()
    def replace_topic(match):
        topic = match.group(1)
        if topic in topics:
            return ""  # 移除重复话题
        topics.add(topic)
        return topic + ' '  # 保留话题内容但移除#符号
    
    text = re.sub(r'#([^#]+)#', replace_topic, text)
    
    text = ht.clean_text(text, t2s=False, emoji=False)

    # replace emoji
    text = emoji.demojize(text, language='zh')
    
    # 移除转发标记和无用符号
    text = re.sub(r'转发微博|回复@.*?:', '', text)
    
    # 全角标点转半角标点
    punctuation_map = {
        '，': ',', '。': '.', '！': '!', '？': '?', '；': ';', '：': ':', 
        '"': '"', '"': '"', ''': "'", ''': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '、': ',', '《': '<', '》': '>'
    }

    for full, half in punctuation_map.items():
        text = text.replace(full, half)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_topic_features(topic_df):
    """提取话题统计特征"""
    features = {}
    
    # 话题规模特征
    features['post_count'] = len(topic_df)
    
    # 时间特征
    if '发布时间' in topic_df.columns:
        # 预处理时间字符串
        def normalize_time(t):
            if pd.isna(t):
                return pd.NaT
            t = str(t).strip()
            
            # 尝试提取标准格式 YYYY-MM-DD HH:MM:SS 或 YYYY/MM/DD HH:MM (支持 - 和 /)
            match_full = re.search(r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}(:\d{1,2})?)', t)
            if match_full:
                return match_full.group(1).replace('/', '-')

            # 尝试提取无年份格式 MM-DD HH:MM:SS 或 MM/DD HH:MM (支持 - 和 /)
            # 例如 "05-17 08:13:00" 或 "05-18 21:25" (即使后面跟着乱码)
            match_no_year = re.search(r'(\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{1,2}(:\d{1,2})?)', t)
            if match_no_year:
                # 默认补全为当前年份
                date_str = match_no_year.group(1).replace('/', '-')
                return f"{pd.Timestamp.now().year}-{date_str}"
            
            return pd.NaT

        # 先应用预处理
        time_series = topic_df['发布时间'].apply(normalize_time)
        
        # 再转换为datetime对象，无法解析的设为NaT
        topic_df['发布时间'] = pd.to_datetime(time_series, errors='coerce', format='mixed')
        valid_times = topic_df['发布时间'].dropna()
        if not valid_times.empty:
            features['timespan_hours'] = (valid_times.max() - valid_times.min()).total_seconds() / 3600
            
            # 发文时间分布
            hour_counts = valid_times.dt.hour.value_counts().sort_index()
            features['peak_hour'] = hour_counts.idxmax() if not hour_counts.empty else 0
            night_hours = list(range(0, 7)) + list(range(22, 24))
            night_count = hour_counts[hour_counts.index.isin(night_hours)].sum() if not hour_counts.empty else 0
            features['night_ratio'] = night_count / hour_counts.sum() if hour_counts.sum() > 0 else 0
        else:
            logger.critical(f"无法解析发布时间: {topic_df['发布时间']}")
            features['timespan_hours'] = 0
            features['peak_hour'] = 0
            features['night_ratio'] = 0
    else:
        features['timespan_hours'] = 0
        features['peak_hour'] = 0
        features['night_ratio'] = 0
    
    # 互动特征
    for col in ['转发数', '评论数', '点赞数']:
        if col in topic_df.columns:
            topic_df[col] = pd.to_numeric(topic_df[col], errors='coerce').fillna(0)
    
    if all(col in topic_df.columns for col in ['转发数', '评论数', '点赞数']):
        features['avg_reposts'] = topic_df['转发数'].mean()
        features['avg_comments'] = topic_df['评论数'].mean()
        features['avg_likes'] = topic_df['点赞数'].mean()
        features['engagement_ratio'] = (topic_df['转发数'].sum() + topic_df['评论数'].sum()) / (topic_df['点赞数'].sum() + 1)
    else:
        features['avg_reposts'] = 0
        features['avg_comments'] = 0
        features['avg_likes'] = 0
        features['engagement_ratio'] = 0
    
    # 用户特征
    if '用户昵称' in topic_df.columns:
        features['unique_users'] = topic_df['用户昵称'].nunique()
        features['user_post_ratio'] = features['unique_users'] / features['post_count'] if features['post_count'] > 0 else 0
    else:
        features['unique_users'] = 0
        features['user_post_ratio'] = 0
    
    # 地域分布
    if '发布位置' in topic_df.columns:
        location_counts = topic_df['发布位置'].fillna('未知').value_counts()
        features['location_diversity'] = len(location_counts) / features['post_count'] if features['post_count'] > 0 else 0
        features['top_location'] = location_counts.index[0] if not location_counts.empty else "未知"
        features['top_location_ratio'] = location_counts.iloc[0] / features['post_count'] if not location_counts.empty and features['post_count'] > 0 else 0
    else:
        features['location_diversity'] = 0
        features['top_location'] = "未知"
        features['top_location_ratio'] = 0
    
    # 保留数值特征
    # 修复：numpy的数值类型(如np.int64)可能不被视为int或float，需添加np.number
    numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float, np.number)) and k != 'top_location'}
    
    return numeric_features

class TopicDataset(Dataset):
    def __init__(self, topic_list, data_dir, emergency_levels, tokenizer, max_len, max_tweets, encoder_type, use_images=False, annotation_dir=None):
        """创建话题数据集"""
        self.topics = []
        self.tweets = []
        self.sentences = []
        self.labels = []
        self.statistical_features = []
        self.tweet_ids = []  # 存储每个推文的ID，用于查找图像
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_tweets = max_tweets
        self.encoder_type = encoder_type
        
        # 处理 data_dir，支持单个路径或路径列表
        if isinstance(data_dir, str):
            self.data_dirs = [data_dir]
        elif isinstance(data_dir, list):
            self.data_dirs = data_dir
        else:
            raise ValueError("data_dir must be a string or a list of strings")
            
        # 添加默认的新数据目录（为了兼容性）
        default_new_dir = r'/data_new/cleaned_data'
        if default_new_dir not in self.data_dirs:
            self.data_dirs.append(default_new_dir)
            
        self.image_transforms = get_image_transforms()
        self.use_images = use_images
        
        # 设置缓存目录
        self.cache_dir = os.path.join(annotation_dir, 'cleaned_data_deduplicated') if annotation_dir else None
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # 使用缓存或预处理数据
        self.load_data(topic_list, self.data_dirs, emergency_levels)

        # show count
        logger.critical(f"数据集初始化完成，包含 {len(self.topics)} 个话题")

        
    def load_data(self, topic_list, data_dirs, emergency_levels):
        """加载数据，如有缓存则使用缓存"""
        num_valid_tweets = 0
        num_cached_topics = 0
        
        for topic in topic_list:
            cached_path = None
            if self.cache_dir:
                cached_path = os.path.join(self.cache_dir, f"{topic}.csv")
            
            # 检查是否存在缓存
            if cached_path and os.path.exists(cached_path):
                logger.info(f"使用缓存数据加载话题: {topic}")
                # 从缓存读取
                try:
                    df = pd.read_csv(cached_path, encoding='utf-8')
                    num_cached_topics += 1
                    
                    # 提取统计特征
                    stat_features = extract_topic_features(df)
                    
                    # 直接从预处理后的数据中获取微博文本
                    texts = df["processed_text"].dropna().tolist()
                    
                    # 获取微博ID (如果存在)
                    if 'id' in df.columns:
                        ids = df['id'].astype(str).tolist()
                    else:
                        ids = [None] * len(texts)
                    
                    # 限制微博数量
                    texts = texts[:self.max_tweets]
                    ids = ids[:self.max_tweets]
                    
                    valid_tweets = len(texts)
                    num_valid_tweets += valid_tweets
                    
                    # 如果不够则用空字符串和None填充
                    if len(texts) < self.max_tweets:
                        texts.extend([""] * (self.max_tweets - len(texts)))
                        ids.extend([None] * (self.max_tweets - len(ids)))

                    # 处理缺失的预警等级，默认为0
                    if topic not in emergency_levels:
                        logger.warning(f"话题 {topic} 不在预警级别文件中，使用默认级别 0")
                        emergency_levels[topic] = 0
                    
                    self.topics.append(topic)
                    self.tweets.append(texts)
                    self.tweet_ids.append(ids)
                    self.labels.append(emergency_levels.get(topic, 0))
                    self.statistical_features.append(stat_features)
                    self.sentences.append(texts)
                    
                except Exception as e:
                    logger.info(f"加载缓存话题 {topic} 时出错: {e}")
            else:
                # 从原始数据读取并预处理
                # 遍历所有数据目录查找话题文件
                topic_path = None
                for d in data_dirs:
                    temp_path = os.path.join(d, topic, f"{topic}.csv")
                    if os.path.exists(temp_path):
                        topic_path = temp_path
                        break

                if topic_path and os.path.exists(topic_path):
                    try:
                        df = pd.read_csv(topic_path, encoding='utf-8')
                        
                        # 提取统计特征
                        stat_features = extract_topic_features(df)
                        
                        # 预处理微博文本并保存处理后的结果
                        df['processed_text'] = df["微博正文"].apply(preprocess_text)
                        
                        # 保存预处理后的数据到缓存
                        if cached_path:
                            df.to_csv(cached_path, index=False, encoding='utf-8')
                        
                        # 获取预处理后的文本, remove nan
                        texts = df['processed_text'].dropna().tolist()
                        
                        # 获取微博ID (如果存在)
                        if 'id' in df.columns:
                            ids = df['id'].astype(str).tolist()
                        else:
                            ids = [None] * len(texts)
                        
                        # 限制微博数量
                        texts = texts[:self.max_tweets]
                        ids = ids[:self.max_tweets]
                        
                        valid_tweets = len(texts)
                        num_valid_tweets += valid_tweets
                        
                        # 如果不够则用空字符串和None填充
                        if len(texts) < self.max_tweets:
                            texts.extend([""] * (self.max_tweets - len(texts)))
                            ids.extend([None] * (self.max_tweets - len(ids)))
                        
                        # 处理缺失的预警等级，默认为0
                        if topic not in emergency_levels:
                            logger.warning(f"话题 {topic} 不在预警级别文件中，使用默认级别 0")
                            emergency_levels[topic] = 0

                        self.topics.append(topic)
                        self.tweets.append(texts)
                        self.tweet_ids.append(ids)
                        self.labels.append(emergency_levels.get(topic, 0))
                        self.statistical_features.append(stat_features)
                        self.sentences.append(texts)
                        
                    except Exception as e:
                        logger.critical(f"加载话题 {topic} 时出错: {e}")
                        raise e
                else:
                    logger.critical(f"未找到话题文件: {topic}")
                    raise FileNotFoundError(f"未找到话题文件: {topic}")
                
        # check inconsistency in statistical features
        for features in self.statistical_features:
            if len(features) != 12:  # 应该有10个数值特征
                logger.critical(f"话题统计特征数量不一致: {features}")
                # find what is missing or extra
                correct_sample = None
                for f in self.statistical_features:
                    if len(f) == 12:
                        correct_sample = f
                        break

                logger.critical(f"正确的特征样本: {correct_sample}")
                if correct_sample:
                    for key in correct_sample.keys():
                        if key not in features:
                            logger.critical(f"缺失的特征: {key}")
                    for key in features.keys():
                        if key not in correct_sample:
                            logger.critical(f"多余的特征: {key}")
                raise ValueError(f"话题统计特征数量不一致: {features}")
        
        # 显示加载统计信息
        logger.info(f"话题数量: {len(self.topics)}, 总微博数量: {sum(len(t) for t in self.tweets)}, 有效微博数量: {num_valid_tweets}")
        if self.cache_dir:
            logger.info(f"从缓存加载的话题数量: {num_cached_topics}, 新处理的话题数量: {len(self.topics) - num_cached_topics}")
    
    def load_tweet_images(self, topic, tweet_ids):
        """加载指定话题下的推文图像"""
        images = []
        image_masks = []
        
        # 图像目录路径：遍历所有数据目录查找
        image_dir = None
        for d in self.data_dirs:
            temp_dir = os.path.join(d, topic, 'images')
            if os.path.exists(temp_dir):
                image_dir = temp_dir
                break
        
        for tweet_id in tweet_ids:
            if tweet_id and image_dir and os.path.exists(image_dir):
                # 尝试不同的可能文件名格式
                possible_paths = [
                    os.path.join(image_dir, f"{tweet_id}.jpg"),
                    os.path.join(image_dir, f"{tweet_id}.png"),
                    os.path.join(image_dir, f"{tweet_id}.jpeg")
                ]
                
                image_found = False
                for img_path in possible_paths:
                    if os.path.exists(img_path):
                        img = load_image(img_path)
                        if img is not None:
                            # 预处理图像
                            img_tensor = self.image_transforms(img)
                            images.append(img_tensor)
                            image_masks.append(True)
                            image_found = True
                            break
                
                if not image_found:
                    # 如果没有找到图像，添加零张量
                    images.append(torch.zeros(3, 224, 224))
                    image_masks.append(False)
            else:
                # 如果tweet_id为None或图像目录不存在，添加零张量
                images.append(torch.zeros(3, 224, 224))
                image_masks.append(False)
        
        return torch.stack(images), torch.tensor(image_masks, dtype=torch.bool)
    


    def plot_word_length_distribution(self, save_path='word_length_distribution.png', dataset_name='Dataset'):
        """
        绘制并保存所有微博文本的词长分布图, 使用jieba分词
        
        Args:
            save_path (str): 图像保存路径
            dataset_name (str): 数据集名称(用于显示)
        """
        # 收集所有非空微博
        all_tweets = [tweet for topic_tweets in self.tweets for tweet in topic_tweets if tweet]
        
        if not all_tweets:
            logger.info(f"{dataset_name}中没有可分析的微博")
            return
        
        # 使用jieba进行中文分词并计算词长
        word_lengths = []
        char_lengths = []
        
        for tweet in all_tweets:
            try:
                # 确保tweet是字符串类型
                if not isinstance(tweet, str):
                    logger.info(f"微博内容不是字符串类型: {tweet}, 类型: {type(tweet)}")
                    tweet = str(tweet)
                
                # 分词并计算长度
                words = list(jieba.cut(tweet))
                word_lengths.append(len(words))
                char_lengths.append(len(tweet))
            except Exception as e:
                logger.info(f"处理微博时出错: {e}, 微博内容: {tweet}, 类型: {type(tweet)}")
                continue

        if not word_lengths:
            logger.info(f"{dataset_name}中没有有效的微博可以分析")
            return
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 绘制词长分布（jieba分词）
        ax1.hist(word_lengths, bins=50, alpha=0.7, color='blue')
        ax1.set_xlabel('词数 (jieba分词)')
        ax1.set_ylabel('频率')
        ax1.set_title('微博词长分布')
        
        # 添加统计信息
        word_mean = np.mean(word_lengths)
        word_median = np.median(word_lengths)
        ax1.axvline(word_mean, color='red', linestyle='--', label=f'均值: {word_mean:.2f}')
        ax1.axvline(word_median, color='green', linestyle='--', label=f'中位数: {word_median:.2f}')
        ax1.legend()
        
        # 绘制字符长度分布
        ax2.hist(char_lengths, bins=50, alpha=0.7, color='orange')
        ax2.set_xlabel('字符数')
        ax2.set_ylabel('频率')
        ax2.set_title('微博字符长度分布')
        
        # 添加统计信息
        char_mean = np.mean(char_lengths)
        char_median = np.median(char_lengths)
        ax2.axvline(char_mean, color='red', linestyle='--', label=f'均值: {char_mean:.2f}')
        ax2.axvline(char_median, color='green', linestyle='--', label=f'中位数: {char_median:.2f}')
        ax2.legend()
        
        # 添加总标题
        plt.suptitle(f'{dataset_name}微博文本长度分布 (共{len(all_tweets)}条微博)', fontsize=16)
        
        # 保存图像
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)
        plt.close()
        
        # 打印统计信息
        logger.info(f"\n{dataset_name} 词长统计 (jieba分词):")
        logger.info(f"  均值: {word_mean:.2f}")
        logger.info(f"  中位数: {word_median:.2f}")
        logger.info(f"  最小值: {min(word_lengths)}")
        logger.info(f"  最大值: {max(word_lengths)}")
        logger.info(f"  标准差: {np.std(word_lengths):.2f}")
        
        logger.info(f"\n{dataset_name} 字符长度统计:")
        logger.info(f"  均值: {char_mean:.2f}")
        logger.info(f"  中位数: {char_median:.2f}")
        logger.info(f"  最小值: {min(char_lengths)}")
        logger.info(f"  最大值: {max(char_lengths)}")
        logger.info(f"  标准差: {np.std(char_lengths):.2f}")
    
    def __len__(self):
        return len(self.topics)
    
    def __getitem__(self, idx):
        topic = self.topics[idx]
        tweets = self.tweets[idx]
        sentences = self.sentences[idx]  # 获取原始句子
        tweet_ids = self.tweet_ids[idx]  # 获取推文ID
        label = self.labels[idx]
        stat_features = self.statistical_features[idx]
        
        # 统计特征转为张量
        stat_features_list = list(stat_features.values())
        stat_features_tensor = torch.tensor(stat_features_list, dtype=torch.float)
        
        # 加载图像
        if self.use_images:
            images, image_masks = self.load_tweet_images(topic, tweet_ids)
        else:
            images = None
            image_masks = None
        
        # 分词
        if self.encoder_type == "bert":
            encodings = []
            for text in tweets:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                encodings.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0)
                })
        
            # 堆叠所有编码
            input_ids = torch.stack([enc['input_ids'] for enc in encodings])
            attention_mask = torch.stack([enc['attention_mask'] for enc in encodings])
            return {
                'topic': topic,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long),
                'statistical_features': stat_features_tensor,
                'images': images,
                'image_masks': image_masks
            }

        elif self.encoder_type == "sentence-transformer":
            # still truncate the sentences
            sentences = [s[:self.max_len] for s in sentences]

            # add the topic as first sentence
            topic_sentence = f"Topic: {topic}"
            sentences = [topic_sentence] + sentences  # 将话题作为第一句

            # 对于sentence-transformer，直接返回原始句子
            return {
                'topic': topic,
                'input_ids': None,  # 不需要input_ids
                'attention_mask': None,  # 不需要attention_mask
                'label': torch.tensor(label, dtype=torch.long),
                'statistical_features': stat_features_tensor,
                'sentences': sentences,  # 返回原始句子
                'images': images,
                'image_masks': image_masks
            }

def custom_collate_fn(batch):
    """自定义collate函数，处理包含None值的批次"""
    # 保持原有代码不变
    elem = batch[0]
    result = {}
    
    # 处理topic字段（字符串列表）
    if 'topic' in elem:
        result['topic'] = [d['topic'] for d in batch]
    
    # 处理各类张量字段
    for key in ['input_ids', 'attention_mask', 'label', 'statistical_features', 'images', 'image_masks']:
        if key in elem and elem[key] is not None:
            try:
                result[key] = torch.stack([d[key] for d in batch])
            except Exception as e:
                logger.critical(f"在collate过程中堆叠字段 {key} 时出错: {e}")
                # show the problematic items
                for d in batch:
                    logger.critical(f"Item {key}: {d[key]}")
                raise e
        else:
            result[key] = None
    
    # 处理sentences字段（句子列表）
    if 'sentences' in elem:
        result['sentences'] = [d['sentences'] for d in batch]
    
    return result

def split_data(data_dir, emergency_level_file, train_ratio=0.8, random_state=42):
    """将话题分割为训练集和验证集"""
    # 加载话题预警级别
    with open(emergency_level_file, 'r', encoding='utf-8') as f:
        topic_emergency_levels = json.load(f)
    
    # 获取所有话题文件夹
    topics = []
    for folder in os.listdir(data_dir):
        topic_path = os.path.join(data_dir, folder)
        if os.path.isdir(topic_path) and os.path.exists(os.path.join(topic_path, f"{folder}.csv")):
            topics.append(folder)
    
    # 按预警级别对话题进行分组
    topics_by_level = {}
    for topic in topics:
        if topic in topic_emergency_levels:
            level = topic_emergency_levels[topic]
            if level not in topics_by_level:
                topics_by_level[level] = []
            topics_by_level[level].append(topic)
    
    # 对每个预警级别进行分层抽样
    train_topics = []
    val_topics = []
    
    for level, level_topics in topics_by_level.items():
        level_train, level_val = train_test_split(
            level_topics, train_size=train_ratio, random_state=random_state
        )
        train_topics.extend(level_train)
        val_topics.extend(level_val)
    
    # 保存分割结果
    split_info = {
        'train_topics': train_topics,
        'val_topics': val_topics
    }
    
    with open(os.path.join(data_dir, 'data_split.json'), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    logger.info(f"数据分割完成: {len(train_topics)}个话题用于训练, {len(val_topics)}个话题用于验证")
    
    # 检查预警级别分布
    train_levels = [topic_emergency_levels[t] for t in train_topics]
    val_levels = [topic_emergency_levels[t] for t in val_topics]
    
    logger.info("训练集预警级别分布:", Counter(train_levels))
    logger.info("验证集预警级别分布:", Counter(val_levels))
    
    return train_topics, val_topics

def get_data_loaders(config: Config, train_topics, val_topics, data_dir, emergency_level_file, tokenizer):
    """创建数据加载器"""
    
    # 加载话题预警级别
    with open(emergency_level_file, 'r', encoding='utf-8') as f:
        topic_emergency_levels = json.load(f)

    # reformat the topic_emergency_levels from list to dict
    try:
        topic_emergency_levels = {item['topic']: item['severity'] for item in topic_emergency_levels}
    except Exception as e:
        # show errored place
        for item in topic_emergency_levels:
            print(item)

        raise e

    # from severity string to int "严重", "中等", "轻微"
    severity_map = {"严重": 2, "中等": 1, "轻微": 0}
    topic_emergency_levels = {k: severity_map[v] for k, v in topic_emergency_levels.items()}
    
    # 创建数据集 - 传入配置的注释目录用于缓存
    train_dataset = TopicDataset(
        train_topics, data_dir, topic_emergency_levels, tokenizer,
        max_len=config.MAX_LEN, max_tweets=config.MAX_TWEETS,
        encoder_type=config.ENCODER_TYPE,
        annotation_dir=config.ANNOTATION_DIR, 
        use_images=config.USE_IMAGES,
    )
    val_dataset = TopicDataset(
        val_topics, data_dir, topic_emergency_levels, tokenizer,
        max_len=config.MAX_LEN, max_tweets=config.MAX_TWEETS,
        encoder_type=config.ENCODER_TYPE,
        annotation_dir=config.ANNOTATION_DIR, 
        use_images=config.USE_IMAGES,
    )

    # 绘制词长分布图
    # train_dataset.plot_word_length_distribution(
    #     save_path=os.path.join(config.ANNOTATION_DIR, 'train_word_length_distribution.png'),
    #     dataset_name='训练集'
    # )
    # val_dataset.plot_word_length_distribution(
    #     save_path=os.path.join(config.ANNOTATION_DIR, 'val_word_length_distribution.png'),
    #     dataset_name='验证集'
    # )

    # print some examples
    logger.info("Example:".center(60, "="))
    for i in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Topic: {train_dataset.topics[i]}")
        logger.info(f"Label: {train_dataset.labels[i]}")
        logger.info(f"Statistical Features: {train_dataset.statistical_features[i]}")
        logger.info(f"Tweets: {train_dataset.tweets[i][:2]}")  # 只显示前两条微博示例
        logger.info("-" * 60)
    logger.info("=" * 60)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, 
                                batch_size=config.BATCH_SIZE,
                                num_workers=config.NUM_WORKERS, 
                                collate_fn=custom_collate_fn,
                                shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=config.BATCH_SIZE,
                            num_workers=config.NUM_WORKERS,
                            collate_fn=custom_collate_fn)
    
    return train_loader, val_loader, train_dataset, val_dataset

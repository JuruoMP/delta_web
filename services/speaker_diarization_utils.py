import torch
import numpy as np
from pyannote.audio import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerFeatureExtractor:
    def __init__(self, hf_api_key):
        """
        初始化说话人特征提取器
        :param hf_api_key: Hugging Face API密钥
        """
        self.hf_api_key = hf_api_key
        self.speaker_database = {}
        self.database_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/speaker_database.json")
        
        # 确保数据目录存在
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"创建数据目录: {data_dir}")

        # 加载数据库
        self._load_database()

        # 延迟加载模型，仅在需要时加载
        self.diarization_pipeline = None
        self.embedding_pipeline = None

    def _load_models(self):
        """
        加载pyannote模型
        """
        if self.diarization_pipeline is None and self.embedding_pipeline is None:
            try:
                logger.info("开始加载pyannote模型...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.hf_api_key
                )
                self.embedding_pipeline = Pipeline.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=self.hf_api_key
                )
                logger.info("pyannote模型加载成功")
            except Exception as e:
                logger.error(f"加载pyannote模型失败: {str(e)}")
                raise

    def _load_database(self):
        """
        从文件加载说话人特征数据库
        """
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, 'r') as f:
                    self.speaker_database = json.load(f)
                logger.info(f"已加载说话人数据库，包含 {len(self.speaker_database)} 个说话人")
            except Exception as e:
                logger.error(f"加载说话人数据库失败: {str(e)}")
                self.speaker_database = {}
        else:
            logger.info("说话人数据库不存在，将创建新数据库")
            self.speaker_database = {}

    def _save_database(self):
        """
        保存说话人特征数据库到文件
        """
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.speaker_database, f, indent=2)
            logger.info(f"已保存说话人数据库到 {self.database_path}")
        except Exception as e:
            logger.error(f"保存说话人数据库失败: {str(e)}")

    def extract_speaker_features(self, audio_path, diarization_result=None):
        """
        从音频文件中提取说话人特征
        :param audio_path: 音频文件路径
        :param diarization_result: 可选，已有的说话人分离结果
        :return: 说话人特征字典 {speaker_id: feature_vector}
        """
        # 确保模型已加载
        self._load_models()

        # 如果没有提供分离结果，执行分离
        if diarization_result is None:
            try:
                logger.info(f"对音频 {audio_path} 执行说话人分离...")
                diarization_result = self.diarization_pipeline(audio_path)
                logger.info("说话人分离完成")
            except Exception as e:
                logger.error(f"说话人分离失败: {str(e)}")
                return {}

        # 提取每个说话人的特征
        speaker_features = {}
        try:
            logger.info("开始提取说话人特征...")
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                # 提取说话人语音片段的嵌入向量
                embedding = self.embedding_pipeline(
                    audio_path,
                    start_time=segment.start,
                    end_time=segment.end
                )
                # 取平均值作为该说话人的特征
                embedding_np = embedding.detach().cpu().numpy().mean(axis=0)

                if speaker not in speaker_features:
                    speaker_features[speaker] = []
                speaker_features[speaker].append(embedding_np)

            # 计算每个说话人的平均特征
            for speaker in speaker_features:
                speaker_features[speaker] = np.mean(speaker_features[speaker], axis=0).tolist()

            logger.info(f"成功提取 {len(speaker_features)} 个说话人的特征")
        except Exception as e:
            logger.error(f"提取说话人特征失败: {str(e)}")
            return {}

        return speaker_features

    def match_speakers_across_recordings(self, new_speaker_features, threshold=0.85):
        print('2222222222222222')
        """
        跨录音匹配说话人
        :param new_speaker_features: 新录音的说话人特征 {speaker_id: feature_vector}
        :param threshold: 相似度阈值
        :return: 匹配结果 {original_speaker_id: matched_global_speaker_id}
        """
        matched_speakers = {}
        existing_speakers = list(self.speaker_database.keys())

        logger.info(f"开始跨录音匹配说话人，现有 {len(existing_speakers)} 个说话人，新录音有 {len(new_speaker_features)} 个说话人")

        for new_speaker, new_features in new_speaker_features.items():
            if not existing_speakers:
                # 如果数据库为空，添加新说话人
                new_id = f"speaker_1"
                self.speaker_database[new_id] = new_features
                matched_speakers[new_speaker] = new_id
                logger.info(f"数据库为空，添加新说话人: {new_id}")
            else:
                # 计算与现有说话人的相似度
                similarities = []
                for existing_speaker in existing_speakers:
                    existing_features = np.array(self.speaker_database[existing_speaker])
                    similarity = cosine_similarity([new_features], [existing_features])[0][0]
                    similarities.append((existing_speaker, similarity))

                # 找到最相似的说话人
                similarities.sort(key=lambda x: x[1], reverse=True)
                best_match, best_similarity = similarities[0]

                if best_similarity >= threshold:
                    # 匹配成功
                    matched_speakers[new_speaker] = best_match
                    logger.info(f"说话人 {new_speaker} 匹配到现有说话人 {best_match}，相似度: {best_similarity:.4f}")
                else:
                    # 添加新说话人
                    new_id = f"speaker_{len(self.speaker_database) + 1}"
                    self.speaker_database[new_id] = new_features
                    matched_speakers[new_speaker] = new_id
                    logger.info(f"说话人 {new_speaker} 未匹配到现有说话人，添加为新说话人: {new_id}，最高相似度: {best_similarity:.4f}")

        # 保存数据库
        self._save_database()
        return matched_speakers

    def reset_database(self):
        """
        重置说话人数据库
        """
        self.speaker_database = {}
        if os.path.exists(self.database_path):
            os.remove(self.database_path)
            logger.info("已重置说话人数据库")
        else:
            logger.info("说话人数据库不存在，无需重置")

    def get_speaker_count(self):
        """
        获取数据库中说话人的数量
        :return: 说话人数量
        """
        return len(self.speaker_database)

    def get_speaker_database(self):
        """
        获取说话人数据库
        :return: 说话人数据库字典
        """
        return self.speaker_database
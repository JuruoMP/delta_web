import numpy as np
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class SpeakerFeatureExtractor:
    def __init__(self, hf_token):
        """
        初始化说话人特征提取器
        :param hf_token: Hugging Face API token
        """ 
        self.hf_token = hf_token
        # 初始化说话人嵌入模型
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cpu"))
        # 初始化说话人数据库（在实际应用中，这应该是一个持久化的数据库）
        self.speaker_database = {}
        self.next_speaker_id = 1
    
    def extract_speaker_features(self, audio_file, diarization_segments):
        """
        从音频文件和说话人分段中提取说话人特征
        :param audio_file: 音频文件路径
        :param diarization_segments: 说话人分段信息
        :return: 说话人特征字典
        """
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # 存储每个说话人的特征
        speaker_features = {}
        
        # 遍历每个说话人分段
        for segment in diarization_segments:
            speaker = segment['speaker']
            start_time = segment['start']
            end_time = segment['end']
            
            # 计算样本索引
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            # 提取该段的音频
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # 提取说话人嵌入
            embedding = self.embedding_model(segment_waveform)
            
            # 存储特征
            if speaker not in speaker_features:
                speaker_features[speaker] = []
            speaker_features[speaker].append(embedding)
        
        # 计算每个说话人的平均特征
        for speaker in speaker_features:
            embeddings = torch.stack(speaker_features[speaker])
            mean_embedding = torch.mean(embeddings, dim=0)
            speaker_features[speaker] = mean_embedding
        
        return speaker_features
    
    def match_speakers_across_recordings(self, speaker_features, threshold=0.85):
        """
        跨录音匹配说话人
        :param speaker_features: 说话人特征
        :param threshold: 匹配阈值
        :return: 说话人映射字典
        """
        # 存储说话人映射
        speaker_mapping = {}
        
        # 遍历当前录音的每个说话人
        for local_speaker, features in speaker_features.items():
            best_match = None
            best_similarity = -1
            
            # 与数据库中的每个说话人比较
            for global_speaker, db_features in self.speaker_database.items():
                # 计算相似度
                similarity = torch.cosine_similarity(features, db_features, dim=0)
                
                # 如果相似度高于阈值且高于当前最佳匹配
                if similarity > threshold and similarity > best_similarity:
                    best_match = global_speaker
                    best_similarity = similarity
            
            # 如果找到匹配的说话人
            if best_match:
                speaker_mapping[local_speaker] = best_match
            else:
                # 添加新说话人到数据库
                new_speaker_id = self.add_speaker(features)
                speaker_mapping[local_speaker] = new_speaker_id
        
        return speaker_mapping
    
    def get_speaker_count(self):
        """
        获取数据库中的说话人数量
        :return: 说话人数量
        """
        return len(self.speaker_database)

    def add_speaker(self, speaker_features):
        """
        添加新说话人到数据库
        :param speaker_features: 说话人特征
        :return: 新说话人ID
        """
        speaker_id = f"SPEAKER_{self.next_speaker_id}"
        self.speaker_database[speaker_id] = speaker_features
        self.next_speaker_id += 1
        return speaker_id
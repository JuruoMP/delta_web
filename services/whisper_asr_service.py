import os
os.environ["PYTORCH_WEIGHTS_ONLY"] = "0"
import time
import uuid
import urllib
from dotenv import load_dotenv
# 导入说话人特征提取器
from services.speaker_diarization_utils import SpeakerFeatureExtractor
# 尝试导入whisper.cpp的Python绑定
# 注意：whisper.cpp的Python绑定可能有不同的导入方式
# 这里使用whispercpp作为示例

try:
    import whispercpp as whisper
except ImportError:
    try:
        # 如果whispercpp不可用，尝试使用openai-whisper
        import whisper
    except ImportError:
        raise ImportError("Please install whisper.cpp Python bindings or openai-whisper")

# 尝试导入whisperx
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("WhisperX not installed. Speaker diarization will be disabled.")

# 尝试导入pyannote.audio
PYANNOTE_AVAILABLE = False
try:
    import pyannote
    PYANNOTE_AVAILABLE = True
except ImportError:
    print("pyannote.audio not installed. Speaker feature extraction will be disabled.")


# 加载环境变量
load_dotenv()

class WhisperASRService:
    def __init__(self):
        # 从环境变量获取配置
        self.model_size = os.getenv("WHISPER_MODEL_SIZE", "base")
        self.sample_rate = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        self.enable_diarization = os.getenv("WHISPERX_ENABLE_DIARIZATION", "False").lower() == "true"
        self.enable_speaker_feature = os.getenv("WHISPERX_ENABLE_SPEAKER_FEATURE", "True").lower() == "true"
        self.speaker_matching_threshold = float(os.getenv("WHISPERX_SPEAKER_MATCHING_THRESHOLD", "0.85"))
        
        # 初始化说话人特征提取器
        self.speaker_extractor = None
        if self.enable_speaker_feature and WHISPERX_AVAILABLE and PYANNOTE_AVAILABLE and self.hf_api_key:
            try:
                self.speaker_extractor = SpeakerFeatureExtractor(self.hf_api_key)
                print(f"SpeakerFeatureExtractor initialized. Database contains {self.speaker_extractor.get_speaker_count()} speakers.")
            except Exception as e:
                print(f"Failed to initialize SpeakerFeatureExtractor: {e}")
                self.enable_speaker_feature = False
        
        # 验证WhisperX配置
        if self.enable_diarization and not WHISPERX_AVAILABLE:
            print("Warning: WHISPERX_ENABLE_DIARIZATION is True but WhisperX is not installed. Diarization will be disabled.")
            self.enable_diarization = False
        
        if self.enable_diarization and not self.hf_api_key:
            print("Warning: WHISPERX_ENABLE_DIARIZATION is True but HF_API_KEY is not set. Diarization will be disabled.")
            self.enable_diarization = False
        
        # 惰性加载模型
        self.models = {}
        self.current_language = None
        self.current_model = None
        self.whisperx_model = None
        self.align_model = None
        self.diarize_model = None
        
        # 确定使用的库类型
        if hasattr(whisper, 'load_model'):
            self.library_type = 'openai-whisper'
        else:
            raise ImportError("Only openai-whisper library is supported")
        
        print(f"WhisperASRService initialized. Using {self.library_type} library and {self.model_size} size model.")
        if self.enable_diarization:
            print("Speaker diarization enabled using WhisperX.")
        
        # 删除模型路径验证相关代码，因为现在使用whisper库默认的模型加载方式
    
    def _validate_model_path(self, path):
        """
        验证模型文件路径是否存在
        :param path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Please check your WHISPER_MODEL_PATH environment variables.")
    
    def switch_model(self, language):
        """
        切换语言模型
        :param language: 语言代码，如'en'表示英文，'zh'表示中文
        """
        supported_languages = ['en', 'zh']
        if language not in supported_languages:
            raise ValueError(f"Unsupported language: {language}. Supported languages: {supported_languages}")
        
        # 如果模型尚未加载，则加载它
        if language not in self.models:
            self._load_model(language)
        
        self.current_language = language
        self.current_model = self.models[language]
        print(f"Switched to {language} model")
    
    def _load_model(self, language):
        """
        加载指定语言的模型
        :param language: 语言代码
        """
        try:
            if self.library_type == 'openai-whisper':
                # 确保设置PYTORCH_WEIGHTS_ONLY环境变量
                os.environ["PYTORCH_WEIGHTS_ONLY"] = "0"
                
                # 使用monkeypatching修改torch.load行为
                import torch
                import functools
                
                # 保存原始的torch.load函数
                original_torch_load = torch.load
                
                # 创建包装函数，强制设置weights_only=False
                @functools.wraps(original_torch_load)
                def patched_torch_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)
                
                # 替换torch.load
                torch.load = patched_torch_load
                
                # 使用whisper库自带的模型加载功能
                model_name = f"{self.model_size}.en" if language == 'en' else self.model_size
                self.models[language] = whisper.load_model(self.model_size)
                
                # 恢复原始的torch.load函数
                torch.load = original_torch_load
            print(f"Loaded {language} model: {self.model_size}")
        except Exception as e:
            print(f"Error loading {language} model: {e}")
            raise
    
    def _load_whisperx_model(self):
        """
        加载WhisperX模型和说话人区分模型
        """
        if not self.enable_diarization or not WHISPERX_AVAILABLE:
            return
        
        try:
            # 加载WhisperX模型
            device = "cuda" if whisperx.utils.is_cuda_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "float32"
            
            # 使用多语言模型
            self.whisperx_model = whisperx.load_model(
                self.model_size,
                device,
                compute_type=compute_type,
                download_root=os.path.dirname(self.model_path_zh)
            )
            
            # 加载对齐模型
            self.align_model, metadata = whisperx.load_align_model(
                language_code=self.current_language,
                device=device
            )
            
            # 加载说话人区分模型
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=self.hf_api_key,
                device=device
            )
            
            print("Loaded WhisperX models for speaker diarization")
        except Exception as e:
            print(f"Error loading WhisperX models: {e}")
            self.enable_diarization = False
        
    def submit_task(self, file_url, format="mp3"):
        """
        提交ASR任务
        :param file_url: 音频文件的路径或URL
        :param format: 音频文件格式
        :return: 任务ID和日志ID
        """
        task_id = str(uuid.uuid4())
        x_tt_logid = str(uuid.uuid4())
        
        headers = {
            "X-Api-Status-Code": "20000000",
            "X-Api-Message": "Success",
            "X-Tt-Logid": x_tt_logid
        }
        
        print(f'Submit task id: {task_id}')
        return task_id, x_tt_logid
        
    def query_task(self, task_id, x_tt_logid, file_url, format="mp3"):
        """
        查询ASR任务结果
        :param task_id: 任务ID
        :param x_tt_logid: 日志ID
        :param file_url: 音频文件的路径或URL
        :param format: 音频文件格式
        :return: 响应对象
        """
        # 模拟响应头
        headers = {
            "X-Api-Status-Code": "20000000",
            "X-Api-Message": "Success",
            "X-Tt-Logid": x_tt_logid
        }
        
        try:
            # 检查是否启用说话人区分
            if self.enable_diarization and WHISPERX_AVAILABLE:
                # 使用WhisperX进行转录和说话人区分
                return self._query_task_with_diarization(task_id, x_tt_logid, file_url, format, headers)
            else:
                # 使用传统Whisper进行转录
                return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        except Exception as e:
            headers["X-Api-Status-Code"] = "50000000"
            headers["X-Api-Message"] = str(e)
            response = type('obj', (object,), {})()
            response.headers = headers
            response.json = lambda: {"error": str(e)}
            return response
    
    def _query_task_without_diarization(self, task_id, x_tt_logid, file_url, format, headers):
        """
        不使用说话人区分的查询任务实现
        """
        # 根据不同的库使用不同的转录方式
        if self.library_type == 'openai-whisper':
            # 使用完整路径
            full_file_path = os.path.abspath(file_url)
            result = self.current_model.transcribe(full_file_path, language=self.current_language, fp16=False)
            
            # 标准化结果格式
            segments = []
            if 'segments' in result:
                for s in result['segments']:
                    segments.append({
                        "text": s['text'],
                        "start_time": s['start'],
                        "end_time": s['end']
                    })
            
            standardized_result = {
                "text": result.get('text', ''),
                "segments": segments
            }
        elif self.library_type == 'whispercpp':
            result = self.current_model.transcribe(file_url, language=self.current_language)
            
            # 标准化结果格式
            segments = []
            if 'segments' in result:
                for s in result['segments']:
                    segments.append({
                        "text": s['text'],
                        "start_time": s['start'],
                        "end_time": s['end']
                    })
            
            standardized_result = {
                "text": result.get('text', ''),
                "segments": segments
            }
        else:
            raise Exception("Unsupported whisper library")
        
        # 构建响应
        response = type('obj', (object,), {})()
        response.headers = headers
        response.json = lambda: {
            "result": standardized_result
        }
        return response
    
    def _query_task_with_diarization(self, task_id, x_tt_logid, file_url, format, headers):
        """
        使用说话人区分的查询任务实现
        """
        if not self.whisperx_model:
            self._load_whisperx_model()
            
        if not self.whisperx_model:
            # 如果加载失败，降级到不使用说话人区分的模式
            return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        
        try:
            # 使用WhisperX转录
            # 使用完整路径
            full_file_path = os.path.abspath(file_url)
            audio = whisperx.load_audio(full_file_path)
            result = self.whisperx_model.transcribe(audio, language=self.current_language)
            
            # 对齐时间戳
            result = whisperx.align(result["segments"], self.align_model, self.align_model.metadata, audio, device="cuda" if whisperx.utils.is_cuda_available() else "cpu")
            
            # 进行说话人区分
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # 提取说话人特征并进行跨录音匹配
            global_speaker_mapping = {}
            if self.enable_speaker_feature and self.speaker_extractor:
                try:
                    # 提取当前录音的说话人特征
                    speaker_features = self.speaker_extractor.extract_speaker_features(file_url, diarize_segments)
                    
                    # 跨录音匹配说话人
                    global_speaker_mapping = self.speaker_extractor.match_speakers_across_recordings(
                        speaker_features, 
                        threshold=self.speaker_matching_threshold
                    )
                    print(f"Matched {len(global_speaker_mapping)} speakers across recordings.")
                except Exception as e:
                    print(f"Error in speaker feature extraction or matching: {e}")

            # 标准化结果格式
            segments = []
            for s in result["segments"]:
                # 使用全局说话人ID替换本地说话人ID
                local_speaker = s.get("speaker", "UNKNOWN")
                global_speaker = global_speaker_mapping.get(local_speaker, local_speaker)
                
                segment = {
                    "text": s["text"],
                    "start_time": s["start"],
                    "end_time": s["end"],
                    "speaker": global_speaker,
                    "local_speaker": local_speaker  # 保留原始本地说话人ID
                }
                segments.append(segment)
            
            # 按说话人分组的文本
            speaker_texts = {}
            for segment in segments:
                speaker = segment["speaker"]
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = []
                speaker_texts[speaker].append(segment["text"])

            # 合并每个说话人的文本
            for speaker in speaker_texts:
                speaker_texts[speaker] = " ".join(speaker_texts[speaker])

            # 添加说话人映射信息
            speaker_mapping_info = {
                "local_to_global": global_speaker_mapping,
                "total_speakers_in_database": self.speaker_extractor.get_speaker_count() if self.speaker_extractor else 0
            }
            
            standardized_result = {
                "text": result["text"],
                "segments": segments,
                "speaker_texts": speaker_texts,
                "speaker_mapping": speaker_mapping_info
            }
            
            # 构建响应
            response = type('obj', (object,), {})()
            response.headers = headers
            response.json = lambda: {
                "result": standardized_result
            }
            return response
        except Exception as e:
            print(f"Error with diarization: {e}")
            # 出错时降级到不使用说话人区分的模式
            return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        
    def transcribe_audio(self, file_url, format="mp3", language=None, enable_diarization=None):
        """
        转录音频文件为文本
        :param file_url: 音频文件的路径或URL
        :param format: 音频文件格式，支持mp3、wav、ogg等
        :param language: 语言代码，如'en'表示英文，'zh'表示中文，可选参数
        :param enable_diarization: 是否启用说话人区分，可选参数，默认为全局配置
        :return: 转录结果
        """
        if language:
            self.switch_model(language)
        
        # 临时覆盖全局配置
        original_enable_diarization = self.enable_diarization
        if enable_diarization is not None:
            self.enable_diarization = enable_diarization
        
        task_id, x_tt_logid = self.submit_task(file_url, format)
        
        # 模拟异步处理的延迟
        time.sleep(1)
        
        query_response = self.query_task(task_id, x_tt_logid, file_url, format)
        
        # 恢复原始配置
        self.enable_diarization = original_enable_diarization
        
        code = query_response.headers.get('X-Api-Status-Code', "")
        if code == '20000000':  # 任务完成
            result = query_response.json()
            print("SUCCESS!")
            return result
        else:  # 任务失败
            print("FAILED!")
            raise Exception(f"Transcription failed with status code: {code}")


# 创建服务实例
whisper_asr_service = WhisperASRService()

# 禁用说话人区分功能以进行测试
whisper_asr_service.enable_diarization = False

# 默认加载多语言模型
whisper_asr_service.switch_model('zh')

if __name__ == '__main__':
    # 示例用法
    try:
        # 使用英文模型
        result_en = whisper_asr_service.transcribe_audio("tmp/demo_audio.mp3")
        print("English transcription:", result_en['result']['text'])
        
        # 切换到中文模型
        result_zh = whisper_asr_service.transcribe_audio("tmp/demo_audio.mp3", language="zh")
        print("Chinese transcription:", result_zh['result']['text'])
    except Exception as e:
        print(f"Error in example usage: {e}")
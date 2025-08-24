import os
os.environ["PYTORCH_WEIGHTS_ONLY"] = "0"
import time
import uuid
import urllib
import torch
from dotenv import load_dotenv
import logging
# 导入说话人特征提取器
from services.speaker_diarization_utils import SpeakerFeatureExtractor

# 配置日志记录
logger = logging.getLogger(__name__)

try:
    import whisper
except ImportError:
    logger.error("Please install openai-whisper")
    raise ImportError("Please install openai-whisper")

# 尝试导入whisperx
try:
    import whisperx
    WHISPERX_AVAILABLE = True
    logger.info("WhisperX is available.")
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.warning("WhisperX not installed. Speaker diarization will be disabled.")

# 尝试导入pyannote.audio
PYANNOTE_AVAILABLE = False
try:
    import pyannote.audio
    PYANNOTE_AVAILABLE = True
    logger.info("pyannote.audio is available.")
except ImportError:
    logger.warning("pyannote.audio not installed. Speaker feature extraction will be disabled.")


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
        self.model_path = os.getenv("WHISPER_MODEL_PATH", "asr_models/")
        
        # WhisperX相关配置
        self.whisperx_model_name = os.getenv("WHISPERX_MODEL_SIZE", "base")
        self.device = os.getenv("WHISPERX_DEVICE", "cpu")
        self.compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
        self.whisperx_model_path = self.model_path
        self.diarize_model_name = os.getenv("WHISPERX_DIARIZE_MODEL_NAME", "pyannote/speaker-diarization-3.1")
        self.diarize_model_cache_dir = self.model_path
        self.hf_token = self.hf_api_key
        
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
        print(f"开始加载WhisperX模型, enable_diarization={self.enable_diarization}, WHISPERX_AVAILABLE={WHISPERX_AVAILABLE}")
        # 尝试加载WhisperX模型
        try:
            # 只有在启用说话人区分时才加载WhisperX模型
            if self.enable_diarization and WHISPERX_AVAILABLE:
                print(f"正在加载WhisperX模型: model_name={self.whisperx_model_name}, device={self.device}, compute_type={self.compute_type}, download_root={self.whisperx_model_path}")
                # 加载WhisperX模型
                try:
                    self.whisperx_model = whisperx.load_model(self.whisperx_model_name, self.device, compute_type=self.compute_type, download_root=self.whisperx_model_path)
                    print(f"WhisperX模型加载成功: {self.whisperx_model}")
                except Exception as e:
                    print(f"加载WhisperX模型时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.whisperx_model = None
                    return
                
                # 加载对齐模型
                try:
                    print(f"正在加载对齐模型: language_code={self.current_language}, device={self.device}")
                    self.align_model, self.align_metadata = whisperx.load_align_model(language_code=self.current_language, device=self.device)
                    print(f"对齐模型加载成功: {self.align_model}")
                except Exception as e:
                    print(f"加载对齐模型时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.align_model = None
                    self.align_metadata = None
                
                # 加载说话人区分模型
                try:
                    print(f"正在加载说话人区分模型: model_name={self.diarize_model_name}, device={self.device}, use_auth_token={bool(self.hf_token)}")
                    self.diarize_model = whisperx.diarize.DiarizationPipeline(
                        model_name=self.diarize_model_name,
                        device=self.device,
                        use_auth_token=self.hf_token
                    )
                    print(f"说话人区分模型加载成功: {self.diarize_model}")
                except Exception as e:
                    print(f"加载说话人区分模型时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    self.diarize_model = None
            else:
                print("WhisperX模型未加载，因为说话人区分功能未启用或WhisperX库不可用")
                print(f"未加载的原因: enable_diarization={self.enable_diarization}, WHISPERX_AVAILABLE={WHISPERX_AVAILABLE}")
                self.whisperx_model = None
                self.diarize_model = None
        except Exception as e:
            print(f"加载WhisperX模型时出错: {e}")
            import traceback
            traceback.print_exc()
            self.whisperx_model = None
            self.diarize_model = None
            # 不抛出异常，允许服务降级到基础Whisper模型

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
        
    def query_task(self, task_id, x_tt_logid, file_url, format="mp3", enable_diarization=None):
        """
        查询ASR任务结果
        :param task_id: 任务ID
        :param x_tt_logid: 日志ID
        :param file_url: 音频文件的路径或URL
        :param format: 音频文件格式
        :param enable_diarization: 是否启用说话人区分
        :return: 响应对象
        """
        # 模拟响应头
        headers = {
            "X-Api-Status-Code": "20000000",
            "X-Api-Message": "Success",
            "X-Tt-Logid": x_tt_logid
        }
        
        try:
            # 如果提供了enable_diarization参数，则使用它，否则使用全局配置
            use_diarization = enable_diarization if enable_diarization is not None else self.enable_diarization
            print(f"查询任务参数: use_diarization={use_diarization}, WHISPERX_AVAILABLE={WHISPERX_AVAILABLE}, enable_diarization参数={enable_diarization}, self.enable_diarization={self.enable_diarization}")
            print(f"条件判断: use_diarization and WHISPERX_AVAILABLE = {use_diarization and WHISPERX_AVAILABLE}")
            print(f"use_diarization类型: {type(use_diarization)}, WHISPERX_AVAILABLE类型: {type(WHISPERX_AVAILABLE)}")
            
            # 检查是否启用说话人区分
            if use_diarization and WHISPERX_AVAILABLE:
                print("启用说话人区分功能")
                # 使用WhisperX进行转录和说话人区分
                return self._query_task_with_diarization(task_id, x_tt_logid, file_url, format, headers)
            else:
                print("未启用说话人区分功能")
                print(f"未启用说话人区分的原因: use_diarization={use_diarization}, WHISPERX_AVAILABLE={WHISPERX_AVAILABLE}")
                # 使用传统Whisper进行转录
                return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        except Exception as e:
            print(f"查询任务时出错: {e}")
            import traceback
            traceback.print_exc()
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
        print(f"开始使用说话人区分功能进行查询, whisperx_model={bool(self.whisperx_model)}, diarize_model={bool(self.diarize_model)}")
        if not self.whisperx_model:
            print("WhisperX模型未加载，正在尝试加载...")
            self._load_whisperx_model()
            
        if not self.whisperx_model:
            print("WhisperX模型加载失败，降级到不使用说话人区分的模式")
            # 如果加载失败，降级到不使用说话人区分的模式
            return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        
        # 检查说话人区分模型是否加载成功
        if not self.diarize_model:
            print("说话人区分模型未加载，降级到不使用说话人区分的模式")
            return self._query_task_without_diarization(task_id, x_tt_logid, file_url, format, headers)
        
        try:
            print(f"开始处理音频文件: {file_url}")
            # 使用WhisperX转录
            # 使用完整路径
            full_file_path = os.path.abspath(file_url)
            print(f"音频文件完整路径: {full_file_path}")
            audio = whisperx.load_audio(full_file_path)
            print(f"音频加载成功")
            transcribe_result = self.whisperx_model.transcribe(audio, language=self.current_language)
            print(f"转录完成，结果段落数: {len(transcribe_result.get('segments', []))}")
            
            # 手动从段落构建完整文本
            full_text = " ".join([s['text'].strip() for s in transcribe_result.get('segments', [])])
            
            # 对齐时间戳
            import torch
            try:
                if self.align_model:
                    print(f"开始对齐时间戳")
                    aligned_result = whisperx.align(transcribe_result["segments"], self.align_model, self.align_metadata, audio, device="cuda" if torch.cuda.is_available() else "cpu")
                    print(f"时间戳对齐完成")
                else:
                    aligned_result = transcribe_result
                    print("对齐模型未加载，跳过时间戳对齐步骤")
            except AttributeError as e:
                # 如果对齐模型没有metadata属性，则跳过对齐步骤
                print(f"对齐模型缺少metadata属性，跳过时间戳对齐步骤: {e}")
                aligned_result = transcribe_result
                pass
            except Exception as e:
                # 如果对齐过程中出现其他错误，也跳过对齐步骤
                print(f"时间戳对齐过程中出现错误，跳过对齐步骤: {e}")
                aligned_result = transcribe_result
                pass
            
            # 进行说话人区分
            print(f"开始说话人区分")
            diarize_segments = self.diarize_model(full_file_path)
            print(f"说话人区分完成，段落数: {len(diarize_segments)}")
            result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            print(f"说话人分配完成")

            # 提取说话人特征并进行跨录音匹配
            global_speaker_mapping = {}
            if self.enable_speaker_feature and self.speaker_extractor:
                try:
                    print(f"开始提取说话人特征")
                    # 提取当前录音的说话人特征
                    speaker_features = self.speaker_extractor.extract_speaker_features(file_url, diarize_segments)
                    
                    # 跨录音匹配说话人
                    global_speaker_mapping = self.speaker_extractor.match_speakers_across_recordings(
                        speaker_features, 
                        threshold=self.speaker_matching_threshold
                    )
                    print(f"说话人特征提取和匹配完成")
                except Exception as e:
                    print(f"说话人特征提取或匹配时出错: {e}")

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
                "text": full_text,
                "segments": segments,
                "speaker_texts": speaker_texts,
                "speaker_mapping": speaker_mapping_info
            }
            
            print(f"标准化结果完成，包含说话人文本: {bool(speaker_texts)}")
            # 构建响应
            response = type('obj', (object,), {})()
            response.headers = headers
            response.json = lambda: {
                "result": standardized_result
            }
            return response
        except Exception as e:
            print(f"说话人区分过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
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
        
        print(f"transcribe_audio方法中的enable_diarization值: {self.enable_diarization}")
        print(f"传递给query_task的enable_diarization参数: {self.enable_diarization}")
        
        task_id, x_tt_logid = self.submit_task(file_url, format)
        
        # 模拟异步处理的延迟
        time.sleep(1)
        
        # 传递enable_diarization参数给query_task方法
        query_response = self.query_task(task_id, x_tt_logid, file_url, format, self.enable_diarization)
        
        # 恢复原始配置
        self.enable_diarization = original_enable_diarization
        
        code = query_response.headers.get('X-Api-Status-Code', "")
        if code == '20000000':  # 任务完成
            result = query_response.json()
            return result
        else:  # 任务失败
            raise Exception(f"Transcription failed with status code: {code}")


if __name__ == '__main__':
    print("!!!!!!!!!! EXECUTING MAIN BLOCK !!!!!!!!!!")
    import pprint
    # 创建服务实例
    whisper_asr_service = WhisperASRService()
    print(f"服务实例创建完成, enable_diarization={whisper_asr_service.enable_diarization}, whisperx_model={bool(whisper_asr_service.whisperx_model)}, diarize_model={bool(whisper_asr_service.diarize_model)}")

    # 启用说话人区分功能以进行测试
    whisper_asr_service.enable_diarization = True
    print(f"手动设置enable_diarization={whisper_asr_service.enable_diarization}")

    # 默认加载多语言模型
    whisper_asr_service.switch_model('en')
    print(f"模型切换完成, whisperx_model={bool(whisper_asr_service.whisperx_model)}, diarize_model={bool(whisper_asr_service.diarize_model)}")
    # 示例用法
    try:
        # 使用英文模型
        result_en = whisper_asr_service.transcribe_audio("tmp/demo_audio.wav")
        print("English transcription:")
        pprint.pprint(result_en['result'])
        
        # 打印说话人区分结果
        if 'speaker_texts' in result_en['result'] and result_en['result']['speaker_texts']:
            for speaker, text in result_en['result']['speaker_texts'].items():
                print(f"Speaker {speaker}: {text}")
        else:
            print("未找到说话人区分结果")
            if 'result' in result_en and isinstance(result_en['result'], dict):
                print("结果键:", result_en['result'].keys())
        
        # 切换到中文模型
        # result_zh = whisper_asr_service.transcribe_audio("tmp/demo_audio_zh.mp3", language="zh")
        # print("Chinese transcription:", result_zh['result'])
    except Exception as e:
        print(f"Error in example usage: {e}")
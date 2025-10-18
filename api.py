import os
import json
import hashlib
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit

# 导入必要的服务和模型
from extensions import db
from models import Conversation, Memory, Event
from services.llm_service import LLMService
from utils.llm_utils import LLMUtils
from utils.memory_utils import MemoryBank
from services.db_service import (
    add_conversation, get_all_conversations, clear_conversations,
    clear_all_data, get_latest_memory, add_memory
)

# 创建API蓝图
api_bp = Blueprint('api', __name__)

# 初始化服务
llm_service = LLMService()
llm_utils = LLMUtils(llm_service)
# 从Flask应用中获取已初始化的memory_bank实例，避免重复初始化
memory_bank = None

# 声明socketio实例，将在app.py中初始化
socketio = None

# 声明全局变量
asr_service = None
GLOBAL_LANG = None

# 标记是否已初始化
_asr_service_initialized = False

# 在应用上下文可用后初始化ASR服务
@api_bp.before_app_request
def init_asr_service():
    global asr_service, GLOBAL_LANG, _asr_service_initialized
    if not _asr_service_initialized:
        GLOBAL_LANG = current_app.config.get('GLOBAL_LANG', 'en')
        if GLOBAL_LANG == 'en':
            from services.whisper_asr_service import WhisperASRService
            asr_service = WhisperASRService()
            asr_service.switch_model('en')  # 强制使用英文
        elif GLOBAL_LANG == 'zh':
            from services.funasr_service import FunASRService
            asr_service = FunASRService()
        else:
            raise ValueError("Language not supported")
        _asr_service_initialized = True

# 配置
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}

# 工具函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 对话相关API
@api_bp.route('/conversations', methods=['GET'])
def get_conversations():
    """获取所有对话"""
    try:
        conversations = Conversation.query.order_by(Conversation.created_at.desc()).all()
        result = []
        for conv in conversations:
            result.append({
                'id': conv.id,
                'content': conv.content,
                'summary': conv.summary,
                'created_at': conv.created_at.isoformat()
            })
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/conversations/<int:conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """获取特定对话"""
    try:
        conv = Conversation.query.get(conv_id)
        if not conv:
            return jsonify({'status': 'error', 'message': '对话不存在'}), 404
        return jsonify({
            'status': 'success',
            'data': {
                'id': conv.id,
                'content': conv.content,
                'summary': conv.summary,
                'created_at': conv.created_at.isoformat()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/create_conversation', methods=['POST'])
def create_conversation():
    """创建新对话"""
    try:
        data = request.json
        if not data or 'content' not in data:
            return jsonify({'status': 'error', 'message': '缺少对话内容'}), 400

        content = data['content']
        # 从请求中获取用户标识，如果没有则使用默认值
        user_id = data.get('user_id', 'default_user')
        
        try:
            # 解析日期
            line0 = content.split('\n', 1)[0].strip()
            script_time = datetime.strptime(line0, "%Y-%m-%d")
        except (ValueError, IndexError):
            script_time = datetime.now()

        # 生成摘要
        summary = llm_utils.gen_conversation_summary(content)
        conversation = Conversation(content=content, summary=summary, created_at=script_time)
        db.session.add(conversation)
        db.session.commit()

        # 异步更新记忆
        def update_memory_background(user_id):
            try:
                latest_memory = Memory.query.order_by(Memory.updated_at.desc()).first()
                if latest_memory:
                    memory_topics = json.loads(latest_memory.content)['topics']
                    latest_day_topics = json.loads(summary)['topics']
                    new_memory = llm_utils.gen_memory(memory_topics, latest_day_topics)
                else:
                    new_memory = json.dumps({'topics': json.loads(summary)['topics']}, ensure_ascii=False)

                memory = Memory(content=new_memory)
                db.session.add(memory)
                db.session.commit()
                
                # 使用应用中已初始化的memory_bank实例，避免重复初始化导致的存储冲突
                global memory_bank
                if memory_bank is None:
                    # 如果memory_bank尚未初始化，则从current_app获取
                    memory_bank = current_app.memory_bank if hasattr(current_app, 'memory_bank') else MemoryBank(user=user_id)
                memory_bank.add_memory('\n\n'.join(x['information'] for x in json.loads(summary)['topics']), created_date=script_time.isoformat())
            except Exception as e:
                print(f'后台更新记忆失败: {str(e)}')

        # 启动后台线程（注意：在生产环境中应使用更可靠的异步任务处理方式）
        import threading
        memory_thread = threading.Thread(target=update_memory_background, args=(user_id,))
        memory_thread.daemon = True
        memory_thread.start()

        return jsonify({
            'status': 'success',
            'message': '对话已成功处理',
            'data': {
                'id': conversation.id,
                'summary': summary
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 问答相关API
@api_bp.route('/qa', methods=['POST'])
def ask_question():
    """提交问题并获取回答"""
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'status': 'error', 'message': '缺少问题内容'}), 400

        question = data['question']
        model = data.get('model', 'default')
        # 从请求中获取用户标识，如果没有则使用默认值
        user_id = data.get('user_id', 'default_user')

        memory_topics = {}
        latest_memory = Memory.query.order_by(Memory.updated_at.desc()).first()
        if latest_memory:
            memory_topics = json.loads(latest_memory.content)['topics']

        # 使用应用中已初始化的memory_bank实例，避免重复初始化导致的存储冲突
        global memory_bank
        if memory_bank is None:
            # 如果memory_bank尚未初始化，则从current_app获取
            memory_bank = current_app.memory_bank if hasattr(current_app, 'memory_bank') else MemoryBank(user=user_id)
        content_list = memory_bank.extract_qa_memorries(question)
        answer = llm_utils.get_qa_answer_soft(question, memory_topics, content_list, model_name=model)

        return jsonify({
            'status': 'success',
            'data': {
                'question': question,
                'answer': answer,
                'context': content_list
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# WebSocket事件处理器
def register_socketio_handlers(app_socketio):
    """注册WebSocket事件处理器"""
    global socketio
    socketio = app_socketio
    
    @socketio.on('stream_qa')
    def handle_stream_qa(data):
        """处理流式问答请求"""
        try:
            if not data or 'question' not in data:
                emit('stream_qa_response', {'status': 'error', 'message': '缺少问题内容'})
                return

            question = data['question']
            model = data.get('model', 'default')
            user_id = data.get('user_id', 'default_user')
            session_id = data.get('session_id', 'default_session')

            # 获取内存主题
            memory_topics = {}
            latest_memory = Memory.query.order_by(Memory.updated_at.desc()).first()
            if latest_memory:
                memory_topics = json.loads(latest_memory.content)['topics']

            # 使用应用中已初始化的memory_bank实例
            global memory_bank
            if memory_bank is None:
                memory_bank = current_app.memory_bank if hasattr(current_app, 'memory_bank') else MemoryBank(user=user_id)
            content_list = memory_bank.extract_qa_memorries(question)

            # 发送上下文信息
            emit('stream_qa_response', {
                'status': 'context',
                'session_id': session_id,
                'context': content_list
            })

            # 流式获取并发送回答
            full_answer = ""
            for chunk in llm_utils.stream_qa_answer_soft(question, memory_topics, content_list, model_name=model):
                full_answer += chunk
                emit('stream_qa_response', {
                    'status': 'chunk',
                    'session_id': session_id,
                    'chunk': chunk
                })
                socketio.sleep(0)  # 让出控制权，避免阻塞

            # 发送完成信号
            emit('stream_qa_response', {
                'status': 'complete',
                'session_id': session_id,
                'full_answer': full_answer
            })

        except Exception as e:
            error_message = str(e)
            session_id = data.get('session_id', 'default_session') if data else 'default_session'
            emit('stream_qa_response', {
                'status': 'error',
                'session_id': session_id,
                'message': error_message
            })
            current_app.logger.error(f'Stream QA error: {error_message}', exc_info=True)

# 记忆相关API
@api_bp.route('/memories', methods=['GET'])
def get_memories():
    """获取所有记忆，支持分页"""
    try:
        # 从请求参数中获取用户标识，如果没有则使用默认值
        user_id = request.args.get('user_id', 'default_user')
        
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        offset = (page - 1) * per_page
        
        # 为特定用户创建memory_bank实例
        user_memory_bank = MemoryBank(user=user_id)
        all_memories = user_memory_bank.get_all(limit=per_page, offset=offset)
        
        formatted_memories = []
        for mem in sorted(all_memories['results'], key=lambda x: x['created_at'], reverse=True):
            formatted_memories.append({
                'content': mem['memory'],
                'created_at': mem.get('created_at', 'Unknown date')
            })
        
        # 返回包含分页信息的响应
        pagination = all_memories.get('pagination', {})
        return jsonify({
            'status': 'success',
            'data': formatted_memories,
            'pagination': {
                'total': pagination.get('total', 0),
                'page': page,
                'per_page': per_page,
                'has_more': pagination.get('has_more', False)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# @api_bp.route('/memories/latest', methods=['GET'])
# def get_latest_memory():
#     """获取最新记忆"""
#     try:
#         latest_memory = Memory.query.order_by(Memory.updated_at.desc()).first()
#         if not latest_memory:
#             return jsonify({'status': 'success', 'data': None})
#         return jsonify({
#             'status': 'success',
#             'data': {
#                 'id': latest_memory.id,
#                 'content': latest_memory.content,
#                 'updated_at': latest_memory.updated_at.isoformat()
#             }
#         })
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# 事件相关API
@api_bp.route('/get_coming_up', methods=['GET'])
def get_coming_up():
    """获取即将到来的安排"""
    try:
        return jsonify({
            'status': 'success',
            'data': 'Meeting up with Jianjian'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/get_attention', methods=['GET'])
def get_attention():
    """获取需要注意的事项"""
    try:
        return jsonify({
            'status': 'success',
            'data': ['Delta Brain', '膝盖治疗', '头脑风暴']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# @api_bp.route('/events', methods=['GET'])
@api_bp.route('/get_current', methods=['GET'])
def get_events():
    """获取所有事件"""
    try:
        memory = get_latest_memory()
        event_list = []
        if memory:
            memory_data = json.loads(memory.content)
            for topic in memory_data.get('topics', []):
                event = {
                    "date": datetime.now(),
                    "title": topic.get('title', ''),
                    "details": topic.get('summary', '')
                }
                event_list.append(event)
        return jsonify({'status': 'success', 'data': event_list})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/get_daily', methods=['GET'])
def get_daily():
    """获取每日对话数据"""
    try:
        # 查询所有对话（API中我们使用api_user）
        conversations = Conversation.query.order_by(Conversation.created_at.desc()).all()
        
        # 按日期分组处理
        daily_conversations = {}
        for conv in conversations:
            date_key = conv.created_at.strftime('%Y-%m-%d')
            if date_key not in daily_conversations:
                daily_conversations[date_key] = []
            
            # 解析摘要数据
            try:
                conv_summary = json.loads(conv.summary)
                conv_topics = conv_summary.get('topics', [])
                conv_actions = conv_summary.get('action_items', [])
            except json.JSONDecodeError:
                current_app.logger.warning(f'对话摘要解析失败: {conv.id}')
                conv_topics = []
                conv_actions = []
            
            # 构建事件列表
            event_list = []
            for topic in conv_topics:
                event = {
                    'date': conv.created_at.isoformat(),
                    'title': topic.get('title', ''),
                    'details': topic.get('summary', '')
                }
                event_list.append(event)
            
            # 构建行动列表
            action_list = []
            hash_hex = hashlib.md5(date_key.encode()).hexdigest()
            cutoff = 3 if int(hash_hex, 16) % 2 == 0 else 2
            for action in conv_actions[:cutoff]:
                action_item = {
                    'owner': action.get('owner', ''),
                    'task': action.get('task', '')
                }
                action_list.append(action_item)
            
            # 添加到每日对话数据中
            daily_conversations[date_key].append({
                'events': event_list,
                'actions': action_list,
                'conversation': {
                    'id': conv.id,
                    'content': conv.content,
                    'summary': conv.summary,
                    'created_at': conv.created_at.isoformat()
                }
            })
        
        # 按日期降序排序
        sorted_dates = sorted(daily_conversations.keys(), reverse=True)
        
        # 构建返回数据
        result = {
            'daily_conversations': [{
                'date': date_key,
                'events': daily_conversations[date_key][0]['events'],
                'actions': daily_conversations[date_key][0]['actions'],
                'conversation': daily_conversations[date_key][0]['conversation'],
                } for date_key in sorted_dates],
            'sorted_dates': sorted_dates
        }
        
        return jsonify({'status': 'success', 'data': result})
    except Exception as e:
        current_app.logger.error(f'获取每日对话数据失败: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/events/latest', methods=['GET'])
def get_latest_events():
    """获取最新事件"""
    try:
        memory = Memory.query.order_by(Memory.updated_at.desc()).first()
        event_list = []
        if memory:
            memory_data = json.loads(memory.content)
            for topic in memory_data.get('topics', []):
                event = {
                    'date': datetime.now().isoformat(),
                    'title': topic.get('title', ''),
                    'details': topic.get('summary', '')
                }
                event_list.append(event)
        return jsonify({'status': 'success', 'data': event_list})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 音频处理API
@api_bp.route('/audio/transcribe', methods=['POST'])
def transcribe_audio():
    """上传音频并获取转录文本"""
    try:
        if 'audio_file' not in request.files:
            return jsonify({'status': 'error', 'message': '未找到音频文件'}), 400

        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': '未选择音频文件'}), 400

        if audio_file and allowed_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            upload_folder = os.path.join(current_app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            audio_file.save(file_path)
            current_app.logger.info(f"Received audio file for upload: {filename}")

            # 获取文件格式
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp3'
            supported_formats = {'mp3', 'wav', 'ogg'}
            if file_ext not in supported_formats:
                current_app.logger.warning(f"Unsupported audio format: {file_ext}")
                return jsonify({'status': 'error', 'message': f'不支持的音频格式: {file_ext}，仅支持mp3、wav、ogg'}), 400

            # 获取是否启用说话人区分的参数
            enable_diarization = request.form.get('enable_diarization', 'true').lower() == 'true'
            
            # 确保ASR服务已初始化
            if asr_service is None:
                return jsonify({'status': 'error', 'message': 'ASR服务未初始化'}), 500

            # 调用ASR服务转换音频为文本
            current_app.logger.info(f"Starting transcription for {filename}")
            try:
                transcription = asr_service.transcribe_audio(file_path, format=file_ext, enable_diarization=enable_diarization)
                current_app.logger.info(f"Transcription successful for {filename}")
            except Exception as e:
                current_app.logger.error(f'Audio processing failed: {str(e)}', exc_info=True)
                return jsonify({'status': 'error', 'message': f'音频处理失败: {str(e)}'}), 500

            # 格式化转录结果
            if isinstance(transcription, dict) and 'dialogue_lines' in transcription:
                text_result = '\n'.join(transcription['dialogue_lines'])
            else:
                text_result = str(transcription)

            # 返回转录结果
            return jsonify({
                'status': 'success',
                'message': '音频转录成功',
                'data': {
                    'filename': filename,
                    'transcription': text_result
                }
            })
        else:
            return jsonify({'status': 'error', 'message': '不支持的音频格式，仅支持mp3、wav、ogg'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# 在app.py中注册蓝图和WebSocket处理器时使用以下代码
# from api import api_bp, register_socketio_handlers
# app.register_blueprint(api_bp, url_prefix='/api')
# register_socketio_handlers(socketio)
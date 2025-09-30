import os
import json
import time
import threading
import uuid
import signal
import sys
import logging
import hashlib
from logging.handlers import RotatingFileHandler
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, session, g, copy_current_request_context
from flask_wtf import FlaskForm
from wtforms import TextAreaField, FileField, SubmitField, StringField, PasswordField, SelectField
from wtforms.validators import DataRequired, Length
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from utils.memory_utils import MemoryBank

# 本地导入
from extensions import db
from models import Conversation, Event, Action, Memory, User
from services.db_service import (
    add_conversation, get_all_conversations, clear_conversations,
    clear_all_data, get_latest_memory, add_memory
)
from services.llm_service import LLMService

from utils.llm_utils import LLMUtils

# 配置加载
load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_key_for_testing')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///conversations.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
    QA_MODE = 'MEM0'
    GLOBAL_LANG = os.getenv('GLOBAL_LANG', 'zh')

# 应用初始化
app = Flask(__name__)
app.config.from_object(Config)

# 导入并注册API蓝图
from api import api_bp
app.register_blueprint(api_bp, url_prefix='/api')

# 配置日志
if not app.debug:
    handler = RotatingFileHandler('gunicorn.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

me = "user"
db.init_app(app)
llm_service = LLMService()
llm_utils = LLMUtils(llm_service)
if Config.GLOBAL_LANG == 'en':
    from services.whisper_asr_service import WhisperASRService
    asr_service = WhisperASRService()
    asr_service.switch_model('en')  # 强制使用英文
elif Config.GLOBAL_LANG == 'zh':
    from services.funasr_service import FunASRService
    asr_service = FunASRService()
else:
    raise ValueError("Language not supported")
memory_bank = MemoryBank(user=me)

# 表单定义
class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired(), Length(min=4, max=80)])
    password = PasswordField('密码', validators=[DataRequired()])
    submit = SubmitField('登录')

class UploadForm(FlaskForm):
    conversation_text = TextAreaField('对话文本', validators=[DataRequired()])
    submit = SubmitField('提交处理')

class QAForm(FlaskForm):
    question = TextAreaField('问题', validators=[DataRequired()])
    model = SelectField('模型选择', choices=[(model, model) for model in llm_service.model_configs.keys()], validators=[DataRequired()])
    submit = SubmitField('获取回答')

class ConversationAnalysisForm(FlaskForm):
    conversation_text = TextAreaField('对话内容', validators=[DataRequired()])
    model = SelectField('模型选择', choices=[(model, model) for model in llm_service.model_configs.keys()], validators=[DataRequired()])
    system_prompt = TextAreaField('System Prompt', validators=[])
    submit = SubmitField('分析对话')

class AudioUploadForm(FlaskForm):
    audio_file = FileField('音频文件', validators=[DataRequired()])
    submit = SubmitField('上传并处理')

# 工具函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# 创建临时文件目录
TEMP_DIR = os.path.join(app.root_path, 'tmp')
os.makedirs(TEMP_DIR, exist_ok=True)

# 登录保护装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 请求钩子
@app.before_request
def load_current_user():
    if 'user_id' in session:
        g.current_user = User.query.get(session['user_id'])
    else:
        g.current_user = None

# 路由函数
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user = User.query.filter_by(username=form.username.data).first()
            if user and user.check_password(form.password.data):
                session['user_id'] = user.id
                flash('登录成功！', 'success')
                return redirect(url_for('index'))
            flash('用户名或密码不正确', 'danger')
        except Exception as e:
            app.logger.error(f'登录处理失败: {str(e)}')
            flash('登录过程中发生错误，请稍后重试', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    # 获取预填充文本和临时文件信息
    prefilled_text = session.get('prefilled_text', '')
    transcription_temp_file = session.get('transcription_temp_file', '')
    is_from_transcription = session.get('is_from_transcription', False)
    
    # 创建表单并设置预填充文本
    form = UploadForm(conversation_text=prefilled_text)
    
    # 表单提交处理
    if form.validate_on_submit():
        content = form.conversation_text.data
        try:
            # 解析日期
            line0 = content.split('\n', 1)[0].strip()
            script_time = datetime.strptime(line0, "%Y-%m-%d")
        except (ValueError, IndexError):
            script_time = datetime.now()
            app.logger.warning('无法解析日期，使用当前时间')
        
        # 表单已提交，清理session中的相关数据
        if 'prefilled_text' in session:
            session.pop('prefilled_text', None)
        if 'transcription_temp_file' in session:
            session.pop('transcription_temp_file', None)
        if 'is_from_transcription' in session:
            session.pop('is_from_transcription', None)

        try:
            # 生成摘要
            summary = llm_utils.gen_conversation_summary(content)
            add_conversation(g.current_user.id, content, summary, script_time)
            flash('对话已成功处理', 'success')

            # 启动后台线程更新长期记忆
            @copy_current_request_context
            def update_memory_background(user_id, username):
                try:
                    with app.app_context():
                        user = User.query.get(user_id)
                        if user:
                            user.memory_updating = True
                            db.session.commit()

                        latest_memory = get_latest_memory(user_id)
                        if latest_memory:
                            memory_topics = json.loads(latest_memory.content)['topics']
                            latest_day_topics = json.loads(summary)['topics']

                            new_memory = llm_utils.gen_memory(memory_topics, latest_day_topics)
                        else:
                            new_memory = json.dumps({'topics': json.loads(summary)['topics']}, ensure_ascii=False)

                        add_memory(user_id, new_memory)
                        # 为当前用户创建memory_bank实例
                        memory_bank.add_memory('\n\n'.join(x['information'] for x in json.loads(summary)['topics']), created_date=script_time.isoformat())
                        app.logger.info('记忆更新成功')

                        # 更新用户状态
                        with app.app_context():
                            user = User.query.get(user_id)
                            if user:
                                user.memory_updating = False
                                db.session.commit()
                except Exception as e:
                    app.logger.error(f'后台更新记忆失败: {str(e)}')
                    # 确保清除更新状态
                    with app.app_context():
                        user = User.query.get(user_id)
                        if user:
                            user.memory_updating = False
                            db.session.commit()

            # 启动后台线程，传入用户ID和用户名
            memory_thread = threading.Thread(target=update_memory_background, args=(g.current_user.id, g.current_user.username))
            memory_thread.start()

            return redirect(url_for('daily'))
        except Exception as e:
            app.logger.error(f'处理对话失败: {str(e)}')
            flash(f'处理对话时发生错误: {str(e)}', 'danger')

    # 只有当用户从转写页面跳转过来时，才传递临时文件信息到模板
    # 否则传递None，避免在直接访问首页时尝试加载不存在的转写文件
    template_transcription_temp_file = transcription_temp_file if is_from_transcription else None
    
    return render_template('index.html', form=form, transcription_temp_file=template_transcription_temp_file)


@app.route('/current-event')
@login_required
def current_event():
    try:
        memory = get_latest_memory(g.current_user.id)
        event_list = []
        if memory:
            memory_data = json.loads(memory.content)
            for topic in memory_data.get('topics', []):
                event = Event(
                    date=datetime.now(),
                    title=topic.get('title', ''),
                    details=topic.get('summary', '')
                )
                event_list.append(event)
    except Exception as e:
        app.logger.error(f'获取当前事件失败: {str(e)}')
        flash('获取当前事件时发生错误', 'danger')
        event_list = []

    # 检查内存更新状态
    memory_updating = g.current_user.memory_updating if g.current_user else False

    return render_template('current_event.html', event_list=event_list, memory_updating=memory_updating)

@app.route('/daily')
@login_required
def daily():
    try:
        # 查询所有对话并按日期分组
        conversations = get_all_conversations(g.current_user.id)
        
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
                app.logger.warning(f'对话摘要解析失败: {conv.id}')
                conv_topics = []
                conv_actions = []
            
            event_list = []
            for topic in conv_topics:
                event = Event(
                    date=conv.created_at,
                    title=topic.get('title', ''),
                    details=topic.get('summary', '')
                )
                event_list.append(event)
            
            action_list = []
            hash_hex = hashlib.md5(date_key.encode()).hexdigest()
            cutoff = 3 if int(hash_hex, 16) % 2 == 0 else 2
            for action in conv_actions[:cutoff]:
                action_item = Action(
                    owner=action.get('owner', ''),
                    task=action.get('task', '')
                )
                action_list.append(action_item)
            
            daily_conversations[date_key].append((event_list, action_list, conv))
        
        # 按日期降序排序
        sorted_dates = sorted(daily_conversations.keys(), reverse=True)
        return render_template('daily.html', daily_conversations=daily_conversations, sorted_dates=sorted_dates)
    
    except Exception as e:
        app.logger.error(f'获取每日对话失败: {str(e)}')
        flash('获取每日对话时发生错误', 'danger')
        return render_template('daily.html', daily_conversations={}, sorted_dates=[])

@app.route('/clear-database', methods=['POST'])
@login_required
def clear_database():
    # 验证确认参数
    confirm_admin = request.form.get('confirmAdmin')
    if confirm_admin == 'admin' and g.current_user.username == 'admin':
        # 管理员清空所有数据
        try:
            deletion_counts = clear_all_data()  # 不传user_id表示清空所有数据
            memory_bank.delete_memory()
            print('All data cleared')
            return jsonify({
                'status': 'success', 
                'message': '所有数据已成功清空',
                'deleted_records': deletion_counts
            })
        except Exception as e:
            print(f'清空所有数据失败: {str(e)}')
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    elif confirm_admin == 'confirm':
        # 普通用户清空自己的数据
        try:
            deletion_counts = clear_all_data(g.current_user.id)
            # memory_bank.delete_memory()  # 注意：这里保留了原有的memory_bank操作
            print('User data cleared')
            return jsonify({
                'status': 'success', 
                'message': '您的数据已成功清空',
                'deleted_records': deletion_counts
            })
        except Exception as e:
            print(f'清空用户数据失败: {str(e)}')
            db.session.rollback()
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        return jsonify({'status': 'error', 'message': '请输入正确的确认信息'}), 400

# 清空数据库确认页面路由
@app.route('/clear-database-confirm', methods=['GET'])
@login_required
def clear_database_confirm():
    # 检查是否为管理员
    if g.current_user.username != 'admin':
        flash('权限不足，只有管理员可以访问此页面', 'danger')
        return redirect(url_for('index'))
    from flask_wtf import FlaskForm
    form = FlaskForm()  # 创建空表单用于CSRF令牌
    return render_template('clear_database.html', form=form)

# 音频处理任务状态字典
processing_tasks = {}

def process_audio_file(file_path, file_ext, task_id):
    """后台线程处理音频文件"""
    try:
        with app.app_context():
            transcription = asr_service.transcribe_audio(file_path, format=file_ext)
            
            # 确保dialogue_lines存在
            if 'dialogue_lines' not in transcription:
                # 使用文本替代
                text_result = transcription.get('text', 'No dialogue lines found')
            else:
                text_result = '\n'.join(transcription['dialogue_lines'])
                
            prefilled_text = f'{datetime.today().date()}\n' + text_result
            
            # 统一使用临时文件存储转写结果，不再根据长度区分
            # 生成唯一的临时文件名
            temp_filename = f'transcription_{task_id}.txt'
            temp_filepath = os.path.join(TEMP_DIR, temp_filename)
            
            # 将转写结果写入临时文件
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                f.write(prefilled_text)
            
            # 存储临时文件名而不是完整文本
            processing_tasks[task_id] = {
                'status': 'completed',
                'use_temp_file': True,
                'temp_filename': temp_filename
            }
    except Exception as e:
        app.logger.error(f'Audio processing failed for task {task_id}: {str(e)}', exc_info=True)
        processing_tasks[task_id] = {
            'status': 'error',
            'message': str(e)
        }

@app.route('/audio-upload', methods=['GET', 'POST'])
@login_required
def audio_upload():
    form = AudioUploadForm()
    if form.validate_on_submit():
        audio_file = form.audio_file.data
        if audio_file:
            # 保存上传的音频文件
            filename = secure_filename(audio_file.filename)
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            audio_file.save(file_path)
            
            # 获取文件格式
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp3'
            supported_formats = {'mp3', 'wav', 'ogg'}
            if file_ext not in supported_formats:
                flash(f'不支持的音频格式: {file_ext}，仅支持mp3、wav、ogg', 'danger')
                return redirect(url_for('audio_upload'))
            
            # 生成任务ID并启动后台线程处理
            task_id = str(uuid.uuid4())
            processing_tasks[task_id] = {'status': 'processing'}
            
            # 启动后台线程处理音频
            thread = threading.Thread(target=process_audio_file, args=(file_path, file_ext, task_id))
            thread.daemon = True
            thread.start()
            
            # 重定向到处理状态页面
            return redirect(url_for('audio_processing_status', task_id=task_id))
    return render_template('audio_upload.html', form=form)

@app.route('/audio-processing-status/<task_id>')
@login_required
def audio_processing_status(task_id):
    if task_id not in processing_tasks:
        flash('无效的任务ID', 'danger')
        return redirect(url_for('audio_upload'))
    
    task_status = processing_tasks[task_id]
    if task_status['status'] == 'completed':
        # 处理完成，设置预填充文本信息并重定向到主页
        if 'use_temp_file' in task_status and task_status['use_temp_file'] and 'temp_filename' in task_status:
            # 统一使用临时文件方式处理所有转写结果
            session['transcription_temp_file'] = task_status['temp_filename']
            # 存储一个简短的提示信息
            session['prefilled_text'] = "[转写内容正在加载中...]\n"
            # 设置标志，表示用户从转写页面跳转过来
            session['is_from_transcription'] = True
            flash('音频上传成功并已转换为文本', 'success')
        else:
            flash('音频处理完成，但无法获取文本结果', 'danger')
        
        # 清理任务状态
        del processing_tasks[task_id]
        return redirect(url_for('index'))
    elif task_status['status'] == 'error':
        # 处理出错
        error_message = task_status.get('message', '未知错误')
        flash(f'音频处理失败: {error_message}', 'danger')
        # 清理任务状态
        del processing_tasks[task_id]
        return redirect(url_for('audio_upload'))
    
    # 处理中，显示状态页面
    return render_template('audio_processing_status.html', task_id=task_id)

@app.route('/api/get-transcription-file/<filename>')
@login_required
def get_transcription_file(filename):
    """获取临时存储的转写文本文件"""
    try:
        # 确保请求的文件存在且在临时目录中
        temp_filepath = os.path.join(TEMP_DIR, filename)
        
        # 验证文件路径，防止目录遍历攻击
        if not os.path.abspath(temp_filepath).startswith(os.path.abspath(TEMP_DIR)):
            return jsonify({'status': 'error', 'message': '无效的文件路径'}), 403
        
        # 检查文件是否存在
        if not os.path.exists(temp_filepath):
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404
        
        # 读取文件内容
        with open(temp_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 读取完成后删除临时文件，避免占用空间
        os.remove(temp_filepath)
        
        return jsonify({'status': 'success', 'content': content})
    except Exception as e:
        app.logger.error(f'读取转写文件失败: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/conversation_analysis', methods=['GET', 'POST'])
@login_required
def conversation_analysis():
    form = ConversationAnalysisForm()
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if form.validate_on_submit():
            conversation_text = form.conversation_text.data
            model = form.model.data
            system_prompt = form.system_prompt.data.strip() if hasattr(form, 'system_prompt') and form.system_prompt.data else None
            try:
                # 调用LLM分析对话内容
                analysis_result = llm_utils.analyze_conversation(conversation_text, model_name=model, system_prompt=system_prompt)
                return jsonify({'status': 'success', 'analysis_result': analysis_result})
            except Exception as e:
                app.logger.error(f'分析对话内容失败: {str(e)}')
                return jsonify({'status': 'error', 'message': f'分析对话内容失败: {str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': '表单验证失败', 'errors': form.errors})
    return render_template('conversation_analysis.html', form=form)

@app.route('/conversation_analysis_stream', methods=['POST'])
@login_required
def conversation_analysis_stream():
    """流式处理对话分析请求"""
    form = ConversationAnalysisForm()
    if form.validate_on_submit() or 'request_id' in request.form:
        # 获取基本参数
        conversation_text = form.conversation_text.data if form.conversation_text.data else request.form.get('conversation_text')
        model = form.model.data if form.model.data else request.form.get('model')
        system_prompt = form.system_prompt.data.strip() if hasattr(form, 'system_prompt') and form.system_prompt.data else request.form.get('system_prompt')
        
        # 获取恢复参数
        request_id = request.form.get('request_id')
        content_length = int(request.form.get('content_length', 0))
        
        def generate():
            try:
                # 使用流式分析对话内容，支持从特定位置恢复
                for chunk_idx, chunk in enumerate(llm_utils.stream_analyze_conversation(
                        conversation_text, 
                        model_name=model, 
                        request_id=request_id,
                        content_length=content_length,
                        system_prompt=system_prompt
                )):
                    # 为每个chunk添加唯一ID，便于前端去重
                    yield f'data: {json.dumps({"chunk": chunk, "chunk_id": chunk_idx})}\n\n'
                # 发送完成信号
                yield 'data: {"complete": true}\n\n'
            except Exception as e:
                app.logger.error(f'流式分析对话内容失败: {str(e)}')
                yield f'data: {json.dumps({"error": str(e)})}\n\n'
        
        # 返回流式响应
        return app.response_class(generate(), mimetype='text/event-stream')
    else:
        # 如果表单验证失败，返回错误信息
        return jsonify({'status': 'error', 'message': '表单验证失败', 'errors': form.errors}), 400


@app.route('/qa', methods=['GET', 'POST'])
@login_required
def qa():
    form = QAForm()
    answer = None
    content_list = []
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        if form.validate_on_submit():
            question = form.question.data
            model = form.model.data
            try:
                memory_topics = {}
                latest_memory = get_latest_memory(g.current_user.id)
                if latest_memory:
                    memory_topics = json.loads(latest_memory.content)['topics']
                if Config.QA_MODE == 'RAW':
                    conversations = get_all_conversations(g.current_user.id)
                    for conv in conversations:
                        content = conv.content
                        content_list.append(content)
                    answer = llm_utils.get_qa_answer(question, memory_topics, content_list, model_name=model)
                elif Config.QA_MODE == 'MEM0':
                    content_list = memory_bank.extract_qa_memorries(question)
                    answer = llm_utils.get_qa_answer_soft(question, memory_topics, content_list, model_name=model)
                else:
                    raise ValueError("QA_MODE 配置错误")
                return jsonify({'status': 'success', 'content_list': content_list, 'answer': answer})
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'获取回答失败: {str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': '表单验证失败', 'errors': form.errors})
    # elif form.validate_on_submit():
    #     question = form.question.data
    #     model = form.model.data
    #     try:
    #         memory_topics = {}
    #         latest_memory = get_latest_memory()
    #         if latest_memory:
    #             memory_topics = json.loads(latest_memory.content)['topics']
    #         conversations = get_all_conversations()
    #         content_list = []
    #         for conv in conversations:
    #             content = conv.content
    #             content_list.append(content)
    #         answer = llm_utils.get_qa_answer(question, memory_topics, content_list, model_name=model)
    #     except Exception as e:
    #         flash(f'获取回答失败: {str(e)}', 'danger')
    return render_template('qa.html', form=form, answer=answer)

@app.route('/memories')
@login_required
def memories():
    try:
        # 获取当前页的记忆数据（限制为50条，避免一次性加载过多数据）
        page = request.args.get('page', 1, type=int)
        per_page = 50  # 每页显示的记忆数量
        offset = (page - 1) * per_page
        
        all_memories = memory_bank.get_all(limit=per_page, offset=offset)
        
        # 格式化记忆数据以便模板使用
        formatted_memories = []
        for mem in sorted(all_memories['results'], key=lambda x: x['created_at'], reverse=True):
            formatted_memories.append({
                'content': mem['memory'],
                'created_at': mem.get('created_at', 'Unknown date')
            })
    except Exception as e:
        app.logger.error(f'获取记忆失败: {str(e)}')
        flash('获取记忆时发生错误', 'danger')
        formatted_memories = []
        all_memories = {'pagination': {'has_more': False}}
    # 检查内存更新状态
    memory_updating = g.current_user.memory_updating if g.current_user else False
    return render_template('memories.html', memories=formatted_memories, memory_updating=memory_updating, 
                           has_more=all_memories.get('pagination', {}).get('has_more', False), 
                           current_page=page)

# 创建数据库表
with app.app_context():
    db.create_all()

    # 创建默认管理员用户（如果不存在）
    if not User.query.first():
        # admin
        admin_username = os.getenv('ADMIN_USERNAME', 'admin')
        admin_password = os.getenv('ADMIN_PASSWORD', 'password@admin#2025?')  # 默认密码，生产环境应修改
        admin_user = User(username=admin_username)
        admin_user.set_password(admin_password)
        db.session.add(admin_user)
        # demo_user
        user_username = 'user'
        user_password = 'password@user#2025!'
        user_user = User(username=user_username)
        user_user.set_password(user_password)
        db.session.add(user_user)
        
        db.session.commit()
        print(f"已创建默认管理员用户: {admin_username}, 密码: {admin_password}")

@app.teardown_appcontext
def shutdown_memory_bank(exception=None):
    global memory_bank
    if 'memory_bank' in globals():
        memory_bank.close()

def signal_handler(sig, frame):
    global memory_bank
    if 'memory_bank' in globals():
        memory_bank.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    app.run(debug=True)
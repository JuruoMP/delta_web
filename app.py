import os
import json
import time
import threading
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

class AudioUploadForm(FlaskForm):
    audio_file = FileField('音频文件', validators=[DataRequired()])
    submit = SubmitField('上传并处理')

# 工具函数
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

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
@login_required
def index():
    prefilled_text = session.pop('prefilled_text', '')
    form = UploadForm(conversation_text=prefilled_text)
    if form.validate_on_submit():
        content = form.conversation_text.data
        try:
            # 解析日期
            line0 = content.split('\n', 1)[0].strip()
            script_time = datetime.strptime(line0, "%Y-%m-%d")
        except (ValueError, IndexError):
            script_time = datetime.now()
            app.logger.warning('无法解析日期，使用当前时间')

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
                        user_memory_bank = MemoryBank(user=username)
                        user_memory_bank.add_memory('\n\n'.join(x['information'] for x in json.loads(summary)['topics']), created_date=script_time.isoformat())
                        app.logger.info('记忆更新成功')
                        # app.logger.info(f'最新的记忆：{user_memory_bank.get_all()}')

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

    return render_template('index.html', form=form)


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

@app.route('/audio-upload', methods=['GET', 'POST'])
@login_required
def audio_upload():
    form = AudioUploadForm()
    if form.validate_on_submit():
        audio_file = form.audio_file.data
        if audio_file:
            # 保存上传的音频文件
            filename = secure_filename(audio_file.filename)
            app.logger.info(f"Received audio file for upload: {filename}")
            upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            audio_file.save(file_path)
            
            # 生成可访问的URL
            file_url = url_for('static', filename=f'uploads/{filename}', _external=True)
            
            # 获取文件格式
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp3'
            supported_formats = {'mp3', 'wav', 'ogg'}
            if file_ext not in supported_formats:
                app.logger.warning(f"Unsupported audio format: {file_ext}")
                flash(f'不支持的音频格式: {file_ext}，仅支持mp3、wav、ogg', 'danger')
                return redirect(url_for('audio_upload'))
            
            # 调用ASR服务转换音频为文本
            try:
                app.logger.info(f"Starting transcription for {filename}")
                transcription = asr_service.transcribe_audio(file_path, format=file_ext)
                # print(f'{transcription=}')
                app.logger.info(f"Transcription successful for {filename}")
                
                # 提取转录文本并格式化为对话形式
                # formatted_text = ""
                # if isinstance(transcription, dict) and 'result' in transcription and 'segments' in transcription['result']:
                #     formatted_text += datetime.now().strftime("%Y-%m-%d") + "\n"
                #     for segment in transcription['result']['segments']:
                #         speaker = segment.get('speaker', 'Unknown')
                #         text = segment.get('text', '').strip()
                #         if text:
                #             formatted_text += f"{speaker}: {text}\n"
                #     text_result = formatted_text.strip()
                # else:
                #     text_result = str(transcription)
                text_result = '\n'.join(transcription['dialogue_lines'])
                
                # 将转录文本作为对话内容处理
                flash('音频上传成功并已转换为文本', 'success')
                session['prefilled_text'] = f'{datetime.today().date()}\n' + text_result
                return redirect(url_for('index'))
            except Exception as e:
                app.logger.error(f'Audio processing failed: {str(e)}', exc_info=True)
                flash(f'音频处理失败: {str(e)}', 'danger')
                return redirect(url_for('audio_upload'))
    return render_template('audio_upload.html', form=form)

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
        # 为当前登录用户创建临时的MemoryBank实例，确保获取正确的用户记忆
        user_memory_bank = MemoryBank(user=g.current_user.username)
        # 获取当前页的记忆数据（限制为50条，避免一次性加载过多数据）
        page = request.args.get('page', 1, type=int)
        per_page = 50  # 每页显示的记忆数量
        offset = (page - 1) * per_page
        
        all_memories = user_memory_bank.get_all(limit=per_page, offset=offset)
        
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
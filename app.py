import os
import json
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from extensions import db
from flask_wtf import FlaskForm
from wtforms import TextAreaField, FileField, SubmitField, StringField, PasswordField
from wtforms.validators import DataRequired, Length
from datetime import datetime
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from services.llm_service import llm_service
from models import Conversation, Event, Action, Memory, User
from services.db_service import add_conversation, get_all_conversations, clear_conversations, clear_all_data, get_latest_memory, add_memory

from fake_data import fake_summary, fake_memory

# 加载环境变量
load_dotenv()

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///conversations.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# 登录表单
class LoginForm(FlaskForm):
    username = StringField('用户名', validators=[DataRequired(), Length(min=4, max=80)])
    password = PasswordField('密码', validators=[DataRequired()])
    submit = SubmitField('登录')

# 上传表单
class UploadForm(FlaskForm):
    conversation_text = TextAreaField('对话文本', validators=[DataRequired()])
    submit = SubmitField('提交处理')

class QAForm(FlaskForm):
    question = TextAreaField('问题', validators=[DataRequired()])
    submit = SubmitField('获取回答')

class AudioUploadForm(FlaskForm):
    audio_file = FileField('音频文件', validators=[DataRequired()])
    submit = SubmitField('上传并处理')

from functools import wraps
from flask import session, g

# 登录保护装饰器
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# 加载当前用户
@app.before_request
def load_current_user():
    if 'user_id' in session:
        g.current_user = User.query.get(session['user_id'])
    else:
        g.current_user = None

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            session['user_id'] = user.id
            flash('登录成功！', 'success')
            return redirect(url_for('index'))
        flash('用户名或密码不正确', 'danger')
    return render_template('login.html', form=form)

# 登出路由
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('已成功登出', 'success')
    return redirect(url_for('login'))

# 路由
@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prefilled_text = request.args.get('prefilled_text', '')
    form = UploadForm(conversation_text=prefilled_text)
    if form.validate_on_submit():
        content = form.conversation_text.data
        try:
            line0 = content.split('\n', 1)[0].strip()
            print(f'line0: {line0}')
            script_time = datetime.strptime(line0, "%Y-%m-%d")
        except:
            script_time = datetime.now()
        print(f'script_time: {script_time}')
        
        # extract daily information
        try:
            summary = llm_service.generate_summary(content)
        except Exception as e:
            flash(f'生成摘要失败: {str(e)}', 'danger')
            return redirect(url_for('index'))
        add_conversation(content, summary, script_time)
        # summary = fake_summary
        # add_conversation(content, summary)
        
        # update long-term memory
        latest_memory = get_latest_memory()
        if latest_memory:
            memory_topics = json.loads(latest_memory.content)['topics']
            latest_day_topics = json.loads(summary)['topics']
            new_memory = llm_service.generate_memory(memory_topics, latest_day_topics)
        else:
            new_memory = json.dumps({'topics': json.loads(summary)['topics']}, ensure_ascii=False)
        add_memory(new_memory)
        # new_memory = fake_memory
        # add_memory(new_memory)
        
        # flash('对话已成功上传并处理！', 'success')
        return redirect(url_for('current_event'))
    return render_template('index.html', form=form)


@app.route('/current-event')
@login_required
def current_event():
    memory = get_latest_memory()
    event_list = []
    if memory:
        for topic in json.loads(memory.content)['topics']:
            event = Event(
                date=datetime.now(),
                title=topic.get('title', ''),
                details=topic.get('summary', '')
            )
            event_list.append(event)
    # print(event_list)
    return render_template('current_event.html', event_list=event_list)


@app.route('/daily')
@login_required
def daily():
    # 查询所有对话并按日期分组
    conversations = get_all_conversations()
    
    # 按日期分组处理
    daily_conversations = {}
    for conv in conversations:
        date_key = conv.created_at.strftime('%Y-%m-%d')
        if date_key not in daily_conversations:
            daily_conversations[date_key] = []
        conv_topics = json.loads(conv.summary).get('topics', [])
        event_list = []
        conv_actions = json.loads(conv.summary).get('action_items', [])
        action_list = []
        num_topic = hash(date_key) % 2 + 2
        num_action = hash(date_key) % 3 + 2
        for topic in conv_topics[:num_topic]:
            event = Event(
                date=conv.created_at,
                title=topic.get('title', ''),
                details=topic.get('summary', '')
            )
            event_list.append(event)
        for action in conv_actions[:num_action]:
            action = Action(
                owner=action.get('owner', ''),
                task=action.get('task', '')
            )
            action_list.append(action)
        # print(action_list)
        daily_conversations[date_key].append((event_list, action_list, conv))
    
    return render_template('daily.html', daily_conversations=daily_conversations)

@app.route('/clear-database', methods=['POST'])
@login_required
def clear_database():
    # 检查是否为管理员用户
    if g.current_user.username != 'admin':
        return jsonify({'status': 'error', 'message': '权限不足，只有管理员可以清空数据库'}), 403
    # 验证确认参数
    confirm_admin = request.form.get('confirmAdmin')
    if confirm_admin != 'admin':
        return jsonify({'status': 'error', 'message': '请输入正确的确认信息'}), 400
    
    try:
        # 删除所有数据
        deletion_counts = clear_all_data()
        print('Clear finished')
        return jsonify({
            'status': 'success', 
            'message': 'Database cleared successfully',
            'deleted_records': deletion_counts
        })
    except Exception as e:
        print(f'Clear failed by error: {str(e)}')
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
            upload_folder = os.path.join(app.root_path, 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            audio_file.save(file_path)
            
            # 调用ASR服务转换音频为文本
            try:
                from services.asr_service import asr_service
                transcription = asr_service.transcribe_audio(file_path)
                
                # 将转录文本作为对话内容处理
                flash('音频上传成功并已转换为文本', 'success')
                return redirect(url_for('index', prefilled_text=transcription))
            except Exception as e:
                flash(f'音频处理失败: {str(e)}', 'danger')
                return redirect(url_for('audio_upload'))
    return render_template('audio_upload.html', form=form)

@app.route('/qa', methods=['GET', 'POST'])
@login_required
def qa():
    form = QAForm()
    answer = None
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' and form.validate_on_submit():
        question = form.question.data
        try:
            memory_topics = {}
            latest_memory = get_latest_memory()
            if latest_memory:
                memory_topics = json.loads(latest_memory.content)['topics']
            conversations = get_all_conversations()
            content_list = []
            for conv in conversations:
                content = conv.content
                content_list.append(content)
            answer = llm_service.generate_answer(question, memory_topics, content_list)
            return jsonify({'status': 'success', 'answer': answer})
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'获取回答失败: {str(e)}'})
    elif form.validate_on_submit():
        question = form.question.data
        try:
            memory_topics = {}
            latest_memory = get_latest_memory()
            if latest_memory:
                memory_topics = json.loads(latest_memory.content)['topics']
            conversations = get_all_conversations()
            content_list = []
            for conv in conversations:
                content = conv.content
                content_list.append(content)
            answer = llm_service.generate_answer(question, memory_topics, content_list)
        except Exception as e:
            flash(f'获取回答失败: {str(e)}', 'danger')
    return render_template('qa.html', form=form, answer=answer)

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

if __name__ == '__main__':
    app.run(debug=True)
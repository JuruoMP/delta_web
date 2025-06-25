import os
import json
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from extensions import db
from flask_wtf import FlaskForm
from wtforms import TextAreaField, FileField, SubmitField
from wtforms.validators import DataRequired
from datetime import datetime
from dotenv import load_dotenv
from services.llm_service import llm_service
from models import Conversation, Event, Action, Memory
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

# 表单定义
class UploadForm(FlaskForm):
    conversation_text = TextAreaField('对话文本', validators=[DataRequired()])
    submit = SubmitField('提交处理')

# 路由
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        content = form.conversation_text.data
        
        # extract daily information
        try:
            summary = llm_service.generate_summary(content)
        except Exception as e:
            flash(f'生成摘要失败: {str(e)}', 'danger')
            return redirect(url_for('index'))
        add_conversation(content, summary)
        # summary = fake_summary
        # add_conversation(content, summary)
        
        # update long-term memory
        latest_memory = get_latest_memory()
        if latest_memory:
            memory_topics = json.loads(latest_memory)['topics']
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
        for topic in conv_topics:
            event = Event(
                date=conv.created_at,
                title=topic.get('title', ''),
                details=topic.get('summary', '')
            )
            event_list.append(event)
        for action in conv_actions:
            action = Action(
                owner=action.get('owner', ''),
                task=action.get('task', '')
            )
            action_list.append(action)
        # print(action_list)
        daily_conversations[date_key].append((event_list, action_list, conv))
    
    return render_template('daily.html', daily_conversations=daily_conversations)

@app.route('/clear-database', methods=['POST'])
def clear_database():
    # 安全验证：仅在设置了CLEAR_DB_SECRET时检查请求头
    clear_db_secret = os.getenv('CLEAR_DB_SECRET')
    if clear_db_secret and request.headers.get('X-Secret-Key') != clear_db_secret:
        print(f'Clear failed by key')
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    
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

# 创建数据库表
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
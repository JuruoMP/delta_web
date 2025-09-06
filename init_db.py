#!/usr/bin/env python
"""
数据库初始化脚本
用于应用模型更改和初始化数据库
"""

import os
from flask import Flask
from extensions import db
from models import User, Conversation, Event, Action, Memory

# 创建Flask应用实例
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_key_for_testing')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///conversations.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# 初始化数据库
with app.app_context():
    db.init_app(app)
    
    # 检查是否需要进行数据库迁移
    # 注意：这是一个简单的检查方法，实际生产环境可能需要使用数据库迁移工具
    try:
        # 检查Conversation表是否已有user_id列
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        conversation_columns = [col['name'] for col in inspector.get_columns('conversation')]
        
        if 'user_id' not in conversation_columns:
            print("检测到数据库结构需要更新...")
            
            # 备份当前数据
            print("正在备份现有数据...")
            conversations = Conversation.query.all()
            events = Event.query.all()
            actions = Action.query.all()
            memories = Memory.query.all()
            
            # 获取第一个用户作为默认用户（通常是admin）
            default_user = User.query.first()
            if not default_user:
                print("错误: 数据库中没有用户，请先创建用户")
                exit(1)
            
            # 删除现有表
            print("正在更新数据库结构...")
            db.drop_all()
            
            # 创建新表
            db.create_all()
            
            # 恢复数据，并将所有数据关联到默认用户
            print("正在恢复数据...")
            for conv in conversations:
                new_conv = Conversation(
                    user_id=default_user.id,
                    content=conv.content,
                    summary=conv.summary,
                    created_at=conv.created_at
                )
                db.session.add(new_conv)
            
            for event in events:
                new_event = Event(
                    user_id=default_user.id,
                    date=event.date,
                    title=event.title,
                    details=event.details,
                    created_at=event.created_at
                )
                db.session.add(new_event)
            
            for action in actions:
                new_action = Action(
                    user_id=default_user.id,
                    owner=action.owner,
                    task=action.task
                )
                db.session.add(new_action)
            
            for memory in memories:
                new_memory = Memory(
                    user_id=default_user.id,
                    content=memory.content,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at
                )
                db.session.add(new_memory)
            
            db.session.commit()
            print("数据库更新成功！所有现有数据已关联到用户: {}".format(default_user.username))
        else:
            print("数据库结构已更新，无需更改。")
    except Exception as e:
        print(f"更新数据库时发生错误: {str(e)}")
        print("提示：如果问题持续存在，您可以考虑删除数据库文件并重新启动应用以创建新的数据库结构。")
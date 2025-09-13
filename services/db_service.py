from datetime import date
from extensions import db
from models import Event, Conversation, Memory, Action


def add_event(user_id, date, title, details):
    """添加新事件到数据库"""
    try:
        event = Event(user_id=user_id, date=date, title=title, details=details)
        db.session.add(event)
        db.session.commit()
        return event
    except Exception as e:
        db.session.rollback()
        raise e


def update_event(user_id, event_id, date=None, title=None, details=None):
    """更新事件记录"""
    try:
        event = Event.query.filter_by(id=event_id, user_id=user_id).first()
        if not event:
            return None
        if date:
            event.date = date
        if title:
            event.title = title
        if details:
            event.details = details
        db.session.commit()
        return event
    except Exception as e:
        db.session.rollback()
        raise e


def add_conversation(user_id, content, summary, date=None):
    """添加新对话到数据库"""
    try:
        if date:
            conversation = Conversation(user_id=user_id, content=content, summary=summary, created_at=date)
        else:
            conversation = Conversation(user_id=user_id, content=content, summary=summary)
        db.session.add(conversation)
        db.session.commit()
        return conversation
    except Exception as e:
        db.session.rollback()
        raise e


def get_all_conversations(user_id):
    """获取指定用户的所有对话记录，按创建时间倒序排列"""
    return Conversation.query.filter_by(user_id=user_id).order_by(Conversation.created_at.desc()).all()


def clear_conversations(user_id):
    """清空指定用户的所有对话记录"""
    try:
        db.session.query(Conversation).filter_by(user_id=user_id).delete()
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        raise e

def clear_all_data(user_id=None):
    """清空数据库中所有表的数据并返回删除记录数
    如果提供了user_id，则只清空该用户的数据；否则清空所有数据（管理员使用）
    同时会清除用户的memory_updating标记"""
    try:
        # 清除用户的memory_updating标记
        from models import User
        if user_id:
            # 只清空指定用户的数据
            user = User.query.get(user_id)
            if user:
                user.memory_updating = False
            
            deleted_conversations = db.session.query(Conversation).filter_by(user_id=user_id).delete()
            deleted_events = db.session.query(Event).filter_by(user_id=user_id).delete()
            deleted_memories = db.session.query(Memory).filter_by(user_id=user_id).delete()
            deleted_actions = db.session.query(Action).filter_by(user_id=user_id).delete()
        else:
            # 清空所有数据（管理员使用）
            # 将所有用户的memory_updating标记设为False
            db.session.query(User).update({User.memory_updating: False})
            
            deleted_conversations = db.session.query(Conversation).delete()
            deleted_events = db.session.query(Event).delete()
            deleted_memories = db.session.query(Memory).delete()
            deleted_actions = db.session.query(Action).delete()
        db.session.commit()
        return {
            'conversations': deleted_conversations,
            'events': deleted_events,
            'memories': deleted_memories,
            'actions': deleted_actions
        }
    except Exception as e:
        db.session.rollback()
        raise e

# def generate_conversation_summary(user_id):
#     """生成指定用户所有对话的汇总"""
#     conversations = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.created_at.asc()).all()
#     if not conversations:
#         return "暂无对话记录"
#     
#     # 拼接所有对话内容
#     all_content = '\n\n'.join([conv.content for conv in conversations])
#     
#     # 这里可以添加更复杂的汇总逻辑
#     summary = f"对话汇总（共{len(conversations)}条）：\n{all_content[:500]}..."
#     return summary

def add_memory(user_id, content):
    """添加新的记忆记录"""
    try:
        memory = Memory(user_id=user_id, content=content)
        db.session.add(memory)
        db.session.commit()
        return memory
    except Exception as e:
        db.session.rollback()
        raise e

def get_latest_memory(user_id):
    """获取指定用户的最新记忆记录"""
    return Memory.query.filter_by(user_id=user_id).order_by(Memory.updated_at.desc()).first()
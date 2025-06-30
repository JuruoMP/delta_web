from datetime import date
from extensions import db
from models import Event, Conversation, Memory


def add_event(date, title, details):
    """添加新事件到数据库"""
    try:
        event = Event(date=date, title=title, details=details)
        db.session.add(event)
        db.session.commit()
        return event
    except Exception as e:
        db.session.rollback()
        raise e


def update_event(event_id, date=None, title=None, details=None):
    """更新事件记录"""
    try:
        event = Event.query.get(event_id)
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


def add_conversation(content, summary, date=None):
    """添加新对话到数据库"""
    try:
        if date:
            conversation = Conversation(content=content, summary=summary, created_at=date)
        else:
            conversation = Conversation(content=content, summary=summary)
        db.session.add(conversation)
        db.session.commit()
        return conversation
    except Exception as e:
        db.session.rollback()
        raise e


def get_all_conversations():
    """获取所有对话记录，按创建时间倒序排列"""
    return Conversation.query.order_by(Conversation.created_at.desc()).all()


def clear_conversations():
    """清空所有对话记录"""
    try:
        db.session.query(Conversation).delete()
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        raise e

def clear_all_data():
    """清空数据库中所有表的数据并返回删除记录数"""
    try:
        deleted_conversations = db.session.query(Conversation).delete()
        deleted_events = db.session.query(Event).delete()
        deleted_memories = db.session.query(Memory).delete()
        db.session.commit()
        return {
            'conversations': deleted_conversations,
            'events': deleted_events,
            'memories': deleted_memories
        }
    except Exception as e:
        db.session.rollback()
        raise e

# def generate_conversation_summary():
#     """生成所有对话的汇总"""
#     conversations = Conversation.query.order_by(Conversation.created_at.asc()).all()
#     if not conversations:
#         return "暂无对话记录"
    
#     # 拼接所有对话内容
#     all_content = '\n\n'.join([conv.content for conv in conversations])
    
#     # 这里可以添加更复杂的汇总逻辑
#     summary = f"对话汇总（共{len(conversations)}条）：\n{all_content[:500]}..."
#     return summary

def add_memory(content):
    """添加新的记忆记录"""
    try:
        memory = Memory(content=content)
        db.session.add(memory)
        db.session.commit()
        return memory
    except Exception as e:
        db.session.rollback()
        raise e

def get_latest_memory():
    """获取最新的记忆记录"""
    return Memory.query.order_by(Memory.updated_at.desc()).first()
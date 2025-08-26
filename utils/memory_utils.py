import datetime
from mem0 import Memory


class MemoryBank:
    def __init__(self, user):
        self.config = {
            "llm": {"provider": "doubao", "config": {"enable_vision": True, "vision_details": "auto"}}, 
            "embedder": {"provider": "doubao"}, 
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "embedding_model_dims": 2560, "on_disk": False, "path": "./qdrant_data"}},
        }
        self.memory = Memory.from_config(self.config)
        self.user = user

    def add_memory(self, message, created_date=None):
        message_list = [{"role": "user", "content": message}]
        created_date = created_date or datetime.datetime.now().isoformat()
        self.memory.add(message_list, user_id=self.user, metadata={"created_at": created_date})

    def search_memory(self, query):
        return self.memory.search(query, user_id=self.user)

    def update_memory(self):
        raise NotImplementedError

    def delete_memory(self):
        # assert False, "DO NOT CLEAR MEMORY!!!"
        self.memory.delete_all(user_id=self.user)

    def get_all(self):
        return self.memory.get_all(user_id=self.user)

    def close(self):
        """关闭Qdrant客户端连接"""
        if hasattr(self.memory, 'vector_store') and hasattr(self.memory.vector_store, 'client'):
            try:
                # 移除客户端关闭调用，避免提前释放资源
                pass
            except Exception as e:
                print(f"关闭Qdrant连接时出错: {e}")

    def extract_qa_memorries(self, query):
        memory_results = self.memory.search(query, user_id=self.user)['results']
        memory_content = [self._extract_date(x['created_at']) + ' : ' + x['memory'] for x in memory_results]
        # print(f"[debug] memory_conent = {memory_content}")
        return memory_content
    
    @classmethod
    def _extract_date(cls, date_str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        # return date_obj.strftime("%Y-%m-%d %H:%M:%S")
        return date_obj.strftime("%Y-%m-%d")


def unittest_memory():
    memory_bank = MemoryBank("test_user")
    memory_bank.add_memory("A是B的好朋友，A今天去看电影没有带B，B很伤心。")
    memory_bank.add_memory("A喜欢看科幻电影，B喜欢看喜剧。")
    memory_bank.add_memory("A今天去超市买菜遇上了C，C说买的是B推荐的商品。")
    # print(memory_bank.get_all())
    print(memory_bank.search_memory("A喜欢看什么？"))
    # memory_bank.delete_memory()
    # print(memory_bank.get_all())
    
if __name__ == '__main__':
    unittest_memory()
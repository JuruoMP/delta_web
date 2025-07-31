from mem0 import Memory


class MemoryBank:
    def __init__(self, user):
        self.config = {
            "llm": {"provider": "doubao"}, 
            "embedder": {"provider": "doubao"}, 
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "embedding_model_dims": 2560}},
        }
        self.memory = Memory.from_config(self.config)
        self.user = user

    def add_memory(self, message):
        message_list = [{"role": "user", "content": message}]
        self.memory.add(message_list, user_id=self.user)

    def search_memory(self, query):
        return self.memory.search(query, user_id=self.user)

    def update_memory(self):
        raise NotImplementedError

    def delete_memory(self):
        assert False, "DO NOT CLEAR MEMORY!!!"
        self.memory.delete_all(user_id=self.user)

    def get_all(self):
        return self.memory.get_all(user_id=self.user)


def unittest_memory():
    memory_bank = MemoryBank("test_user")
    memory_bank.add_memory("A是B的好朋友，A今天去看电影没有带B，B很伤心。")
    memory_bank.add_memory("A喜欢看科幻电影，B喜欢看喜剧。")
    memory_bank.add_memory("A今天去超市买菜遇上了C，C说买的是B推荐的商品。")
    print(memory_bank.get_all())
    print(memory_bank.search_memory("A喜欢看什么？"))
    memory_bank.delete_memory()
    print(memory_bank.get_all())
    
if __name__ == '__main__':
    unittest_memory()

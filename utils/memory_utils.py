from mem0 import Memory


class MemoryBank:
    def __init__(self, user):
        self.memory = Memory()
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

    

import json
import os
from services.llm_service import LLMService

class LLMUtils:
    def __init__(self, llm_service):
        # 定义prompt文件路径
        self.PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../prompts')
        self.llm_service = llm_service
        
        # 加载prompt模板
        self.summary_system_prompt = self.load_prompt('summary_system_prompt.txt')
        self.summary_prompt_template = self.load_prompt('summary_prompt_template.txt')
        self.memory_system_prompt = self.load_prompt('memory_system_prompt.txt')
        self.memory_prompt_template = self.load_prompt('memory_prompt_template.txt')
        self.qa_system_prompt = self.load_prompt('qa_system_prompt.txt')
        self.qa_prompt_template_en = self.load_prompt('qa_prompt_template_en.txt')
        self.qa_prompt_template_zh = self.load_prompt('qa_prompt_template_zh.txt')

    def load_prompt(self, file_name):
        """加载prompt模板文件"""
        file_path = os.path.join(self.PROMPTS_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def gen_conversation_summary(self, json_content, model_name=None):
        summary_prompt_str = self.summary_prompt_template.replace('{{json_str}}', json.dumps(json_content, indent=2, ensure_ascii=False))
        return self.llm_service.chat(self.summary_system_prompt, summary_prompt_str, model_name=model_name)

    def gen_memory(self, historical_data, latest_day_data, model_name=None):
        memory_prompt_str = self.memory_prompt_template.replace('{{historical_data}}', json.dumps(historical_data, indent=2, ensure_ascii=False)).replace('{{latest_day_data}}', json.dumps(latest_day_data, indent=2, ensure_ascii=False))
        return self.llm_service.chat(self.memory_system_prompt, memory_prompt_str, model_name=model_name)

    def get_qa_answer(self, current_memory, retrieved_contexts, user_query, model_name=None):
        chinese_chars = sum(1 for c in user_query if '\u4e00' <= c <= '\u9fff')
        total_chars = max(len(user_query), 1)
        
        if chinese_chars / total_chars > 0.3:  # 中文占比超过30%判定为中文问题
            qa_prompt_str = self.qa_prompt_template_zh.replace('{{current_memory}}', current_memory).replace('{{retrieved_contexts}}', retrieved_contexts).replace('{{user_query}}', user_query)
        else:
            qa_prompt_str = self.qa_prompt_template_en.replace('{{current_memory}}', current_memory).replace('{{retrieved_contexts}}', retrieved_contexts).replace('{{user_query}}', user_query)
        
        return self.llm_service.chat(self.qa_system_prompt, qa_prompt_str, model_name=model_name)


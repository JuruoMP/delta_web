import os
import json
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
        self.qa_soft_system_prompt = self.load_prompt('qa_soft_system_prompt.txt')
        self.qa_soft_prompt_template_en = self.load_prompt('qa_soft_prompt_template_en.txt')
        self.qa_soft_prompt_template_zh = self.load_prompt('qa_soft_prompt_template_zh.txt')
        self.conversation_analysis_system_prompt = self.load_prompt('conversation_analysis_system_prompt.txt')
        self.conversation_analysis_prompt_template = self.load_prompt('conversation_analysis_prompt_template.txt')

    def load_prompt(self, file_name):
        """加载prompt模板文件"""
        file_path = os.path.join(self.PROMPTS_DIR, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def gen_conversation_summary(self, content, model_name=None):
        summary_prompt_str = self.summary_prompt_template.replace('{{json_str}}', json.dumps({'content': content}, indent=2, ensure_ascii=False))
        return self.llm_service.chat(self.summary_system_prompt, summary_prompt_str, model_name=model_name)

    def gen_memory(self, historical_data, latest_day_data, model_name=None):
        memory_prompt_str = self.memory_prompt_template.replace('{{historical_data}}', json.dumps(historical_data, indent=2, ensure_ascii=False)).replace('{{latest_day_data}}', json.dumps(latest_day_data, indent=2, ensure_ascii=False))
        return self.llm_service.chat(self.memory_system_prompt, memory_prompt_str, model_name=model_name)

    def get_qa_answer(self, user_query, current_memory, retrieved_contexts, model_name=None):
        chinese_chars = sum(1 for c in user_query if '\u4e00' <= c <= '\u9fff')
        total_chars = max(len(user_query), 1)

        current_memory_json = json.dumps(current_memory, indent=2, ensure_ascii=False)
        retrived_contexts_str = '\n\n'.join(retrieved_contexts)
        
        if chinese_chars / total_chars > 0.3:  # 中文占比超过30%判定为中文问题
            qa_prompt_str = self.qa_prompt_template_zh.replace('{{current_memory}}', current_memory_json).replace('{{retrieved_contexts}}', retrived_contexts_str).replace('{{user_query}}', user_query)
        else:
            qa_prompt_str = self.qa_prompt_template_en.replace('{{current_memory}}', current_memory_json).replace('{{retrieved_contexts}}', retrived_contexts_str).replace('{{user_query}}', user_query)
        
        return self.llm_service.chat(self.qa_system_prompt, qa_prompt_str, model_name=model_name)
    
    def get_qa_answer_soft(self, user_query, current_memory, retrieved_contexts, model_name=None):
        chinese_chars = sum(1 for c in user_query if '\u4e00' <= c <= '\u9fff')
        total_chars = max(len(user_query), 1)

        current_memory_json = json.dumps(current_memory, indent=2, ensure_ascii=False)
        retrived_contexts_str = '\n\n'.join(retrieved_contexts)
        
        if chinese_chars / total_chars > 0.3:  # 中文占比超过30%判定为中文问题
            qa_prompt_str = self.qa_soft_prompt_template_zh.replace('{{current_memory}}', current_memory_json).replace('{{retrieved_contexts}}', retrived_contexts_str).replace('{{user_query}}', user_query)
        else:
            qa_prompt_str = self.qa_soft_prompt_template_en.replace('{{current_memory}}', current_memory_json).replace('{{retrieved_contexts}}', retrived_contexts_str).replace('{{user_query}}', user_query)
        
        return self.llm_service.chat(self.qa_soft_system_prompt, qa_prompt_str, model_name=model_name)

    def analyze_conversation(self, conversation_content, model_name=None, system_prompt=None):
        """分析对话内容并生成结构化信息整理文档"""
        analysis_prompt_str = self.conversation_analysis_prompt_template.replace('{{conversation_content}}', conversation_content)
        
        # 使用自定义的system_prompt或默认的system_prompt
        selected_system_prompt = system_prompt if system_prompt else self.conversation_analysis_system_prompt
        
        return self.llm_service.chat(selected_system_prompt, analysis_prompt_str, model_name=model_name)
        
    def stream_analyze_conversation(self, conversation_content, model_name=None, request_id=None, content_length=0, system_prompt=None):
        """流式分析对话内容并生成结构化信息整理文档，支持从特定位置恢复"""
        analysis_prompt_str = self.conversation_analysis_prompt_template.replace('{{conversation_content}}', conversation_content)
        
        # 使用自定义的system_prompt或默认的system_prompt
        selected_system_prompt = system_prompt if system_prompt else self.conversation_analysis_system_prompt
        
        # 如果提供了content_length，需要跳过前面的内容
        if request_id and content_length > 0:
            # 从指定位置恢复流式响应
            full_response = ""
            chunks = self.llm_service.stream_chat(
                selected_system_prompt,
                analysis_prompt_str,
                model_name=model_name
            )
            
            for chunk in chunks:
                full_response += chunk
                # 只有当累积的内容长度超过指定的content_length时，才开始yield内容
                if len(full_response) > content_length:
                    # 计算需要跳过的字符数
                    skip_count = len(full_response) - content_length
                    # 只yield新的内容
                    yield chunk[skip_count:]
        else:
            # 正常流式响应，不跳过任何内容
            for chunk in self.llm_service.stream_chat(
                selected_system_prompt,
                analysis_prompt_str,
                model_name=model_name
            ):
                yield chunk


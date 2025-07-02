import os
import json
import random
import requests
import time
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark
from utils.llm_utils import llm_gen_conversation_summary, llm_gen_memory, llm_get_qa_answer

# 加载环境变量
load_dotenv()

class LLMService:
    def __init__(self):
        self.model_configs = {
            "default": {
                "model_id": "doubao-seed-1-6-250615", 
                "max_tokens": 32768,
                "api_key": os.getenv("ARK_API_KEY_DOBAO_1_6")
            },
            "default-flash": {
                "model_id": "ep-20250702234129-6tnzb", 
                "max_tokens": 32768,
                "api_key": os.getenv("ARK_API_KEY_DOBAO_1_6_FLASH")
            },
            "R1": {
                "model_id": "ep-20250703015232-5mrzd",
                "max_tokens": 32768,
                "api_key": os.getenv("ARK_API_KEY_DEEPSEEK_R1")
            }
        }
        self.model_clients = {}
        for model_name, config in self.model_configs.items():
            api_key = config.get("api_key")
            if not api_key:
                raise ValueError(f"API key for model {model_name} is not set in environment variables")
            self.model_clients[model_name] = Ark(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=api_key,
            )
        self.default_model = "default"

    def call_openai_api_with_retry(self, messages, model, max_retries=3, delay=5):
        retries = 0
        while retries < max_retries:
            try:
                # if self.model == 'doubao-1.6':
                #     response = self.model_client.chat.completions.create(model=self.conf["model_name"], messages=messages, max_tokens=self.conf["max_tokens"], thinking={"type":"disabled"})
                # else:
                response = self.model_clients[model].chat.completions.create(model=self.model_configs[model]["model_id"], messages=messages)
                return response
            # except openai.OpenAIError as e:
            #     print(f"OpenAI error occurred: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
            retries += 1
            _delay = delay + random.randint(1, 3) * delay
            print(f"Retrying... ({retries}/{max_retries}), sleep: {_delay}s")
            time.sleep(_delay)
        raise Exception("API call failed after maximum retries")

    def chat(self, system_prompt, prompt, model_name=None):
        messages = []
        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            model = model_name or self.default_model
            if model not in self.model_configs:
                raise ValueError(f"Unsupported model: {model}")
            response = self.call_openai_api_with_retry(messages, model=model)
            call_llm_log = {
                "model": model,
                "messages": messages,
                "response": response.choices[0].message.content.strip()
            }
            with open('call_llm_log.json', 'a') as f:
                json.dump(call_llm_log, f, ensure_ascii=False)
                f.write('\n')
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Final error: {e}")
            return ''

    def generate_summary(self, conversation_content, model_name=None):
        return llm_gen_conversation_summary(self, conversation_content, model_name=model_name)

    def generate_memory(self, historical_data, latest_day_data, model_name=None):
        return llm_gen_memory(self, historical_data, latest_day_data, model_name=model_name)

    def generate_answer(self, question, current_memory, retrived_contexts, model_name=None):
        # return f"answer of {question}"
        current_memory_json = json.dumps(current_memory, indent=2, ensure_ascii=False)
        retrived_contexts_str = '\n\n'.join(retrived_contexts)
        return llm_get_qa_answer(self, current_memory_json, retrived_contexts_str, question, model_name=model_name)


llm_service = LLMService()


if __name__ == "__main__":
    llm_service = LLMService()
    print(llm_service.chat("你是一个助手", "你好"))
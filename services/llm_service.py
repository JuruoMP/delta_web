import os
import json
import random
import requests
import time
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark
from utils.llm_utils import llm_gen_conversation_summary, llm_gen_memory

# 加载环境变量
load_dotenv()

class LLMService:
    def __init__(self):
        self.model_client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.getenv("ARK_API_KEY"),
        )
        self.conf = {
            "model_name": "doubao-seed-1-6-250615",
            "max_tokens": 32768,
        }

    def call_openai_api_with_retry(self, messages, max_retries=2, delay=60):
        retries = 0
        while retries < max_retries:
            try:
                # if self.model == 'doubao-1.6':
                #     response = self.model_client.chat.completions.create(model=self.conf["model_name"], messages=messages, max_tokens=self.conf["max_tokens"], thinking={"type":"disabled"})
                # else:
                response = self.model_client.chat.completions.create(model=self.conf["model_name"], messages=messages)
                return response
            # except openai.OpenAIError as e:
            #     print(f"OpenAI error occurred: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
            retries += 1
            _delay = delay + random.randint(1, 5) * delay
            print(f"Retrying... ({retries}/{max_retries}), sleep: {_delay}s")
            time.sleep(_delay)
        raise Exception("API call failed after maximum retries")

    def chat(self, system_prompt, prompt):
        messages = []
        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self.call_openai_api_with_retry(messages)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Final error: {e}")
            return ''

    def generate_summary(self, conversation_content):
        return llm_gen_conversation_summary(self, conversation_content)

    def generate_memory(self, historical_data, latest_day_data):
        return llm_gen_memory(self, historical_data, latest_day_data)


llm_service = LLMService()


if __name__ == "__main__":
    llm_service = LLMService()
    print(llm_service.chat("你是一个助手", "你好"))
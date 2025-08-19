import json
import time
import uuid
import requests
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class ASRService:
    def __init__(self):
        # 从环境变量获取配置
        self.appid = os.getenv("ASR_APP_ID", "7866300538")
        self.token = os.getenv("ASR_ACCESS_TOKEN", "If-UiysVVmEMyGNjP8smhltrnzMdn4LS")
        self.submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
        self.query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"

    def submit_task(self, file_url, format="mp3"):
        task_id = str(uuid.uuid4())

        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1"
        }

        # 根据格式设置默认参数
        audio_params = {
            "url": file_url,
            "format": format,
            "codec": "raw",
            "rate": 16000,
            "bits": 16,
            "channel": 1
        }

        # 针对不同格式可能需要调整的参数
        if format == "wav":
            audio_params["codec"] = "pcm"
        elif format == "ogg":
            audio_params["codec"] = "vorbis"

        request = {
            "user": {
                "uid": "fake_uid"
            },
            "audio": audio_params,
            "request": {
                "model_name": "bigmodel",
                # "enable_itn": True,
                # "enable_punc": True,
                # "enable_ddc": True,
                "show_utterances": True,
                # "enable_channel_split": True,
                # "vad_segment": True,
                # "enable_speaker_info": True,
                "corpus": {
                    # "boosting_table_name": "test",
                    "correct_table_name": "",
                    "context": ""
                }
            }
        }
        print(f'Submit task id: {task_id}')
        response = requests.post(self.submit_url, data=json.dumps(request), headers=headers)
        if 'X-Api-Status-Code' in response.headers and response.headers["X-Api-Status-Code"] == "20000000":
            print(f'Submit task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
            print(f'Submit task response header X-Api-Message: {response.headers["X-Api-Message"]}')
            x_tt_logid = response.headers.get("X-Tt-Logid", "")
            print(f'Submit task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
            return task_id, x_tt_logid
        else:
            print(f'Submit task failed and the response headers are: {response.headers}')
            raise Exception(f"Submit task failed with status code: {response.headers.get('X-Api-Status-Code', 'unknown')}")

    def query_task(self, task_id, x_tt_logid):
        headers = {
            "X-Api-App-Key": self.appid,
            "X-Api-Access-Key": self.token,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            "X-Tt-Logid": x_tt_logid  # 固定传递 x-tt-logid
        }

        response = requests.post(self.query_url, json.dumps({}), headers=headers)

        if 'X-Api-Status-Code' in response.headers:
            print(f'Query task response header X-Api-Status-Code: {response.headers["X-Api-Status-Code"]}')
            print(f'Query task response header X-Api-Message: {response.headers["X-Api-Message"]}')
            print(f'Query task response header X-Tt-Logid: {response.headers["X-Tt-Logid"]}\n')
        else:
            print(f'Query task failed and the response headers are: {response.headers}')
            raise Exception(f"Query task failed with headers: {response.headers}")
        return response

    def transcribe_audio(self, file_url, format="mp3"):
        """
        转录音频文件为文本
        :param file_url: 音频文件的在线URL
        :param format: 音频文件格式，支持mp3、wav、ogg等
        :return: 转录结果
        """
        task_id, x_tt_logid = self.submit_task(file_url, format)
        while True:
            query_response = self.query_task(task_id, x_tt_logid)
            code = query_response.headers.get('X-Api-Status-Code', "")
            if code == '20000000':  # task finished
                result = query_response.json()
                print("SUCCESS!")
                return result
            elif code != '20000001' and code != '20000002':  # task failed
                print("FAILED!")
                raise Exception(f"Transcription failed with status code: {code}")
            time.sleep(1)


if __name__ == '__main__':
    # 示例用法
    file_url = ""  # 需要使用在线URL，推荐使用TOS
    if file_url:
        try:
            result = asr_service.transcribe_audio(file_url)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("请设置有效的音频文件URL")

# Delta Web API 文档

## 概述
Delta Web API 提供了一系列无需登录即可访问的接口，支持手机App进行数据交互。API涵盖了对话管理、问答系统、记忆存储和音频处理等功能。

## 基础URL
所有API的基础URL为：`http://<your-server-ip>:<port>/api`

## 认证方式
目前所有API接口均不需要登录认证即可访问。

## API端点列表

### 对话相关API

#### 获取所有对话
- **URL**: `/conversations`
- **方法**: `GET`
- **描述**: 获取系统中所有对话记录
- **响应格式**: 
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "content": "对话内容...",
      "summary": "对话摘要...",
      "created_at": "2023-10-01T12:00:00Z"
    },
    ...
  ]
}
```

#### 获取特定对话
- **URL**: `/conversations/<conv_id>`
- **方法**: `GET`
- **描述**: 获取指定ID的对话详情
- **参数**: 
  - `conv_id`: 对话ID
- **响应格式**: 
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "content": "对话内容...",
    "summary": "对话摘要...",
    "created_at": "2023-10-01T12:00:00Z"
  }
}
```

#### 创建新对话
- **URL**: `/create_conversation`
- **方法**: `POST`
- **描述**: 提交新对话并处理
- **请求体**: 
```json
{
  "content": "对话内容..."
}
```
- **响应格式**: 
```json
{
  "status": "success",
  "message": "对话已成功处理",
  "data": {
    "id": 1,
    "summary": "对话摘要..."
  }
}
```

### 问答相关API

#### 提交问题并获取回答
- **URL**: `/qa`
- **方法**: `POST`
- **描述**: 提交问题并获取AI回答
- **请求体**: 
```json
{
  "question": "你的问题...",
  "model": "default"  // 可选，模型名称
}
```
- **响应格式**: 
```json
{
  "status": "success",
  "data": {
    "question": "你的问题...",
    "answer": "AI回答...",
    "context": ["相关上下文..."]
  }
}
```

### 记忆相关API

#### 获取所有记忆
- **URL**: `/memories`
- **方法**: `GET`
- **描述**: 获取系统中所有记忆记录
- **响应格式**: 
```json
{
  "status": "success",
  "data": [
    {
      "content": "记忆内容...",
      "created_at": "2023-10-01T12:00:00Z"
    },
    ...
  ]
}
```

#### 获取最新记忆
- **URL**: `/memories/latest`
- **方法**: `GET`
- **描述**: 获取最新的记忆记录
- **响应格式**: 
```json
{
  "status": "success",
  "data": {
    "id": 1,
    "content": "记忆内容...",
    "updated_at": "2023-10-01T12:00:00Z"
  }
}
```

### 事件相关API

#### 获取所有事件
- **URL**: `/events`
- **方法**: `GET`
- **描述**: 获取系统中所有事件记录
- **响应格式**: 
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "date": "2023-10-01T12:00:00Z",
      "title": "事件标题...",
      "details": "事件详情..."
    },
    ...
  ]
}
```

#### 获取最新事件
- **URL**: `/events/latest`
- **方法**: `GET`
- **描述**: 获取基于最新记忆生成的事件列表
- **响应格式**: 
```json
{
  "status": "success",
  "data": [
    {
      "date": "2023-10-01T12:00:00Z",
      "title": "事件标题...",
      "details": "事件详情..."
    },
    ...
  ]
}
```

### 音频处理API

#### 上传音频并获取转录文本
- **URL**: `/audio/transcribe`
- **方法**: `POST`
- **描述**: 上传音频文件，获取语音识别后的文本
- **请求格式**: `multipart/form-data`
- **参数**: 
  - `audio_file`: 音频文件（支持mp3、wav、ogg格式）
- **响应格式**: 
```json
{
  "status": "success",
  "data": {
    "transcription": "转录文本..."
  }
}
```

## 错误处理
所有API错误响应格式如下：
```json
{
  "status": "error",
  "message": "错误描述..."
}
```

## 使用示例
### Python请求示例
```python
import requests

# 获取所有对话
def get_all_conversations():
    url = "http://<your-server-ip>:<port>/api/conversations"
    response = requests.get(url)
    return response.json()

# 提交问题
def ask_question(question):
    url = "http://<your-server-ip>:<port>/api/qa"
    data = {"question": question}
    response = requests.post(url, json=data)
    return response.json()
```

### JavaScript请求示例
```javascript
// 获取所有对话
async function getAllConversations() {
  const response = await fetch('http://<your-server-ip>:<port>/api/conversations');
  return response.json();
}

// 提交问题
async function askQuestion(question) {
  const response = await fetch('http://<your-server-ip>:<port>/api/qa', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ question })
  });
  return response.json();
}
```
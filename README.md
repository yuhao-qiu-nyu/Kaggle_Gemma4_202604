# ASL Sign Language Recognition API

手语识别 + Gemma 4 教练反馈，一个 API 搞定。

## 快速开始

```bash
# 1. 安装依赖
cd /Volumes/senzu/LLMPROJECTs/ASL
.venv/bin/pip install -r requirements.txt

# 2. 启动服务
GEMINI_API_KEY=你的key TF_USE_LEGACY_KERAS=1 \
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 获取 Gemini API Key

1. 打开 https://aistudio.google.com/apikey
2. 用 Google 账号登录
3. 点 **Create API Key**，复制生成的 Key
4. 启动时通过 `GEMINI_API_KEY` 环境变量传入

> 不设 Key 也能启动，`/predict` 正常可用，只是 `/coach` 会返回 503。

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/predict` | POST | 纯识别（接收 landmark JSON） |
| `/predict_from_file` | POST | 纯识别（接收 parquet 文件路径） |
| `/coach` | POST | 识别 + Gemma 4 教练反馈 |
| `/coach_from_file` | POST | 同上（parquet 文件版） |
| `/coach_video` | POST | **完整管线**：上传视频 → MediaPipe → 识别 → 教练反馈 |

交互式文档：启动后访问 http://localhost:8000/docs

## 调用示例

```bash
# 纯识别
curl -X POST http://localhost:8000/predict_from_file \
  -H "Content-Type: application/json" \
  -d '{"parquet_path": "/path/to/landmarks.parquet", "topk": 5}'

# 识别 + 教练反馈（从 parquet）
curl -X POST http://localhost:8000/coach_from_file \
  -H "Content-Type: application/json" \
  -d '{
    "parquet_path": "/path/to/landmarks.parquet",
    "topk": 3,
    "user_goal": "learn daily ASL vocabulary",
    "history_errors": ["cloud", "grandma"]
  }'

# 完整管线：上传视频，一步到位
curl -X POST http://localhost:8000/coach_video \
  -F "video=@my_sign.mp4" \
  -F "topk=3" \
  -F "user_goal=learn daily ASL vocabulary" \
  -F "history_errors=cloud,grandma"
```

## 项目结构

```
app/
├── main.py                 # FastAPI 路由 + 启动
├── model.py                # 模型架构 + Ensemble 推理
├── preprocess.py           # Landmark 预处理
├── schemas.py              # 请求/响应数据模型
├── llm.py                  # Gemma 4 调用层（可替换为其他 LLM）
└── mediapipe_extractor.py  # 视频 → 543 landmarks/帧
holistic_landmarker.task    # MediaPipe 全身地标模型（13MB）
```

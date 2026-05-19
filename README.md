# ASL Sign Language Recognition API

Isolated ASL sign recognition with optional Gemini coaching feedback, exposed as a FastAPI service and a simple web UI.

The recognizer is built on the [Google — Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/) (ISLR) Kaggle competition architecture. This deployment restricts inference to **25 signs** listed in `app/sign_map.json` (the full competition vocabulary has 250 signs).

## Supported signs (25)

These are the only labels the API will return. Predictions outside this set are masked at inference time.

| | | | | |
|---|---|---|---|---|
| bye | please | thankyou | hello | who |
| where | why | yes | no | drink |
| water | milk | apple | dad | mom |
| cat | dog | home | sleep | hungry |
| happy | sad | hot | look | book |

Alphabetical list: `apple`, `book`, `bye`, `cat`, `dad`, `dog`, `drink`, `happy`, `hello`, `home`, `hot`, `hungry`, `look`, `milk`, `mom`, `no`, `please`, `sad`, `sleep`, `thankyou`, `water`, `where`, `who`, `why`, `yes`.

**How to test:** record a short `.mp4` of one of the signs above and call `POST /coach_video`, or use pre-extracted landmark `.parquet` files with `/predict_from_file` / `/coach_from_file`. The web UI at `http://localhost:8000/` uses the same vocabulary.

## Prerequisites & downloads

### 1. Python dependencies

```bash
cd /path/to/ASL
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Ensemble model weights (required)

Place the ISLR `.h5` weight files under:

```
tensorflow/islr-models/
├── islr-fp16-192-8-seed42-foldall-last.h5
├── islr-fp16-192-8-seed43-foldall-last.h5
├── islr-fp16-192-8-seed44-foldall-last.h5
└── islr-fp16-192-8-seed45-foldall-last.h5
```

These come from the 1st-place ISLR solution (Kaggle notebook input path: `/kaggle/input/islr-models/`). Download them from the competition’s model/dataset attachments or your own training run, then copy them into `tensorflow/islr-models/`. The server will not start without at least one matching `islr-fp16-192-8-seed*-foldall-last.h5` file.

Set `ASL_BASE_DIR` if the project root is not the default path used in `app/main.py`.

### 3. MediaPipe Holistic model (required for video upload)

```bash
wget -O holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

Put the file in the repository root (or set `HOLISTIC_MODEL_PATH`).

### 4. Kaggle datasets (for training, evaluation, and parquet-based testing)

| Dataset | Kaggle slug | Purpose |
|---------|-------------|---------|
| **ASL Signs (competition)** | [`google/asl-signs`](https://www.kaggle.com/competitions/asl-signs/data) | Official ISLR data: `train.csv`, `sign_to_prediction_index_map.json` (250 signs), and `train_landmark_files/*.parquet` (543 landmarks per frame). Accept competition rules before downloading. |
| **Preprocessed landmarks** | [`sohier/461054610546105`](https://www.kaggle.com/datasets/sohier/461054610546105) | Ready-to-use landmark parquets used in the project notebooks (`kaggle datasets download -d sohier/461054610546105`). |
| **WLASL processed videos** (optional) | [`risangbaskoro/wlasl-processed`](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) | Raw/processed WLASL clips if you want to build landmarks from video instead of using competition parquets. |

Example (requires [Kaggle API](https://www.kaggle.com/docs/api) credentials in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c asl-signs -p datasets/asl-signs
kaggle datasets download -d sohier/461054610546105 -p datasets/sohier --unzip
```

Large artifacts (`*.h5`, `holistic_landmarker.task`, `datasets/`) are gitignored; download them locally after cloning.

## Quick start

```bash
export GEMINI_API_KEY=your-api-key-here   # optional, for /coach endpoints
export TF_USE_LEGACY_KERAS=1

.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/ for the UI, or http://localhost:8000/docs for the interactive API reference.

## Gemini API key (optional)

1. Go to https://aistudio.google.com/apikey  
2. Sign in with a Google account  
3. Click **Create API Key** and copy the key  
4. Pass it as `GEMINI_API_KEY` when starting the server  

Without a key, the service still runs: `/predict*` works; `/coach*` returns HTTP 503.

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/predict` | POST | Recognition from landmark JSON |
| `/predict_from_file` | POST | Recognition from a `.parquet` path on disk |
| `/coach` | POST | Recognition + Gemini coaching feedback |
| `/coach_from_file` | POST | Same as `/coach`, from a parquet path |
| `/coach_video` | POST | Full pipeline: upload `.mp4` → MediaPipe → recognize → coach |

## Example requests

```bash
# Recognition from parquet
curl -X POST http://localhost:8000/predict_from_file \
  -H "Content-Type: application/json" \
  -d '{"parquet_path": "/path/to/landmarks.parquet", "topk": 5}'

# Recognition + coaching (parquet)
curl -X POST http://localhost:8000/coach_from_file \
  -H "Content-Type: application/json" \
  -d '{
    "parquet_path": "/path/to/landmarks.parquet",
    "topk": 3,
    "user_goal": "learn daily ASL vocabulary",
    "history_errors": ["cloud", "grandma"]
  }'

# Full pipeline: video upload
curl -X POST http://localhost:8000/coach_video \
  -F "video=@my_sign.mp4" \
  -F "topk=3" \
  -F "user_goal=learn daily ASL vocabulary" \
  -F "history_errors=cloud,grandma"
```

## Project layout

```
app/
├── main.py                 # FastAPI routes and startup
├── model.py                # Architecture + ensemble inference
├── preprocess.py           # Landmark preprocessing
├── schemas.py              # Request/response models
├── llm.py                  # Gemini coaching layer
├── mediapipe_extractor.py  # Video → 543 landmarks per frame
└── sign_map.json           # 25-sign subset used at inference
tensorflow/islr-models/     # Ensemble .h5 weights (download separately)
holistic_landmarker.task    # MediaPipe model (~13 MB, download separately)
static/index.html           # Web UI
```

# 🎙️ VoiceGuard — AI Voice Detection API (v2.0) — Complete Project Reference

> **One-line summary:** A FastAPI-based REST API that classifies audio as **AI-Generated** or **Human** using a Hugging Face Wav2Vec2-based deepfake-audio-detection model.

---

## 📌 Table of Contents

1. [Why This API Was Created](#-why-this-api-was-created)
2. [What It Does (Features)](#-what-it-does-features)
3. [Tech Stack & Dependencies](#-tech-stack--dependencies)
4. [Project Structure](#-project-structure)
5. [Core Architecture & How It Works](#-core-architecture--how-it-works)
6. [API Endpoints](#-api-endpoints)
7. [Request / Response Format (Hackathon Spec)](#-request--response-format-hackathon-spec)
8. [Authentication & Security](#-authentication--security)
9. [Environment Variables](#-environment-variables)
10. [Running Locally](#-running-locally)
11. [Deployment (Hugging Face Spaces)](#-deployment-hugging-face-spaces)
12. [Deployment (Alternative — Render)](#-deployment-alternative--render)
13. [Model Training & Fine-Tuning](#-model-training--fine-tuning)
14. [Testing the API](#-testing-the-api)
15. [Troubleshooting Guide](#-troubleshooting-guide)
16. [Hugging Face Space URLs](#-hugging-face-space-urls)
17. [Hugging Face CLI Fixes (Windows / PowerShell)](#-hugging-face-cli-fixes-windows--powershell)
18. [Key Files Quick Reference](#-key-files-quick-reference)

---

## 🎯 Why This API Was Created

This API was built for a **hackathon** focused on AI deepfake audio detection. The goal is to provide:

- A reliable, production-ready API that can detect whether a given voice sample is **real (human)** or **AI-generated (deepfake / synthetic)**.
- Hackathon-compliant endpoints with a standardised request/response format.
- Both a **developer API** (`POST /detect`) and a **Web UI** for non-technical testers.
- Easy deployment on **Hugging Face Spaces** (free tier, Docker-based).

The project addresses the growing threat of AI-generated voice deepfakes used in fraud, social engineering, and misinformation.

---

## ✨ What It Does (Features)

| Feature | Description |
|---|---|
| **Voice Classification** | Classifies audio as `AI_GENERATED` or `HUMAN` with a confidence score (0.0–1.0). |
| **Web UI** | Record audio in-browser or upload a file — no coding needed. |
| **REST API** | `POST /detect` endpoint accepts Base64-encoded audio. |
| **Multi-Language** | Supports Tamil, English, Hindi, Malayalam, Telugu labels. |
| **Multi-Format** | Accepts MP3, WAV, FLAC, OGG audio files. |
| **Human-Readable Explanations** | Returns a plain-English reason for the decision (varies by confidence). |
| **API Key Auth** | Optional API key authentication via `X-API-Key` header. |
| **Health Check** | `GET /health` for uptime monitoring. |
| **Dockerised** | Ready-to-deploy Docker image (Python 3.9 slim). |
| **Model Fine-Tuning** | Scripts included for fine-tuning on custom datasets (local + Google Colab). |

---

## 🛠 Tech Stack & Dependencies

### Runtime / Framework
| Component | Technology |
|---|---|
| **Web Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **Containerisation** | Docker (Python 3.9-slim base) |
| **Hosting** | Hugging Face Spaces (primary), Render (alternative) |

### ML / Audio
| Library | Purpose |
|---|---|
| `torch` | PyTorch — inference engine |
| `torchaudio` | Audio utilities |
| `transformers` | Hugging Face model loading (`AutoModelForAudioClassification`) |
| `librosa` | Audio loading, resampling to 16 kHz, normalisation |
| `soundfile` / `scipy` | Audio I/O backends |

### Utilities
| Library | Purpose |
|---|---|
| `python-dotenv` | Loads `.env` configuration |
| `python-multipart` | Multipart form handling by FastAPI |
| `numpy` | Array manipulation |
| `datasets` / `evaluate` / `accelerate` | Used for model training scripts |

### System Dependencies (Dockerfile)
- `libsndfile1` — required by librosa / soundfile
- `ffmpeg` — audio format conversion

---

## 📁 Project Structure

```
voice-detection-api/
│
├── app/                          # Core application
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, routes, request/response models
│   ├── config.py                 # Settings (model name, sample rate, API key)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio.py              # Base64 decoding & audio preprocessing
│   │   ├── model.py              # VoiceDetector singleton (loads HF model, runs inference)
│   │   └── explanation.py        # Human-readable explanation generator
│   └── static/
│       └── index.html            # Web UI (record / upload audio in browser)
│
├── tests/                        # Test suite
│   ├── audio_samples/            # Sample audio files
│   ├── sample_silence.wav
│   ├── create_sample.py
│   ├── create_test_audio.py
│   ├── full_test.py
│   ├── simple_test.py
│   └── test_provided_audio.py
│
├── train_model.py                # Local fine-tuning script
├── colab_train_script.py         # Google Colab fine-tuning notebook (Python script)
│
├── test_api.py                   # API integration tests (various)
├── test_auth.py                  # Authentication tests
├── test_deployed.py              # Tests against deployed endpoint
├── test_deployed_api.py
├── test_hf_deployment.py
├── test_hackathon_format.py
├── test_8001.py
├── test_detailed.py
├── quick_test.py
├── debug_test.py
├── check_deployment.py           # Deployment health checker
├── verify.py                     # Model verification
├── evaluate_baseline.py          # Baseline evaluation
├── get_base64.py                 # Utility to get Base64 of audio file
│
├── Dockerfile                    # Docker build instructions
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore
│
├── README.md                     # Hugging Face Space README (shown on HF)
├── README_LOCAL.md               # Local development quick-start guide
├── HUGGINGFACE_QUICKSTART.md     # Step-by-step HF deployment guide
├── HF_CLI_FIX.md                 # Fixes for HF CLI on Windows/PowerShell
├── SPACE_URLS.md                 # Deployed Space & API URLs
├── TROUBLESHOOTING_404.md        # Debugging 404 errors after deployment
│
├── base64_output.txt             # Pre-generated Base64 audio (for testing)
└── PROJECT_INFO.md               # ← THIS FILE
```

---

## ⚙ Core Architecture & How It Works

The detection pipeline has **4 steps**, all executed inside the `POST /detect` handler:

```
Client (Base64 audio) ──▶ Decode ──▶ Preprocess ──▶ Model Inference ──▶ Response
```

### Step 1 — Base64 Decode (`app/core/audio.py → decode_base64_audio`)
- Strips optional `data:audio/...;base64,` header.
- Decodes the Base64 string into raw bytes (`io.BytesIO`).

### Step 2 — Audio Preprocessing (`app/core/audio.py → preprocess_audio`)
- Writes bytes to a temporary file (librosa needs a file path).
- Loads & resamples audio to **16 kHz** (required by Wav2Vec2).
- Converts to mono if stereo.
- Normalises the waveform amplitude.
- Cleans up the temp file.

### Step 3 — Model Inference (`app/core/model.py → VoiceDetector.predict`)
- **Singleton pattern** — model is loaded once at startup and reused.
- Uses `mo-thecreator/Deepfake-audio-detection` from Hugging Face Hub.
- Architecture: `AutoModelForAudioClassification` (Wav2Vec2-based).
- Feature extraction → forward pass → softmax probabilities.
- Maps model labels to standardised output:
  - Labels containing "fake" or "spoof" → `AI_GENERATED`
  - Labels containing "real" or "bonafide" → `HUMAN`
- Returns `(label, confidence)`.
- Supports **CUDA GPU** if available, falls back to CPU.

### Step 4 — Explanation Generation (`app/core/explanation.py → generate_explanation`)
- Rule-based explanations that vary by classification + confidence level:
  - **AI_GENERATED** with high confidence → "Strong indicators of synthetic voice generation detected with high confidence"
  - **HUMAN** with high confidence → "Natural human voice characteristics confirmed with high confidence"
  - Lower confidence → more cautious wording.

---

## 🌐 API Endpoints

| Method | Path | Auth Required | Description |
|--------|------|---------------|-------------|
| `GET` | `/` | No | Serves the Web UI (`index.html`), or a JSON health message if UI is missing. |
| `GET` | `/health` | No | Returns API status, version, model name, and auth status. |
| `GET` | `/docs` | No | Auto-generated Swagger/OpenAPI documentation. |
| `POST` | `/detect` | Yes (if configured) | **Main endpoint** — accepts Base64 audio, returns classification. |

---

## 📋 Request / Response Format (Hackathon Spec)

### Request (`POST /detect`)
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "UklGR..."
}
```

| Field | Type | Required | Valid Values |
|-------|------|----------|--------------|
| `language` | string | Yes | `"Tamil"`, `"English"`, `"Hindi"`, `"Malayalam"`, `"Telugu"` |
| `audioFormat` | string | Yes | `"mp3"` |
| `audioBase64` | string | Yes | Base64-encoded MP3 audio data |

### Successful Response
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.95,
  "explanation": "Strong indicators of synthetic voice generation detected with high confidence"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"error"` |
| `language` | string | Echo of the input language |
| `classification` | string | `"AI_GENERATED"` or `"HUMAN"` |
| `confidenceScore` | float | 0.0 to 1.0 |
| `explanation` | string | Human-readable reason |

### Error Response
```json
{
  "status": "error",
  "message": "Invalid Base64 audio: ..."
}
```

---

## 🔐 Authentication & Security

- **API Key Auth** is optional, enabled by setting the `API_KEY` environment variable.
- When enabled, every request to `/detect` must include the header:
  ```
  X-API-Key: <your-api-key>
  ```
- If `API_KEY` is **not set**, the API runs in **development mode** (no auth required).
- **CORS** is fully open (`allow_origins=["*"]`) to support browser-based clients.
- Pre-configured API Key (for the hackathon): `uNaRqJimOAQUK4uL-YRN_DjvHwpGiV8igbhJUUVm3NkY`

### Auth Error Codes
| Code | Meaning |
|------|---------|
| `401` | API key is required but not provided |
| `403` | API key is invalid |

---

## 🔧 Environment Variables

Configured via `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | `None` (auth disabled) | Secret API key for authentication |
| `MODEL_NAME` | `mo-thecreator/Deepfake-audio-detection` | Hugging Face model identifier |
| `SAMPLE_RATE` | `16000` | Audio sample rate for preprocessing |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

---

## 💻 Running Locally

### Prerequisites
- Python 3.9+
- `pip` package manager
- (Optional) Virtual environment

### Steps

```bash
# 1. Clone the repo
git clone <repo-url>
cd voice-detection-api

# 2. Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment
cp .env.example .env
# Edit .env to set API_KEY, etc.

# 5. Run the server
python -m uvicorn app.main:app --reload
# Server starts at http://localhost:8000
```

### Quick Test
```bash
python test_api.py tests/audio_samples/human_voice_test.wav
```

---

## 🚀 Deployment (Hugging Face Spaces)

Hugging Face Spaces is the **primary deployment target** (free Docker-based hosting).

### Step-by-Step

1. **Install HF CLI**: `pip install huggingface_hub`
2. **Get a token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Write access).
3. **Login**: `huggingface-cli login`
4. **Create a Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space)
   - Name: `voice-detection-api`
   - SDK: **Docker**
   - Hardware: **CPU basic** (free)
5. **Add API Key Secret** in Space settings → Repository secrets → `API_KEY`.
6. **Add Git remote & push**:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/voice-detection-api
   git push hf main
   ```
7. **Wait ~10–15 minutes** for the Docker build.
8. API will be live at: `https://YOUR_USERNAME-voice-detection.hf.space`

### Important: README.md
The `README.md` file doubles as the Hugging Face Space configuration (YAML frontmatter with `sdk: docker`, `app_port: 8000`, etc.). The local dev readme is in `README_LOCAL.md`.

---

## 🌐 Deployment (Alternative — Render)

1. Push code to GitHub.
2. Go to [render.com](https://render.com) → Create new Web Service.
3. Connect your GitHub repo.
4. Select **Docker** environment.
5. Deploy.
6. API URL: `https://your-api.onrender.com/detect`

---

## 🧠 Model Training & Fine-Tuning

Two training scripts are provided:

### Local Training (`train_model.py`)
```bash
python train_model.py --dataset_path ./my_dataset --epochs 3 --batch_size 4
```

**Dataset structure:**
```
my_dataset/
├── real/          # Human voice samples
│   ├── audio1.mp3
│   └── audio2.wav
└── fake/          # AI-generated samples
    ├── audio1.mp3
    └── audio2.wav
```

- Fine-tunes the base model (`mo-thecreator/Deepfake-audio-detection`).
- Outputs saved to `./voice_detection_model_finetuned/`.
- Label mapping: `0 → HUMAN`, `1 → AI_GENERATED`.
- Training args: learning rate `3e-5`, gradient accumulation `4`, warmup ratio `0.1`.

### Google Colab Training (`colab_train_script.py`)
- Uses the public dataset `not-lain/deepfake-audio-dataset` from Hugging Face.
- Fine-tunes `facebook/wav2vec2-base`.
- Takes ~15–20 minutes on a free T4 GPU.
- Outputs a zip file for download.

---

## 🧪 Testing the API

Multiple test scripts are available:

| Script | Purpose |
|--------|---------|
| `test_api.py` | General API integration test |
| `test_auth.py` | Authentication / API key tests |
| `test_deployed.py` | Tests against the live deployed API |
| `test_deployed_api.py` | Lightweight deployed API test |
| `test_hf_deployment.py` | Hugging Face-specific deployment test |
| `test_hackathon_format.py` | Validates hackathon-compliant request/response format |
| `test_8001.py` | Tests against port 8001 |
| `test_detailed.py` | Detailed output test |
| `quick_test.py` | Quick smoke test |
| `debug_test.py` | Debug-level test |
| `check_deployment.py` | Deployment health verifier |
| `verify.py` | Model loading verification |
| `evaluate_baseline.py` | Baseline accuracy evaluation |
| `get_base64.py` | Converts an audio file to Base64 (utility) |

### Using the Hackathon Endpoint Tester
1. Get your deployed API URL (e.g., `https://your-api.onrender.com/detect`).
2. Get a public audio file URL.
3. Fill in the tester with your endpoint and audio URL.

---

## 🔍 Troubleshooting Guide

### 404 Error After Deployment
- **Space page URL** ≠ **API endpoint URL**:
  - Space page: `https://huggingface.co/spaces/USERNAME/voice-detection-api`
  - API endpoint: `https://USERNAME-voice-detection.hf.space`
- Make sure you're hitting the **API endpoint**, not the Space page.
- Check the **"App" tab** on HF to see if the app is running.
- Check **logs** for `INFO: Uvicorn running on http://0.0.0.0:8000`.
- Wait a few minutes after build completes for the domain to activate.

### Build Still Running
- Watch the progress bar and real-time logs.
- Status changes from "Building" → "Running".

### Connection Error
- The Space might still be building. Wait and retry.
- Check build logs for errors.

### Quick Checklist
- [ ] Space build completed successfully
- [ ] Logs show "Uvicorn running"
- [ ] "App" tab is accessible
- [ ] API endpoint responds to `GET /`

---

## 🌐 Hugging Face Space URLs

| Purpose | URL |
|---------|-----|
| **Space Management** (logs, settings) | `https://huggingface.co/spaces/Pandaisop/voice-detection-api` |
| **API Endpoint** (actual API calls) | `https://Pandaisop-voice-detection.hf.space` |
| **API Docs** (Swagger UI) | `https://Pandaisop-voice-detection.hf.space/docs` |
| **Health Check** | `https://Pandaisop-voice-detection.hf.space/health` |

---

## 🪟 Hugging Face CLI Fixes (Windows / PowerShell)

PowerShell may not recognise `huggingface-cli` properly. Use these alternatives instead:

| Task | Command |
|------|---------|
| **Login** | `.\venv\Scripts\python -m huggingface_hub.commands.huggingface_cli login` |
| **Create Space** | Use [web interface](https://huggingface.co/new-space) (recommended) |
| **Add Secret** | Use web interface in Space settings |
| **Deploy** | `git push hf main` |

Alternative direct executable call:
```bash
.\venv\Scripts\huggingface-cli.exe login
```

---

## 📎 Key Files Quick Reference

| File | What It Does |
|------|-------------|
| `app/main.py` | FastAPI app definition, all routes, request/response models, error handlers, CORS setup |
| `app/config.py` | Loads environment variables, validates API key presence |
| `app/core/audio.py` | Base64 decoding + audio preprocessing (resample, mono, normalise) |
| `app/core/model.py` | Singleton `VoiceDetector` class — loads HF model, runs inference, maps labels |
| `app/core/explanation.py` | Generates human-readable explanations based on classification + confidence |
| `app/static/index.html` | Browser-based Web UI for recording/uploading audio |
| `Dockerfile` | Docker image build (Python 3.9-slim + libsndfile + ffmpeg) |
| `requirements.txt` | All Python dependencies |
| `.env.example` | Template for environment variables |
| `README.md` | HF Space config (YAML frontmatter) + user-facing docs |
| `train_model.py` | Fine-tuning script for local datasets |
| `colab_train_script.py` | Fine-tuning notebook for Google Colab (free GPU) |

---

> **Created:** 2026-02-10  
> **Project:** VoiceGuard — AI Voice Detection API v2.0  
> **License:** MIT

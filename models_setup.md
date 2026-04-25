# Multimodal Meeting Assistant — Full Local Setup Guide

This guide explains how to set up, run, and test the full project locally.

The system contains three main services:

| Service | Description | Port |
|---|---|---|
| `orchestrator` | Runs the LangGraph pipeline and coordinates everything | `5000` |
| `asr_api` | Speech-to-text + speaker diarization | `5002` |
| `sign_api` | Sign language / gesture recognition | `5001` |

The pipeline flow is:

```txt
Audio + Video (Backend)
   ↓
Orchestrator
   ↓
ASR API + Sign API
   ↓
Validation + Merge
   ↓
LLM via Ollama
   ↓
Final summary + tasks + transcript
```

---

# 0. Before You Begin

## 🌐 Download Size (Internet Usage) (DON'T CURSE ME PLEASE)

Expected total download size:

- ~13–19 GB (using qwen2.5:7b)

This includes:
- Docker images and dependencies
- HuggingFace models (downloaded on first run)
- Ollama model (qwen2.5:7b)

Note:
Final disk usage after setup will be much larger (~40–60 GB).

## Required tools

You need:

- Docker Desktop
- Docker Compose
- NVIDIA GPU drivers
- NVIDIA Container Toolkit / Docker GPU support
- Ollama

## Important notes

- The ASR image is large because it uses CUDA, PyTorch, Transformers, and Pyannote.
- First run is slow because models are loaded/downloaded.
- Ollama must be running on the host machine.
- Keep `LOCAL_PARALLEL=false` unless your machine can handle multiple local LLM requests.
- Your Pyannote key must be valid.
---
# ⚠️ Hardware Requirements (IMPORTANT)

This project supports both GPU and CPU environments, but performance and setup differ.

---

## 🟢 Option 1 — NVIDIA GPU (Recommended)

If you have an NVIDIA GPU:

- Full support with CUDA
- Fast ASR and Sign processing
- Recommended for best performance

Requirements:
- NVIDIA GPU
- NVIDIA drivers installed
- Docker GPU support enabled

---

## 🟡 Option 2 — CPU Only (No NVIDIA GPU)

If you do NOT have an NVIDIA GPU (e.g., Intel CPU / Intel GPU):

- The project **can still run**
- BUT you must use the **CPU version of the Dockerfile**
- Performance will be **much slower**

### What to change

1. Use CPU Dockerfile for ASR:

```yaml
# docker-compose.yml
asr_api:
  build:
    context: ./asr_service
    dockerfile: Dockerfile.asr.cpu
---
```


# 1. Setup the `.env` File

Create a `.env` file in the project root.

Example:

```env
PYANNOTE_API_KEY=your_pyannote_key_here

LLM_MODE=qwen
QWEN_MODEL=qwen2.5:7b
OLLAMA_URL=http://host.docker.internal:11434/api/generate

# Keep false unless the machine is strong
LOCAL_PARALLEL=false

ORCHESTRATOR_PORT=5000
ASR_PORT=5002
SIGN_PORT=5001

ASR_API_URL=http://asr_api:5002/predict
SIGN_API_URL=http://sign_api:5001/predict

HF_CACHE_CONTAINER=/root/.cache/huggingface

# Pick 1
HF_CACHE_PATH=C:/Users/YOUR_USERNAME/hf-cache (For Windows)
HF_CACHE_PATH=/home/YOUR_USERNAME/hf-cache (For Linux)
```
---

# 2. Setup Docker

## 2.1 Install Docker

Download Docker Desktop:

```txt
https://www.docker.com/products/docker-desktop/
```

## 2.2 Verify Docker

Run:

```bash
docker --version
docker compose version
```

Expected: both commands should print versions.

## 2.3 Verify GPU support in Docker

Run:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Expected: it should show your NVIDIA GPU.

If this fails, Docker GPU support is not ready.

Common fixes:
- Update NVIDIA driver
- Enable WSL2 integration in Docker Desktop
- Install NVIDIA Container Toolkit if on Linux
- Restart Docker Desktop

## 2.4 Build Docker images

From the project root:

```bash
docker compose build
```

## 2.5 Run containers

```bash
docker compose up
```

## 2.6 Check running containers (Another terminal)

```bash
docker ps
```

You should see:

```txt
meeting_orchestrator
meeting_asr
meeting_sign
```
---

# 3. Setup Ollama

## 3.1 Install Ollama

Download Ollama:

```txt
https://ollama.com/download
```

## 3.2 Pull Qwen model

```bash
ollama pull qwen2.5:7b
```

## 3.3 Check installed models

```bash
ollama list
```

Expected:

```txt
qwen2.5:7b
```

## 3.4 Start Ollama

Usually Ollama runs automatically in the background.

If needed:

```bash
ollama serve
```

If you get this:

```txt
bind: Only one usage of each socket address is normally permitted
```

That means Ollama is already running.

## 3.5 Test Ollama on Windows PowerShell

Use this PowerShell-safe request:

```powershell
Invoke-RestMethod -Uri "http://localhost:11434/api/generate" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
```

Expected response contains:

```txt
Hello
```
## 3.6 Test Ollama on Linux/macOS

```bash
curl -X POST "http://localhost:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:7b","prompt":"hello","stream":false}'
```

Expected response contains:

```json
"response"
```

## 3.7 If Ollama crashes

If you see:

```txt
llama runner process has terminated
```

Then Ollama crashed due to RAM/VRAM pressure.

Fix: update `.env`:

```env
LOCAL_PARALLEL=false
```

Restart Docker:

```bash
docker compose down
docker compose up
```

---

# 4. Run and Test Each Service

Place test files somewhere easy, for example (Note: audio exists there but add the webcam yourself):

```txt
pipeline_scripts/uploads/audio.webm
pipeline_scripts/uploads/webcam.webm
```

Then open another terminal inside that folder.

Windows:

```powershell
cd "C:\path\to\Multimodal-Meating-Assistant\pipeline_scripts\uploads"
```

Linux/macOS:

```bash
cd /path/to/Multimodal-Meating-Assistant/pipeline_scripts/uploads
```

---

## 4.1 Test Sign API

Windows:

```powershell
curl.exe -X POST "http://localhost:5001/predict" -F "video=@webcam.webm"
```

Linux/macOS:

```bash
curl -X POST "http://localhost:5001/predict" -F "video=@webcam.webm"
```

Expected response:

```json
{
  "sign": "...",
  "confidence": 0.9,
  "top10": []
}
```

If this works, Sign API is good.

---

## 4.2 Test ASR API

Windows:

```powershell
curl.exe -X POST "http://localhost:5002/predict" -F "audio=@audio.webm" -F "language=mix"
```

Linux/macOS:

```bash
curl -X POST "http://localhost:5002/predict" -F "audio=@audio.webm" -F "language=mix"
```

Expected response:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.445,
      "end": 3.525,
      "text": "..."
    }
  ]
}
```

If this works, ASR API is good.

---

## 4.3 Test Full Pipeline

Windows:

```powershell
curl.exe -X POST "http://localhost:5000/process" -F "audio=@audio.webm" -F "video=@webcam.webm" -F "language=mix"
```

Linux/macOS:

```bash
curl -X POST "http://localhost:5000/process" -F "audio=@audio.webm" -F "video=@webcam.webm" -F "language=mix"
```

Expected response:

```json
{
  "summary": {
    "text": "..."
  },
  "tasks": {
    "tasks": []
  },
  "name_recognition": {
    "mappings": []
  },
  "transcription": [],
  "full_transcript_text": "..."
}
```

If this works, the full system is running correctly.

---

# 5. Recommended Testing Order

Always test in this order:

```txt
1. Docker GPU support
2. Ollama alone
3. Sign API
4. ASR API
5. Full Orchestrator pipeline
```

Do not test the full pipeline first.

---

# 6. Useful Docker Commands

Show images:

```bash
docker images
```

Show disk usage:

```bash
docker system df
```

Remove unused build cache:

```bash
docker builder prune -a
```

---

# 7. Final Checklist

Before saying the setup works, confirm:

- [ ] Docker is installed
- [ ] Docker Compose works
- [ ] GPU works inside Docker
- [ ] `.env` exists
- [ ] Pyannote key is valid
- [ ] Ollama is installed
- [ ] Qwen model is pulled
- [ ] Ollama test works
- [ ] `docker compose build` succeeds
- [ ] `docker compose up` runs all containers
- [ ] Sign API test works
- [ ] ASR API test works
- [ ] Full pipeline test works

---

# 8. Expected Final Output

The final pipeline should return:

```json
{
  "summary": {
    "text": "meeting summary"
  },
  "tasks": {
    "tasks": [
      {
        "task_name": "task",
        "assignee": "SPEAKER_01",
        "assigned_by": "SPEAKER_00",
        "deadline": "today"
      }
    ]
  },
  "name_recognition": {
    "mappings": []
  },
  "transcription": [],
  "full_transcript_text": "..."
}
```

If this appears, the project is working.

# Multimodal Meeting Assistant — Setup Guide

## Prerequisites

Make sure you have all of these installed before starting:

- [Git](https://git-scm.com/)
- [Node.js](https://nodejs.org/) (v18 or higher)
- [PostgreSQL](https://www.postgresql.org/download/) (running on port `5432`)
- [Docker](https://www.docker.com/products/docker-desktop)
- Docker Compose (included with Docker Desktop)

Also Make sure to have about **45 GBs** free on your disk for safety

---

## 1. Clone the Repository

```bash
git clone https://github.com/jowayria-aladdin/Multimodal-Meeting-Assistant.git
cd Multimodal-Meeting-Assistant
git checkout fully_integrated
```

---

## 2. Set Up the Database

Make sure PostgreSQL is running, then create a new database:

```sql
CREATE DATABASE meeting_assistant;
```

You can do this via `psql`, pgAdmin, or any PostgreSQL client.

---

## 3. Configure Environment Variables

You need to create `.env` files in three places: `backend/`, `frontend/`, and the root (for the orchestrator/Docker).

### `backend/.env`

```env
NODE_ENV=development
PORT=3000

DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/meeting_assistant

JWT_SECRET=your_jwt_secret
JWT_EXPIRES_IN=7d

CORS_ORIGIN=*
INTERNAL_CALLBACK_SECRET=your_internal_secret

CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
CLOUDINARY_API_KEY=your_cloudinary_api_key
CLOUDINARY_API_SECRET=your_cloudinary_api_secret
```

### `frontend/.env`

```env
NEXT_PUBLIC_CLOUDINARY_CLOUD_NAME=your_cloudinary_cloud_name
NEXT_PUBLIC_CLOUDINARY_PRESET=your_cloudinary_upload_preset
```

> **Note:** The Cloudinary upload preset must be set to **Unsigned** in your Cloudinary dashboard.
> Go to: Cloudinary Dashboard → Settings → Upload → Upload Presets (Or simply edit ml_default preset to Unsigned)

### Root `.env` (for Docker / Orchestrator)

Create a `.env` file in the root of the project:

```env
PYANNOTE_API_KEY=your_pyannote_key

LLM_MODE=gemini
GEMINI_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your_gemini_api_key

QWEN_MODEL=qwen2.5:7b
OLLAMA_URL=http://host.docker.internal:11434/api/generate

# true for parallel LLM tasks (THE DEVICE WILL EXPLODE)
LOCAL_PARALLEL=false

ORCHESTRATOR_PORT=5000
ASR_PORT=5002
SIGN_PORT=5001

ASR_API_URL=http://asr_api:5002/predict
SIGN_API_URL=http://sign_api:5001/predict

HF_CACHE_PATH=C:/Users/YOUR_USERNAME/hf-cache
HF_CACHE_CONTAINER=/root/.cache/huggingface

INTERNAL_SECRET=your_internal_secret
BACKEND_URL=http://host.docker.internal:3000
```

> **Important:** `INTERNAL_SECRET` must match `INTERNAL_CALLBACK_SECRET` in `backend/.env`.

> **Windows users:** Set `HF_CACHE_PATH` to a folder on your machine (e.g. `C:/Users/YourName/hf-cache`).
> **Mac/Linux users:** Set it to something like `/Users/yourname/hf-cache` or `~/.cache/hf-cache`.

---

## 4. Set Up the Backend

```bash
cd backend
npm install
npx prisma migrate dev
npx prisma generate
npm run dev
```

The backend will start on `http://localhost:3000`.

---

## 5. Set Up the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:3001`.

---

## 6. Start the AI Pipeline (Docker)

Open a new terminal in the root of the project:

```bash
docker compose up --build
```

This starts three containers:
- **Orchestrator** — coordinates the full pipeline (port `5000`)
- **ASR Service** — speech-to-text (port `5002`)
- **Sign Language Service** — sign language recognition (port `5001`)

> The first build will take a while as it downloads AI models. Subsequent starts will be faster.

---

## 7. Install the Chrome Extension

1. Open the app at `http://localhost:3001`
2. On the home/landing page, download the Chrome extension
3. Open Chrome → go to `chrome://extensions/`
4. Enable **Developer Mode** (top right)
5. Click **Load unpacked** and select the extension folder
6. Grant the required permissions when prompted

---

## 8. How to Use

1. Open `http://localhost:3001` and sign up / sign in
2. Create a workspace
3. Click **Upload Recording**
4. Upload your files:
   - **Raw Audio (.wav or .webm)** — required
   - **Main Meeting Video (.webm or .mp4)** — required
   - **Sign Language Video (.webm)** — optional
5. Click **Process Recording** and wait for the AI to finish
6. View your transcript, summary, and extracted tasks in the meeting page

---

## Running Everything Together

You need **three terminals** running simultaneously:

| Terminal | Command | Directory |
|---|---|---|
| 1 | `npm run dev` | `backend/` |
| 2 | `npm run dev` | `frontend/` |
| 3 | `docker compose up` | root |

---

## Troubleshooting

**Gemini 429 error**
Create a new project in gemini's api keys page then create a new api key.

**Gemini 503 error**
The Gemini API is temporarily unavailable. Wait a few minutes and retry.

**Cloudinary upload fails**
Make sure your upload preset is set to **Unsigned** and that `NEXT_PUBLIC_CLOUDINARY_PRESET` matches the preset name exactly.

**Docker build fails**
Make sure Docker Desktop is running and you have enough disk space for the AI models (~5GB+).

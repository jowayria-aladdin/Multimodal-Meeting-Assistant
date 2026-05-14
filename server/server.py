from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Header, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import sys
import anyio
import httpx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline_scripts.langgraph_pipeline import build_graph

app = FastAPI(title="Multimodal Meeting Assistant API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

INTERNAL_SECRET = os.environ.get("INTERNAL_SECRET", "changeme")
BACKEND_URL     = os.environ.get("BACKEND_URL", "http://localhost:3000")

# ── helper: send callback to backend ─────────────────────────────────────────
async def send_callback(meeting_id: str, payload: dict):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            await client.post(
                f"{BACKEND_URL}/api/internal/meetings/{meeting_id}/callback",
                json=payload,
                headers={"X-Internal-Secret": INTERNAL_SECRET}
            )
        except Exception as e:
            print(f"Callback error: {e}")

# ── background task: run pipeline and send callbacks ─────────────────────────
async def run_pipeline(
    meeting_id: str,
    wav_bytes: bytes,
    sign_bytes: bytes,
    language: str,
    wav_suffix: str,
    sign_suffix: str
):
    wav_path  = None
    sign_path = None

    try:
        # save files to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=wav_suffix) as f:
            f.write(wav_bytes)
            wav_path = f.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=sign_suffix) as f:
            f.write(sign_bytes)
            sign_path = f.name

        # progress: starting
        await send_callback(meeting_id, {
            "status":   "PROCESSING",
            "progress": 10,
            "stage":    "validate_input",
            "message":  "Starting processing"
        })

        # progress: transcribing
        await send_callback(meeting_id, {
            "status":   "PROCESSING",
            "progress": 30,
            "stage":    "transcribe",
            "message":  "Transcribing audio"
        })

        # run langgraph pipeline
        result = await anyio.to_thread.run_sync(
            lambda: graph.invoke({
                "audio_path":  wav_path,
                "webcam_path": sign_path,
                "screen_path": sign_path,
                "language":    language
            })
        )

        # progress: summarizing
        await send_callback(meeting_id, {
            "status":   "PROCESSING",
            "progress": 80,
            "stage":    "summarize",
            "message":  "Generating summary"
        })

        final = result.get("final_output", {})

        # format tasks to match backend expectation
        raw_tasks = final.get("tasks", {}).get("tasks", [])
        tasks = [
            {
                "task_text": t.get("task_name", t.get("task_text", "")),
                "due_date":  t.get("deadline", None),
                "status":    "TODO"
            }
            for t in raw_tasks
        ]

        # send COMPLETED callback
        await send_callback(meeting_id, {
            "status":   "COMPLETED",
            "progress": 100,
            "stage":    "completed",
            "message":  "Processing finished",
            "result": {
                "transcript": final.get("transcription", []),
                "summary":    final.get("summary", {}).get("text", ""),
                "tasks":      tasks
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        await send_callback(meeting_id, {
            "status":   "FAILED",
            "progress": 100,
            "stage":    "failed",
            "message":  "Processing failed",
            "error":    str(e)
        })

    finally:
        for path in [wav_path, sign_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

########### endpoints  ###########

@app.get("/health")
def health():
    return {"status": "ok"}

# new endpoint — called by the backend 
@app.post("/process-audio")
async def process_audio(
    background_tasks: BackgroundTasks,
    meetingId:  str        = Form(...),
    companyId:  str        = Form(...),
    title:      str        = Form(...),
    language:       str        = Form(...),
    audio:    UploadFile = File(...),
    video:  UploadFile = File(...),
    x_internal_secret: str = Header(None)
):
    # validate secret
    if x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # validate lang
    if language not in ["en", "ar", "cs"]:
        raise HTTPException(status_code=400, detail="lang must be en, ar, or cs")

    # read files into memory immediately before background task
    wav_bytes   = await audio.read()
    sign_bytes  = await video.read()
    wav_suffix  = os.path.splitext(audio.filename)[-1] or ".wav"
    sign_suffix = os.path.splitext(video.filename)[-1] or ".webm"

    # acknowledge immediately
    background_tasks.add_task(
        run_pipeline,
        meetingId, wav_bytes, sign_bytes,
        language, wav_suffix, sign_suffix
    )

    return {"status": "QUEUED", "meetingId": meetingId}

# old endpoint — keep for direct testing
@app.post("/process")
async def process(
    audio: UploadFile = File(...),
    video: UploadFile = File(...),
    language: str = "mix"
):
    audio_path = None
    video_path = None

    try:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_audio:
            tmp_audio.write(await audio.read())
            audio_path = tmp_audio.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_video:
            tmp_video.write(await video.read())
            video_path = tmp_video.name

        # Run pipeline
        result = await anyio.to_thread.run_sync(
            lambda: graph.invoke({
                "audio_path": audio_path,
                "webcam_path": video_path,
                "screen_path": video_path,
                "language": language
            })
        )

        return result["final_output"]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [audio_path, video_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
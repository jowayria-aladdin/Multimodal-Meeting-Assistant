from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import sys
import anyio
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

@app.get("/health")
def health():
    return {"status": "ok"}

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
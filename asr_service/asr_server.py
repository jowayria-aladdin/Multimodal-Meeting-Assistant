from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import json

from asr_runner import run_asr_pipeline, load_asr_resources
from media_utils import convert_webm_to_wav
import sys
sys.stdout.reconfigure(line_buffering=True)

app = FastAPI(title="ASR API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ASR_RESOURCES = None


def get_asr_resources():
    global ASR_RESOURCES

    if ASR_RESOURCES is None:
        ASR_RESOURCES = load_asr_resources()

    return ASR_RESOURCES


@app.get("/health")
def health():
    print("[ASR API] /health HIT", flush=True)
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    language: str = Form("mix")
):
    tmp_input = None
    tmp_wav = None
    output_json = None

    try:
        suffix = os.path.splitext(audio.filename or "")[-1].lower() or ".webm"
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_input = tmp.name

        if suffix == ".wav":
            tmp_wav = tmp_input
            print("[ASR API] Input is already WAV. Skipping conversion.", flush=True)
        else:
            tmp_wav = tmp_input.rsplit(".", 1)[0] + ".wav"
            convert_webm_to_wav(tmp_input, tmp_wav)
            print("[ASR API] Conversion to WAV is done.", flush=True)

        if not os.path.exists(tmp_wav):
            raise RuntimeError(f"WAV file was not created/found: {tmp_wav}")

        output_json = tmp_wav.rsplit(".", 1)[0] + ".json"
        resources = get_asr_resources()
        run_asr_pipeline(
            audio_path=tmp_wav,
            output_path=output_json,
            lang=language,
            resources=resources,
        )

        if not os.path.exists(output_json):
            raise RuntimeError(f"ASR output JSON was not created: {output_json}")

        with open(output_json, "r", encoding="utf-8") as f:
            result = json.load(f)

        return {"segments": result}

    except Exception as e:
        import traceback
        print("[ASR API] Exception during /predict:", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [output_json]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    print(f"[ASR API] Failed to remove {path}: {cleanup_error}", flush=True)

        if tmp_wav and tmp_wav != tmp_input and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception as cleanup_error:
                print(f"[ASR API] Failed to remove {tmp_wav}: {cleanup_error}", flush=True)

        if tmp_input and os.path.exists(tmp_input):
            try:
                os.remove(tmp_input)
            except Exception as cleanup_error:
                print(f"[ASR API] Failed to remove {tmp_input}: {cleanup_error}", flush=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
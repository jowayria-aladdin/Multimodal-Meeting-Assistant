import os
import json
import os
import shutil
import ffmpeg
import torch
import torchaudio
from pathlib import Path
from pyannote.audio import Pipeline
from transformers import pipeline
from dotenv import load_dotenv


LANGUAGE_PROMPTS = {
    "en": "Transcribe this audio into English ONLY.",
    "ar": "Transcribe this audio in Egyptian Arabic Dialect only.",
    "mix": "Transcribe this audio exactly as spoken, preserving code-switching between Arabic and English. Do not rephrase, correct, or translate anything.",
}

MODEL_LANGUAGE = {
    "en" : "english",
    "ar" : "arabic",
    "mix" : ""
}

lang = input("Enter audio language (en / ar / mix): ").strip().lower()

while lang not in LANGUAGE_PROMPTS:
    lang = input("Invalid choice. Enter audio language (en / ar / mix): ").strip().lower()

PROMPT = LANGUAGE_PROMPTS[lang]
LANGUAGE = MODEL_LANGUAGE[lang]

print(f"Selected language mode: {lang}")
print(f"Using prompt: {PROMPT}")


AUDIO_PATH = "../data/raw/audio/preprocessed_audios/english_cleaned.wav"
SEGMENTS_DIR = "segments"
OUTPUT_DIR = "raw_output"
MODEL_ID = "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition"
PYANNOTE_MODEL = "pyannote/speaker-diarization-precision-2"

load_dotenv(dotenv_path=Path("../.env"))
PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
if not PYANNOTE_API_KEY:
    raise RuntimeError(
        "Missing PYANNOTE_API_KEY environment variable. "
        "Set it before running the script."
    )

Path(SEGMENTS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# Device setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Using device: {device}")


# =========================
# Helpers
# =========================
def clean_segments(raw_segments, gap_threshold=0.5, min_duration=0.4):
    merged = []

    for seg in raw_segments:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            if seg["start"] - merged[-1]["end"] <= gap_threshold:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())
        else:
            merged.append(seg.copy())

    cleaned = [
        s for s in merged
        if (s["end"] - s["start"]) >= min_duration
    ]
    return cleaned


def cut_segments(audio_path, segments, out_dir):
    for i, seg in enumerate(segments):
        out_path = os.path.join(out_dir, f"seg_{i}_{seg['speaker']}.wav")
        (
            ffmpeg
            .input(audio_path, ss=seg["start"], to=seg["end"])
            .output(out_path, ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        seg["audio_path"] = out_path


def prepare_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform.squeeze().numpy()


# =========================
# 1) Speaker diarization
# =========================
print("Loading pyannote diarization pipeline...")
pipeline_diarization = Pipeline.from_pretrained(
    PYANNOTE_MODEL,
    token=PYANNOTE_API_KEY,
)

# precision-2 is a remote service, but .to(device) is harmless to skip here.
# For community/local pipelines, you would usually move the model to GPU.

print("Running diarization...")
waveform, sample_rate = torchaudio.load(AUDIO_PATH)
diarization = pipeline_diarization({
    "waveform": waveform,
    "sample_rate": sample_rate,
})

segments = []
for turn, speaker in diarization.speaker_diarization:
    segments.append({
        "speaker": speaker,
        "start": float(turn.start),
        "end": float(turn.end),
    })

print(f"Raw segments: {len(segments)}")
segments = clean_segments(segments)
print(f"Cleaned segments: {len(segments)}")

print("Saving diarized segments...")
cut_segments(AUDIO_PATH, segments, SEGMENTS_DIR)
print("Segments saved.")


# =========================
# 2) ASR pipeline
# =========================
print("Loading ASR pipeline...")
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

tokenizer = asr.tokenizer
prompt_ids = tokenizer.get_prompt_ids(PROMPT, return_tensors="pt").to(device)

print("Running transcription...")
diarized_transcript = []

generate_kwargs = {
    "temperature": 0,
    "task": "transcribe",
    "num_beams": 5,
    "prompt_ids": prompt_ids,
}

if LANGUAGE:
    generate_kwargs["language"] = LANGUAGE

for seg in segments:
    audio_array = prepare_audio(seg["audio_path"])

    result = asr(
        audio_array,
        generate_kwargs=generate_kwargs,
    )

    diarized_transcript.append({
        "speaker": seg["speaker"],
        "start": seg["start"],
        "end": seg["end"],
        "text": result["text"].strip(),
    })

print("Transcription complete.")

output_file = os.path.join(OUTPUT_DIR, "1st_pipeline.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(diarized_transcript, f, indent=4, ensure_ascii=False)

print(f"Saved: {output_file}")

choice = input("Do you want to delete the segments folder? (y/n): ").strip().lower()

if choice == "y":
    if os.path.exists(SEGMENTS_DIR):
        shutil.rmtree(SEGMENTS_DIR)
        print("Segments folder deleted.")
    else:
        print("Segments folder not found.")
else:
    print("Segments folder kept.")

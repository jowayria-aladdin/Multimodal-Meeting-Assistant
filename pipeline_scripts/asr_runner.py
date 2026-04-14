import os
import json
import shutil
import ffmpeg
import torch
import torchaudio
from pathlib import Path
from pyannote.audio import Pipeline
from transformers import pipeline
from dotenv import load_dotenv


# =========================
# Language Config
# =========================
LANGUAGE_PROMPTS = { 
    "en": "Transcribe the speech exactly as spoken in English only.",
    "ar": "Transcribe the speech exactly as spoken in Arabic with Egyptian Dialect only.",
    "mix": "Transcribe the speech exactly as spoken, preserving Arabic-English code-switching."
}

MODEL_LANGUAGE = {
    "en": "english",
    "ar": "arabic",
    "mix": ""
}
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
# MAIN FUNCTION
# =========================
def run_asr_pipeline(
    audio_path: str,
    output_path: str,
    lang: str,
    segments_dir: str = "segments",
    cleanup_segments: bool = True
):
    # Validate language
    if lang not in LANGUAGE_PROMPTS:
        raise ValueError(f"Invalid language: {lang}. Must be one of {list(LANGUAGE_PROMPTS.keys())}")

    PROMPT = LANGUAGE_PROMPTS[lang]
    LANGUAGE = MODEL_LANGUAGE[lang]

    print(f"[ASR] Language: {lang}")
    print(f"[ASR] Prompt: {PROMPT}")

    # Setup dirs
    Path(segments_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    # Load env
    load_dotenv()
    PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")
    if not PYANNOTE_API_KEY:
        raise RuntimeError("Missing PYANNOTE_API_KEY")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[ASR] Using device: {device}")

    # =========================
    # 1) Diarization
    # =========================
    print("[ASR] Loading diarization model...")
    pipeline_diarization = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-precision-2",
        token=PYANNOTE_API_KEY,
    )

    print("[ASR] Running diarization...")
    waveform, sample_rate = torchaudio.load(audio_path)

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

    print(f"[ASR] Raw segments: {len(segments)}")

    segments = clean_segments(segments)
    print(f"[ASR] Cleaned segments: {len(segments)}")

    # Cut segments
    cut_segments(audio_path, segments, segments_dir)

    # =========================
    # 2) ASR
    # =========================
    print("[ASR] Loading ASR model...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )

    tokenizer = asr.tokenizer
    prompt_ids = tokenizer.get_prompt_ids(PROMPT, return_tensors="pt").to(device)

    generate_kwargs = {
        "temperature": 0,
        "task": "transcribe",
        "num_beams": 5,
        "prompt_ids": prompt_ids,
    }

    if LANGUAGE:
        generate_kwargs["language"] = LANGUAGE

    print("[ASR] Running transcription...")

    diarized_transcript = []

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

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diarized_transcript, f, indent=4, ensure_ascii=False)

    print(f"[ASR] Saved: {output_path}")

    # Cleanup
    if cleanup_segments and os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)
        print("[ASR] Segments cleaned up")

    return output_path
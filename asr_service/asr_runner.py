import os
import json
import shutil
from pathlib import Path

import ffmpeg
import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(line_buffering=True)


LANGUAGE_PROMPTS = {
    "en": "Transcribe the speech exactly as spoken in English only.",
    "ar": "Transcribe the speech exactly as spoken in Arabic with Egyptian Dialect only.",
    "mix": "Transcribe the speech exactly as spoken, preserving Arabic-English code-switching.",
    "cs": "Transcribe the speech exactly as spoken, preserving Arabic-English code-switching.",
}

MODEL_LANGUAGE = {
    "en": "english",
    "ar": "arabic",
    "mix": "",
    "cs": "",
}


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
    os.makedirs(out_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        if seg["end"] <= seg["start"]:
            continue

        duration = seg["end"] - seg["start"]
        out_path = os.path.join(out_dir, f"seg_{i}_{seg['speaker']}.wav")

        try:
            (
                ffmpeg
                .input(audio_path, ss=seg["start"], t=duration)
                .output(out_path, ac=1, ar=16000)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            print("[ASR] ffmpeg stdout:")
            print(e.stdout.decode("utf-8", errors="ignore") if e.stdout else "")
            print("[ASR] ffmpeg stderr:")
            print(e.stderr.decode("utf-8", errors="ignore") if e.stderr else "")
            raise

        seg["audio_path"] = out_path


def prepare_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform.squeeze().numpy()


def load_asr_resources():
    load_dotenv(override=True)

    pyannote_api_key = os.getenv("PYANNOTE_API_KEY")
    if not pyannote_api_key:
        raise RuntimeError("Missing PYANNOTE_API_KEY")

    has_cuda = torch.cuda.is_available()
    hf_device = 0 if has_cuda else -1
    model_dtype = torch.float16 if has_cuda else torch.float32
    device_name = "cuda" if has_cuda else "cpu"


    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-precision-2",
        token=pyannote_api_key,
    )
    
    print("[ASR] Diarization model loaded.", flush=True)

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        dtype=model_dtype,
        device=hf_device,
        return_timestamps=True,
    )

    print("[ASR] ASR model loaded.", flush=True)

    return {
        "diarization_pipeline": diarization_pipeline,
        "asr_pipeline": asr_pipeline,
        "torch_device": device_name,
    }


def run_asr_pipeline(
    audio_path: str,
    output_path: str,
    lang: str,
    resources: dict,
    segments_dir: str = "segments",
    cleanup_segments: bool = True
):
    if lang not in LANGUAGE_PROMPTS:
        raise ValueError(f"Invalid language: {lang}. Must be one of {list(LANGUAGE_PROMPTS.keys())}")

    prompt = LANGUAGE_PROMPTS[lang]
    language = MODEL_LANGUAGE[lang]

    diarization_pipeline = resources["diarization_pipeline"]
    asr_pipeline = resources["asr_pipeline"]
    torch_device = resources["torch_device"]

    Path(segments_dir).mkdir(parents=True, exist_ok=True)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("[ASR] Running diarization...", flush=True)
    waveform, sample_rate = torchaudio.load(audio_path)

    try:
        diarization = diarization_pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate,
        })
    except Exception as e:
        import traceback
        print("[ASR] Diarization failed:", flush=True)
        traceback.print_exc()
        raise

    segments = []
    for turn, speaker in diarization.speaker_diarization:
        segments.append({
            "speaker": speaker,
            "start": float(turn.start),
            "end": float(turn.end),
        })

    segments = clean_segments(segments)

    print("[ASR] Diarization Finished.", flush=True)

    cut_segments(audio_path, segments, segments_dir)

    tokenizer = asr_pipeline.tokenizer
    prompt_ids = tokenizer.get_prompt_ids(prompt, return_tensors="pt")

    if torch_device == "cuda":
        prompt_ids = prompt_ids.to("cuda")

    generate_kwargs = {
        "temperature": 0,
        "task": "transcribe",
        "num_beams": 5,
        "prompt_ids": prompt_ids,
    }

    if language:
        generate_kwargs["language"] = language

    print("[ASR] Running transcription...", flush=True)

    diarized_transcript = []

    for i, seg in enumerate(segments, start=1):
        if "audio_path" not in seg:
            continue

        audio_array = prepare_audio(seg["audio_path"])

        result = asr_pipeline(
            audio_array,
            generate_kwargs=generate_kwargs,
        )

        diarized_transcript.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "text": result["text"].strip(),
        })

    print("[ASR] Transcription Finished.", flush=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diarized_transcript, f, indent=4, ensure_ascii=False)

    if cleanup_segments and os.path.exists(segments_dir):
        shutil.rmtree(segments_dir)

    return output_path
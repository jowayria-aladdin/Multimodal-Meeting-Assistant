from typing import TypedDict, Optional
import json
import os
from langgraph.graph import StateGraph, START, END

# =========================
# STATE
# =========================
class PipelineState(TypedDict, total=False):
    audio_path: str
    webcam_path: str
    screen_path: str
    language: str

    audio_json: Optional[str]
    video_json: Optional[str]

    validated_audio_json: Optional[str]
    validated_video_json: Optional[str]

    merged_json: Optional[str]
    validated_merged_json: Optional[str]

    llm_input_json: Optional[str]

    summary_path: Optional[str]
    tasks_path: Optional[str]

    final_output: Optional[dict]
    error: Optional[str]

# =========================
# NODES
# =========================

# 1) ASR
def run_asr_node(state):
    from asr_runner import run_asr_pipeline
    from media_utils import convert_webm_to_wav

    input_audio = state["audio_path"]

    wav_audio = "./uploads/audio_converted.wav"

    print("[ASR] Converting input → WAV...")
    convert_webm_to_wav(input_audio, wav_audio)

    output = "./raw_output/audio_raw.json"

    run_asr_pipeline(
        audio_path=wav_audio,
        output_path=output,
        lang=state["language"]
    )

    return {"audio_json": output}

# 2) SIGN LANGUAGE (TEMP DUMMY)
def run_sign_node(state: PipelineState):
    output = "./raw_output/video_raw.json"

    dummy = [
        {
            "speaker": "SIGN_00",
            "start": 1.0,
            "end": 3.0,
            "text": "dummy sign language output"
        }
    ]

    os.makedirs("./raw_output", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(dummy, f, indent=4)

    return {"video_json": output}

# 3) VALIDATE AUDIO
def validate_audio_node(state: PipelineState):
    from pydantic_validation import validate_json_file

    output = "./validated_output/validated_audio.json"
    validate_json_file(state["audio_json"], output)

    return {"validated_audio_json": output}

# 4) VALIDATE VIDEO
def validate_video_node(state: PipelineState):
    from pydantic_validation import validate_json_file

    output = "./validated_output/validated_video.json"
    validate_json_file(state["video_json"], output)

    return {"validated_video_json": output}

# 5) MERGE
def merge_node(state: PipelineState):
    from merge_JSON import merge_jsons

    output = "./validated_output/merged.json"

    merge_jsons(
        state["validated_audio_json"],
        state["validated_video_json"],
        output
    )

    return {"merged_json": output}

# 6) VALIDATE MERGED
def validate_merged_node(state: PipelineState):
    from pydantic_validation_merged import validate_merged_json_file

    output = "./validated_output/validated_merged.json"

    validate_merged_json_file(state["merged_json"], output)

    return {"validated_merged_json": output}

# 7) SIMPLIFY
def simplify_node(state: PipelineState):
    from simplify_json import simplify_json

    output = "./llm_input/llm_input.json"

    simplify_json(
        state["validated_merged_json"],
        output
    )

    return {"llm_input_json": output}

# 8) LLM (PLACEHOLDER)
def llm_node(state: PipelineState):
    # simulate teammate output for now

    os.makedirs("./llm_output", exist_ok=True)

    summary_path = "./llm_output/summary.txt"
    tasks_path = "./llm_output/executive_tasks.json"

    with open(summary_path, "w") as f:
        f.write("This is a dummy summary.")

    with open(tasks_path, "w") as f:
        json.dump(
            [{"task": "dummy task"}],
            f,
            indent=4
        )

    return {
        "summary_path": summary_path,
        "tasks_path": tasks_path
    }

# 9) FINAL OUTPUT
def final_node(state: PipelineState):
    with open(state["summary_path"], "r") as f:
        summary = f.read()

    with open(state["tasks_path"], "r") as f:
        tasks = json.load(f)

    with open(state["validated_audio_json"], "r", encoding="utf-8") as f:
        transcription = json.load(f)

    full_transcript_text = "\n".join(
        f"{seg['speaker']}: {seg['text']}" for seg in transcription
    )

    final = {
        "summary": summary,
        "tasks": tasks,
        "transcription": transcription,
        "full_transcript_text": full_transcript_text
    }

    return {"final_output": final}

# =========================
# GRAPH
# =========================

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("asr", run_asr_node)
    graph.add_node("sign", run_sign_node)
    graph.add_node("validate_audio", validate_audio_node)
    graph.add_node("validate_video", validate_video_node)
    graph.add_node("merge", merge_node)
    graph.add_node("validate_merged", validate_merged_node)
    graph.add_node("simplify", simplify_node)
    graph.add_node("llm", llm_node)
    graph.add_node("final", final_node)

    # fan-out
    graph.add_edge(START, "asr")
    graph.add_edge(START, "sign")

    # branch 1
    graph.add_edge("asr", "validate_audio")

    # branch 2
    graph.add_edge("sign", "validate_video")

    # fan-in
    graph.add_edge(["validate_audio", "validate_video"], "merge")

    # continue normally
    graph.add_edge("merge", "validate_merged")
    graph.add_edge("validate_merged", "simplify")
    graph.add_edge("simplify", "llm")
    graph.add_edge("llm", "final")
    graph.add_edge("final", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "audio_path": "./uploads/audio.wav",
        "webcam_path": "./uploads/webcam.webm",
        "screen_path": "./uploads/screen.webm",
        "language": "mix"
    })

    os.makedirs("./final_output", exist_ok=True)

    output_path = "./final_output/final_output.json"
    print("\n===== FINAL OUTPUT =====")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result["final_output"], f, indent=4, ensure_ascii=False)

    print(f"\n[FINAL] Saved to: {output_path}")
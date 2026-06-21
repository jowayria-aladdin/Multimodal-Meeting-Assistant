from typing import TypedDict, Optional
import json
import os
import httpx
from langgraph.graph import StateGraph, START, END
import sys
sys.stdout.reconfigure(line_buffering=True)

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
    llm_output_path: Optional[str]
    full_segments_path: Optional[str]

    final_output: Optional[dict]
    error: Optional[str]

# =========================
# NODES
# =========================

def run_asr_node(state):
    ASR_API_URL = os.getenv("ASR_API_URL")
    with open(state["audio_path"], "rb") as f:
        response = httpx.post(
            ASR_API_URL,
            files={"audio": f},
            data={"language": state["language"]},
            timeout=None
        )

    response.raise_for_status()

    output = "./raw_output/audio_raw.json"
    os.makedirs("./raw_output", exist_ok=True)

    with open(output, "w", encoding="utf-8") as out:
        json.dump(response.json()["segments"], out, indent=4)

    return {"audio_json": output}


def run_sign_node(state):
    if not state.get("webcam_path"):
        return {"video_json": None}

    SIGN_API_URL = os.getenv("SIGN_API_URL")

    with open(state["webcam_path"], "rb") as f:
        response = httpx.post(
            SIGN_API_URL,
            files={"video": f},
            timeout=None
        )

    response.raise_for_status()
    sign_segments = response.json()["sign"]

    output = "./raw_output/video_raw.json"
    os.makedirs("./raw_output", exist_ok=True)

    with open(output, "w", encoding="utf-8") as out:
        json.dump(sign_segments, out, indent=4)

    return {"video_json": output}


def validate_audio_node(state):
    from pipeline_scripts.pydantic_validation import validate_json_file
    output = "./validated_output/validated_audio.json"
    validate_json_file(state["audio_json"], output)
    return {"validated_audio_json": output}


def validate_video_node(state):
    if not state.get("video_json"):
        return {"validated_video_json": None}
    
    from pipeline_scripts.pydantic_validation import validate_json_file
    output = "./validated_output/validated_video.json"
    validate_json_file(state["video_json"], output)
    return {"validated_video_json": output}


def merge_node(state):
    import shutil
    from pipeline_scripts.merge_JSON import merge_jsons
    output = "./validated_output/merged.json"

    if not state.get("validated_video_json"):
        shutil.copy(state["validated_audio_json"], output)
    else:
        merge_jsons(
            state["validated_audio_json"],
            state["validated_video_json"],
            output
        )

    return {"merged_json": output}


def validate_merged_node(state):
    from pipeline_scripts.pydantic_validation_merged import validate_merged_json_file
    output = "./validated_output/validated_merged.json"

    input_path = state.get("merged_json") or state.get("validated_merged_json")
    validate_merged_json_file(input_path, output)
    return {"validated_merged_json": output}


def simplify_node(state):
    from pipeline_scripts.simplify_json import simplify_json
    output = "./llm_input/llm_input.json"

    simplify_json(
        state["validated_merged_json"],
        output
    )

    return {"llm_input_json": output}


def llm_node(state):
    from pipeline_scripts.llm_pipeline import run_llm_pipeline

    os.makedirs("./llm_output", exist_ok=True)
    output = "./llm_output/llm_results.json"

    run_llm_pipeline(
        input_path=state["llm_input_json"],
        output_path=output,
        language=state["language"]
    )

    return {
        "llm_output_path": output,
        "full_segments_path": state["validated_merged_json"]
    }


def final_node(state):
    with open(state["llm_output_path"], "r", encoding="utf-8") as f:
        llm_output = json.load(f)

    with open(state["full_segments_path"], "r", encoding="utf-8") as f:
        full_segments = json.load(f)

    speaker_to_name = {
        item["speaker_id"]: item["predicted_name"]
        for item in llm_output.get("name_recognition", {}).get("mappings", [])
        if item.get("predicted_name")
    }

    resolved_segments = [
        {
            **seg,
            "speaker": speaker_to_name.get(seg["speaker"], seg["speaker"])
        }
        for seg in full_segments
    ]

    final = {
        "summary": llm_output["summary"],
        "tasks": llm_output["tasks"],
        "name_recognition": llm_output["name_recognition"],
        "segments": resolved_segments
    }

    return {"final_output": final}

# =========================
# GRAPH
# =========================

def build_graph():
    os.makedirs("./raw_output", exist_ok=True)
    os.makedirs("./validated_output", exist_ok=True)
    os.makedirs("./llm_input", exist_ok=True)
    os.makedirs("./llm_output", exist_ok=True)

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

    graph.add_edge(START, "asr")
    graph.add_edge(START, "sign")
    graph.add_edge("asr", "validate_audio")
    graph.add_edge("sign", "validate_video")
    graph.add_edge(["validate_audio", "validate_video"], "merge")
    graph.add_edge("merge", "validate_merged")
    graph.add_edge("validate_merged", "simplify")
    graph.add_edge("simplify", "llm")
    graph.add_edge("llm", "final")
    graph.add_edge("final", END)

    return graph.compile()
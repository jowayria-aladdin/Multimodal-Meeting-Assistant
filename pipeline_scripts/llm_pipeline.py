import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

# ----------------------------
# Pydantic models
# ----------------------------

class SummaryOutput(BaseModel):
    title: str = Field(..., min_length=1)
    short_summary: str = Field(..., min_length=1)
    detailed_summary: str = Field(..., min_length=1)
    key_points: List[str]

class TaskItem(BaseModel):
    task_name: str = Field(..., min_length=1)
    assignee: Optional[str] = None
    assigned_by: Optional[str] = None
    deadline: Optional[str] = None
    evidence: Optional[str] = None

class TasksOutput(BaseModel):
    tasks: List[TaskItem]

class NameRecognitionItem(BaseModel):
    speaker_id: str = Field(..., min_length=1)
    predicted_name: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Optional[str] = None

class NameRecognitionOutput(BaseModel):
    mappings: List[NameRecognitionItem]

class FinalLLMOutput(BaseModel):
    summary: SummaryOutput
    tasks: TasksOutput
    name_recognition: NameRecognitionOutput

# ----------------------------
# Helpers
# ----------------------------

def load_merged_json(input_path: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def merged_json_to_text(input_path: str) -> str:
    segments = load_merged_json(input_path)
    lines = []

    for seg in segments:
        lines.append(
            f"[{seg['start']:.3f} - {seg['end']:.3f}] {seg['speaker']}: {seg['text']}"
        )

    return "\n".join(lines)

def extract_json_from_response(response_text: str) -> dict:
    return json.loads(response_text)

# ----------------------------
# Prompt builders
# ----------------------------

def build_summary_prompt(transcript_text: str) -> str:
    return f"""
You are given a meeting transcript.

Your task is to produce a structured JSON summary.

Rules:
- Return JSON only.
- Do not include markdown.
- Do not invent facts not supported by the transcript.

Required JSON format:
{{
    "title": "string",
    "short_summary": "string",
    "detailed_summary": "string",
    "key_points": ["string", "string"]
}}

Transcript:
{transcript_text}
""".strip()

def build_tasks_prompt(transcript_text: str) -> str:
    return f"""
You are given a meeting transcript.

Extract only explicit or strongly implied action items.

Rules:
- Return JSON only.
- Do not include markdown.
- Do not invent tasks that are not supported by the transcript.
- If assignee, assigned_by, or deadline are unknown, return null.
- Include a short evidence phrase from the transcript.

Required JSON format:
{{
    "tasks": [
    {{
        "task_name": "string",
        "assignee": "string or null",
        "assigned_by": "string or null",
        "deadline": "string or null",
        "evidence": "string or null"
    }}
    ]
}}

Transcript:
{transcript_text}
""".strip()

def build_names_prompt(transcript_text: str) -> str:
    return f"""
You are given a meeting transcript with speaker IDs such as SPEAKER_00 and SPEAKER_01.

Infer possible real names only when there is evidence in the transcript.
If there is not enough evidence, keep predicted_name as null.

Rules:
- Return JSON only.
- Do not include markdown.
- Do not guess blindly.
- Confidence must be between 0 and 1.
- Include the evidence used for the guess.

Required JSON format:
{{
    "mappings": [
    {{
        "speaker_id": "string",
        "predicted_name": "string or null",
        "confidence": 0.0,
        "evidence": "string or null"
    }}
    ]
}}

Transcript:
{transcript_text}
""".strip()

# ----------------------------
# LLM call placeholder
# ----------------------------

def call_llm(prompt: str) -> str:
    """
    Replace this with your actual LLM call.
    Example: OpenAI, Azure, local Ollama, etc.
    Must return raw text that is valid JSON.
    """
    raise NotImplementedError("Connect your LLM here.")

# ----------------------------
# Pipeline functions
# ----------------------------

def run_summary(transcript_text: str) -> SummaryOutput:
    prompt = build_summary_prompt(transcript_text)
    raw_response = call_llm(prompt)
    parsed = extract_json_from_response(raw_response)
    return SummaryOutput.model_validate(parsed)

def run_tasks(transcript_text: str) -> TasksOutput:
    prompt = build_tasks_prompt(transcript_text)
    raw_response = call_llm(prompt)
    parsed = extract_json_from_response(raw_response)
    return TasksOutput.model_validate(parsed)

def run_names(transcript_text: str) -> NameRecognitionOutput:
    prompt = build_names_prompt(transcript_text)
    raw_response = call_llm(prompt)
    parsed = extract_json_from_response(raw_response)
    return NameRecognitionOutput.model_validate(parsed)

def run_llm_pipeline(input_path: str, output_path: str) -> dict:
    transcript_text = merged_json_to_text(input_path)

    summary = run_summary(transcript_text)
    tasks = run_tasks(transcript_text)
    names = run_names(transcript_text)

    final_output = FinalLLMOutput(
        summary=summary,
        tasks=tasks,
        name_recognition=names
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(final_output.model_dump(), f, indent=4, ensure_ascii=False)

    return final_output.model_dump()

if __name__ == "__main__":
    input_file = "./validated_output/validated_merged.json"
    output_file = "./llm_output/llm_results.json"

    try:
        result = run_llm_pipeline(input_file, output_file)
        print("LLM pipeline completed successfully.")
        print(f"Saved: {output_file}")
    except (ValidationError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        print("LLM pipeline failed:")
        print(e)
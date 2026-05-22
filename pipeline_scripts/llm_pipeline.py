import asyncio
from datetime import date
import json
import os
from pathlib import Path
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(line_buffering=True)

# ----------------------------
# Config
# ----------------------------

load_dotenv(override=True)

# LLM Modes:
# - gemini : use Gemini only
# - qwen   : use Qwen only
# - dual   : try Gemini first, fallback to Qwen
LLM_MODE = os.getenv("LLM_MODE", "dual").lower()

# Gemini config
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Qwen config (through Ollama)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# If using a local model on one GPU, parallel calls may hurt instead of help.
LOCAL_PARALLEL = os.getenv("LOCAL_PARALLEL", "false").lower() == "true"

# Request timeouts
GEMINI_TIMEOUT = float(os.getenv("GEMINI_TIMEOUT", "60"))
QWEN_TIMEOUT = float(os.getenv("QWEN_TIMEOUT", "120"))


# ----------------------------
# Pydantic models
# ----------------------------

class SummaryOutput(BaseModel):
    text: str = Field(..., min_length=1)


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
    segments: List[dict]


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
            f"{seg['speaker']}: {seg['text']}"
        )

    return "\n".join(lines)


def extract_json_from_response(response_text: str) -> dict:
    return json.loads(response_text)


def build_output_language_instruction(language: str) -> str:
    language = (language or "mix").lower()

    if language == "en":
        return ("Write all output in English only. Keep names unchanged."
                "Do NOT introduce any new languages or weird unicode characters (no Chinese or unrelated languages)."
                )

    if language == "ar":
        return ("اكتب كل النتايج بالعربي باللهجة المصرية بس و حافظ على الأسماء زي ما هي."
                "لا تكتب أي لغة تانية أو أي حروف غريبة (ممنوع تكتب إنجليزي أو صيني أو أي لغة غير مرتبطة).")

    return (
        "Write the output in mixed Egyptian Arabic and English, matching the transcript style. "
        "Keep names and technical terms unchanged. "
        "STRICTLY use only Arabic letters (أ-ي), English letters (A-Z, a-z), numbers, and basic punctuation. "
        "Do NOT output any other Unicode characters or symbols (no Chinese, Japanese, or weird symbols). "
    )


def apply_name_mappings(
    tasks: TasksOutput,
    names: NameRecognitionOutput,
    segments: List[dict]
) -> tuple[TasksOutput, List[dict]]:
    speaker_to_name = {
        item.speaker_id: item.predicted_name
        for item in names.mappings
        if item.predicted_name
    }

    # Apply to tasks
    updated_tasks = []
    for task in tasks.tasks:
        updated_tasks.append(
            TaskItem(
                task_name=task.task_name,
                assignee=speaker_to_name.get(task.assignee, task.assignee),
                assigned_by=speaker_to_name.get(task.assigned_by, task.assigned_by),
                deadline=task.deadline,
                evidence=task.evidence
            )
        )

    # Apply to transcript segments
    updated_segments = []
    for seg in segments:
        updated_segments.append({
            **seg,
            "speaker": speaker_to_name.get(seg["speaker"], seg["speaker"])
        })

    return TasksOutput(tasks=updated_tasks), updated_segments

# ----------------------------
# Prompt builders
# ----------------------------

def build_summary_prompt(transcript_text: str, language: str) -> str:
    output_language_instruction = build_output_language_instruction(language)

    return f"""
You are analyzing a technical meeting transcript in Arabic, English, or mixed code-switching.

Task:
Write a concise meeting summary based only on the transcript.

Language:
{output_language_instruction}

Rules:
- Return JSON only.
- Do not use markdown.
- Do not invent facts.
- Summarize at the idea level, not sentence by sentence.
- Make it as a paragraph and keep it short and focused on key points.
- Focus on the main topic, key findings, problems, decisions, and next steps.
- Do not copy long phrases from the transcript.
- If any invalid or foreign character appears, regenerate the output correctly.

Required JSON format:
{{
    "text": "string"
}}

Transcript:
{transcript_text}
""".strip()


def build_tasks_prompt(transcript_text: str, language: str) -> str:
    output_language_instruction = build_output_language_instruction(language)
    today = date.today().isoformat()

    return f"""
You are analyzing a technical meeting transcript in Arabic, English, or mixed code-switching.

Task:
Extract action items only.

Language:
{output_language_instruction}

Rules:
- Return JSON only.
- Do not use markdown.
- Do not invent tasks.
- Extract a task only if it is a clear commitment, request, agreement, or next step.
- The task_name must follow the same language style used in the transcript.
- Evidence must be a short quote from the transcript.
- Extract ALL tasks. Do not miss any task even if multiple tasks exist.

Role assignment:
- If a speaker says "I will..." or "هعمل..." → 
    assignee = that speaker, assigned_by = that speaker
- If one speaker asks another to do something → 
    assigned_by = the speaker asking, assignee = the target if clear
- If phrased as "let's..." or "خلّينا..." → 
    assigned_by = the speaker, assignee = null
- If unclear → 
    assignee = null, assigned_by = null

Example ():
"هعمل test عليه النهاردة" →
{{
    "task_name": "هعمل test عليه النهاردة",
    "assignee": "SPEAKER_01",
    "assigned_by": "SPEAKER_01",
    "deadline": "2026-05-20"
}}

Deadline:
    - If not explicit → deadline = null
    - If explicit, convert it to an ISO 8601 date string (YYYY-MM-DD) relative to today's date: {today}.
    - Examples:
        - "النهاردة" → today's date
        - "بكرا" → tomorrow's date
        - "كمان أسبوع" → today + 7 days
        - "نهاية الأسبوع" → the upcoming Sunday
        - "الأسبوع الجاي" → the upcoming Monday
    - Always return a concrete date string, never a relative phrase.


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


def build_names_prompt(transcript_text: str, language: str) -> str:
    output_language_instruction = build_output_language_instruction(language)

    return f"""
You are analyzing a meeting transcript with speaker IDs such as SPEAKER_00 and SPEAKER_01.

Task:
Infer real names only when there is clear evidence.

Language:
{output_language_instruction}

Rules:
- Return JSON only.
- Do not use markdown.
- Do not guess.
- Do not translate names.
- Valid evidence includes self-introduction, one speaker explicitly naming another, or direct address.
- If a speaker directly addresses a person by name at the start of an utterance, that name usually refers to another speaker, not the current speaker.
- In a two-speaker conversation, if one speaker says something like "Hi, Adam" and there is no conflicting evidence, Adam is more likely the other speaker.
- Only assign a name if the transcript clearly supports that speaker.
- If unclear:
    - predicted_name = null
    - confidence = 0.0
- Use real JSON null, not the string "null".

Example:
Transcript:
SPEAKER_00: Hi, Adam
SPEAKER_01: Hey

Output:
{{
    "mappings": [
    {{
        "speaker_id": "SPEAKER_00",
        "predicted_name": null,
        "confidence": 0.0
    }},
    {{
        "speaker_id": "SPEAKER_01",
        "predicted_name": "Adam",
        "confidence": 0.7
    }}
    ]
}}

Required JSON format:
{{
    "mappings": [
        {{
            "speaker_id": "string",
            "predicted_name": "string or null",
            "confidence": 0.0
        }}
    ]
}}

Transcript:
{transcript_text}
""".strip()


# ----------------------------
# Provider calls
# ----------------------------

async def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        if not response.is_success:
            print(f"[LLM] Gemini error response: {response.text}", flush=True)
        response.raise_for_status()

    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response format: {data}") from e


async def call_qwen(prompt: str) -> str:
    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1
        }
    }

    async with httpx.AsyncClient(timeout=QWEN_TIMEOUT) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()

    if "response" not in data:
        raise ValueError(f"Unexpected Ollama/Qwen response format: {data}")

    return data["response"]


async def call_llm_async(prompt: str) -> str:
    if LLM_MODE == "gemini":
        return await call_gemini(prompt)

    if LLM_MODE == "qwen":
        return await call_qwen(prompt)

    try:
        return await call_gemini(prompt)
    except Exception as gemini_error:
        print(f"[LLM] Gemini failed, falling back to Qwen: {gemini_error}")
        return await call_qwen(prompt)


# ----------------------------
# Async task runners
# ----------------------------

async def run_summary_async(transcript_text: str, language: str) -> SummaryOutput:
    prompt = build_summary_prompt(transcript_text, language)
    raw_response = await call_llm_async(prompt)
    parsed = extract_json_from_response(raw_response)
    print("[LLM] Summary done", flush=True)
    return SummaryOutput.model_validate(parsed)


async def run_tasks_async(transcript_text: str, language: str) -> TasksOutput:
    prompt = build_tasks_prompt(transcript_text, language)
    raw_response = await call_llm_async(prompt)
    parsed = extract_json_from_response(raw_response)
    print("[LLM] Tasks done", flush=True)
    return TasksOutput.model_validate(parsed)


async def run_names_async(transcript_text: str, language: str) -> NameRecognitionOutput:
    prompt = build_names_prompt(transcript_text, language)
    raw_response = await call_llm_async(prompt)
    parsed = extract_json_from_response(raw_response)
    print("[LLM] Names done", flush=True)
    return NameRecognitionOutput.model_validate(parsed)


# ----------------------------
# Pipeline
# ----------------------------

async def run_llm_pipeline_async(input_path: str, output_path: str, language: str = "mix") -> dict:
    transcript_text = merged_json_to_text(input_path)

    use_parallel = True
    if LLM_MODE == "qwen" and not LOCAL_PARALLEL:
        use_parallel = False

    if use_parallel:
        summary, tasks, names = await asyncio.gather(
            run_summary_async(transcript_text, language),
            run_tasks_async(transcript_text, language),
            run_names_async(transcript_text, language)
        )
    else:
        summary = await run_summary_async(transcript_text, language)
        tasks = await run_tasks_async(transcript_text, language)
        names = await run_names_async(transcript_text, language)
    
    print("[LLM] Applying name mappings...", flush=True)
    segments = load_merged_json(input_path)
    tasks, segments = apply_name_mappings(tasks, names, segments)

    final_output = FinalLLMOutput(
        summary=summary,
        tasks=tasks,
        name_recognition=names,
        segments=segments
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(final_output.model_dump(), f, indent=4, ensure_ascii=False)

    return final_output.model_dump()


def run_llm_pipeline(input_path: str, output_path: str, language: str = "mix") -> dict:
    return asyncio.run(run_llm_pipeline_async(input_path, output_path, language))


if __name__ == "__main__":
    input_file = "./validated_output/validated_merged.json"
    output_file = "./llm_output/llm_results.json"

    try:
        result = run_llm_pipeline(input_file, output_file)
        print("[LLM] Pipeline completed successfully.")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (
        ValidationError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError,
        httpx.HTTPError,
        RuntimeError
    ) as e:
        print("[LLM] Pipeline failed:")
        print(e)
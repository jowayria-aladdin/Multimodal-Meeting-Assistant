import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field, ValidationError, field_validator

class SegmentItem(BaseModel):
    speaker: str = Field(..., min_length=1)
    start: float = Field(..., ge=0)
    end: float = Field(..., ge=0)
    text: str = Field(..., min_length=1)

    @field_validator("speaker", "text")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("end")
    @classmethod
    def end_must_be_after_start(cls, end_value: float, info):
        start_value = info.data.get("start")
        if start_value is not None and end_value < start_value:
            raise ValueError("end must be greater than or equal to start")
        return end_value

def validate_json_file(input_path: str, output_path: str | None = None) -> List[dict]:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Top-level JSON must be a list")

    validated_items = [SegmentItem.model_validate(item) for item in raw_data]
    validated_data = [item.model_dump() for item in validated_items]

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(validated_data, f, indent=4, ensure_ascii=False)

    return validated_data

if __name__ == "__main__":
    # Audio
    audio_input = "./raw_output/audio_raw.json"
    audio_output = "./validated_output/validated_audio.json"

    try:
        validated_audio = validate_json_file(audio_input, audio_output)
        print(f"Audio JSON validated successfully. Items: {len(validated_audio)}")
        print(f"Saved: {audio_output}")
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print("Audio JSON validation failed:")
        print(e)

    # Video
    video_input = "./raw_output/video_raw.json"
    video_output = "./validated_output/validated_video.json"

    try:
        validated_video = validate_json_file(video_input, video_output)
        print(f"Video JSON validated successfully. Items: {len(validated_video)}")
        print(f"Saved: {video_output}")
    except FileNotFoundError:
        print("Video file not found yet, skipped.")
    except (ValidationError, ValueError) as e:
        print("Video JSON validation failed:")
        print(e)
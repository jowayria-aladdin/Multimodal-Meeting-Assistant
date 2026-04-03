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


def load_and_validate(path: str) -> List[SegmentItem]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"{file_path} must contain a JSON list")

    return [SegmentItem.model_validate(item) for item in raw_data]


def merge_jsons(audio_json_path: str, video_json_path: str, output_path: str) -> List[dict]:
    audio_items = load_and_validate(audio_json_path)
    video_items = load_and_validate(video_json_path)

    merged_items = audio_items + video_items
    merged_items.sort(key=lambda item: (item.start, item.end, item.speaker))

    merged_data = [item.model_dump() for item in merged_items]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    return merged_data


if __name__ == "__main__":
    audio_file = "output/validated_audio.json"
    video_file = "output/validated_video.json"
    merged_file = "output/merged.json"

    try:
        merged = merge_jsons(audio_file, video_file, merged_file)
        print(f"Merged successfully. Total items: {len(merged)}")
        print(f"Saved: {merged_file}")
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print("Merge failed:")
        print(e)
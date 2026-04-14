import json
from pathlib import Path
from typing import List, Tuple
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

def load_merged_json(path: str) -> List[dict]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Top-level merged JSON must be a list")

    if len(raw_data) == 0:
        raise ValueError("Merged JSON must not be empty")

    return raw_data

def validate_sorted(items: List[SegmentItem]) -> None:
    sorted_items = sorted(items, key=lambda item: (item.start, item.end, item.speaker))
    if items != sorted_items:
        raise ValueError("Merged JSON is not sorted by (start, end, speaker)")

def validate_no_exact_duplicates(items: List[SegmentItem]) -> None:
    seen: set[Tuple[str, float, float, str]] = set()

    for item in items:
        key = (item.speaker, item.start, item.end, item.text)
        if key in seen:
            raise ValueError(
                f"Duplicate segment found: speaker={item.speaker}, "
                f"start={item.start}, end={item.end}, text={item.text}"
            )
        seen.add(key)

def validate_merged_json_file(input_path: str, output_path: str | None = None) -> List[dict]:
    raw_data = load_merged_json(input_path)

    validated_items = [SegmentItem.model_validate(item) for item in raw_data]

    validate_sorted(validated_items)
    validate_no_exact_duplicates(validated_items)

    validated_data = [item.model_dump() for item in validated_items]

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(validated_data, f, indent=4, ensure_ascii=False)

    return validated_data

if __name__ == "__main__":
    merged_input = "./validated_output/merged.json"
    merged_output = "./validated_output/validated_merged.json"

    try:
        validated_merged = validate_merged_json_file(merged_input, merged_output)
        print(f"Merged JSON validated successfully. Items: {len(validated_merged)}")
        print(f"Saved: {merged_output}")
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print("Merged JSON validation failed:")
        print(e)
import json
from pathlib import Path
from typing import List

def load_json_file(path: str) -> List[dict]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a JSON list")

    return data

def merge_jsons(audio_json_path: str, video_json_path: str, output_path: str) -> List[dict]:
    audio_items = load_json_file(audio_json_path)
    video_items = load_json_file(video_json_path)

    merged_items = audio_items + video_items
    merged_items.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(merged_items, f, indent=4, ensure_ascii=False)

    return merged_items

if __name__ == "__main__":
    audio_file = "./validated_output/validated_audio.json"
    video_file = "./validated_output/validated_video.json"
    merged_file = "./validated_output/merged.json"

    try:
        merged = merge_jsons(audio_file, video_file, merged_file)
        print(f"Merged successfully. Total items: {len(merged)}")
        print(f"Saved: {merged_file}")
    except (ValueError, FileNotFoundError, KeyError) as e:
        print("Merge failed:")
        print(e)
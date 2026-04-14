import json
from pathlib import Path
from typing import List

def simplify_json(input_path: str, output_path: str) -> List[dict]:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Merged JSON must be a list")

    simplified = []

    for item in data:
        simplified.append({
            "speaker": item["speaker"],
            "text": item["text"]
        })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(simplified, f, indent=4, ensure_ascii=False)

    return simplified

if __name__ == "__main__":
    input_file = "./validated_output/validated_merged.json"
    output_file = "./llm_input/llm_input.json"

    try:
        result = simplify_json(input_file, output_file)
        print(f"Simplified JSON created. Items: {len(result)}")
        print(f"Saved: {output_file}")
    except Exception as e:
        print("Simplification failed:")
        print(e)
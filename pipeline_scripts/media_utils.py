import ffmpeg
import os

def convert_webm_to_wav(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    (
        ffmpeg
        .input(input_path)
        .output(output_path, ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path
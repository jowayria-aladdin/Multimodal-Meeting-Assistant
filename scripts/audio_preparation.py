import librosa
import soundfile as sf
import os

# -----------------------------------------
# EDIT THESE TWO PATHS ONLY
# -----------------------------------------
INPUT_AUDIO_PATH = r"copy relative path"        # <-- replace with your file path
OUTPUT_AUDIO_PATH = r"copy relative path" # <-- replace output location
# -----------------------------------------

def prepare_audio():
    # Create folders if not exist
    os.makedirs(os.path.dirname(OUTPUT_AUDIO_PATH), exist_ok=True)

    print(f"[INFO] Loading audio from: {INPUT_AUDIO_PATH}")

    # Load and convert to mono 16kHz
    audio, sr = librosa.load(INPUT_AUDIO_PATH, sr=16000, mono=True)

    # Save cleaned audio
    sf.write(OUTPUT_AUDIO_PATH, audio, 16000)

    print(f"[SUCCESS] Cleaned audio saved at: {OUTPUT_AUDIO_PATH}")

if __name__ == "__main__":
    prepare_audio()


import os
import glob
import subprocess
import time
import requests
from pathlib import Path
from datetime import datetime

# configurations 
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "converted_output")
DOWNLOADS_DIR = str(Path.home() / "Downloads")
CHECK_INTERVAL = 120  # check every 2 minutes 
SIGN_API_URL   = "http://localhost:5001/predict"

#setup output directory 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f" MEETING WATCHER SERVICE ")
print(f" Monitoring: {DOWNLOADS_DIR}")
print(f" Output to:  {OUTPUT_DIR}")
print(f" Interval:   Every 2 minutes")

def call_sign_language_api(webm_path):
    try:
        with open(webm_path, "rb") as f:
            response = requests.post(
                SIGN_API_URL,
                files={"video": (os.path.basename(webm_path), f, "video/webm")},
                timeout=120
            )

        if response.status_code == 200:
            result      = response.json()
            base_name   = os.path.basename(webm_path).replace(".webm", "")
            result_path = os.path.join(OUTPUT_DIR, f"{base_name}_signs.txt")

            with open(result_path, "w", encoding="utf-8") as out:
                out.write(f"Top prediction : {result['sign']} "
                          f"({result['confidence']*100:.1f}%)\n\n")
                out.write("Top 10:\n")
                for i, p in enumerate(result.get("top10", []), 1):
                    out.write(f"  #{i:2d}  {p['sign']:<30} {p['confidence']*100:.1f}%\n")

            print(f"   Sign: {result['sign']} ({result['confidence']*100:.1f}%)")
        else:
            print(f"   API error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("   API not running — start sign_lang_api.py first")
    except Exception as e:
        print(f"   Error: {e}")

def process_files():
    #find all meeting files in Downloads
    search_pattern = os.path.join(DOWNLOADS_DIR, "meeting_*_*.webm")
    found_files = glob.glob(search_pattern)

    if not found_files:
        return False # No files found

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  Found {len(found_files)} new recordings! Processing.")

    for source_path in found_files:
        filename = os.path.basename(source_path)
        cmd = []
        output_filename = ""
        
        #  screen recordingto MP4
        if "_video" in filename:
            output_filename = filename.replace(".webm", ".mp4")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = ['ffmpeg', '-i', source_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-y', output_path]
            
        # webcam recording to MP4
        elif "_webcam" in filename:
            output_filename = filename.replace(".webm", ".mp4")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = ['ffmpeg', '-i', source_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-y', output_path]

        # audio recording to WAV
        elif "_audio" in filename:
            output_filename = filename.replace(".webm", ".wav")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = ['ffmpeg', '-i', source_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', '-y', output_path]
        
        else:
            continue

        print(f" Converting: {filename}.")
        
        try:
            # Run FFmpeg
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f" Saved: {output_filename}")

            if "_webcam" in filename:
                print(f"   Running sign detection...")
                call_sign_language_api(source_path)
            
            # Delete Original
            try:
                os.remove(source_path)
                print(f"   Deleted original.")
            except:
                pass
        except Exception as e:
            print(f"  Error: {e}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}]  Batch finished. Resuming watch.")
    return True

if __name__ == "__main__":
    try:
        while True:
            # Run the check
            found = process_files()
            
            if not found:
                # Print a dot every check to show it's alive, or stay silent
                # print(".", end="", flush=True) 
                pass

            # Wait for 2 minutes
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n Watcher stopped by user.")


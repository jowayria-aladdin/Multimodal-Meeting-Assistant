
import os
import glob
import subprocess
import time
import requests
import json
from datetime import timedelta
from pathlib import Path
from datetime import datetime

# configurations 
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS_DIR  = os.environ.get("DOWNLOADS_DIR", str(Path.home() / "Downloads"))
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_DIR, "converted_output"))
CHECK_INTERVAL = 120  # check every 2 minutes
SIGN_API_URL   = os.environ.get("SIGN_API_URL", "http://localhost:5001/predict")

#setup output directory 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f" MEETING WATCHER SERVICE ")
print(f" Monitoring: {DOWNLOADS_DIR}")
print(f" Output to:  {OUTPUT_DIR}")
print(f" Interval:   Every 2 minutes")

def call_sign_language_api(webm_path):
    try:
        # count frames directly, use fixed 30fps for webm
        probe = subprocess.run(
            ['ffprobe', '-v', 'error',
             '-select_streams', 'v:0',
             '-count_frames',
             '-show_entries', 'stream=nb_read_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', webm_path],
            capture_output=True, text=True
        )

        val = probe.stdout.strip()
        frames = int(val) if val.isdigit() else 0
        fps = 30
        duration = frames / fps
        print(f"   Frames: {frames}, FPS: {fps}, Duration: {duration:.1f}s")

        if duration == 0:
            print("   Could not determine duration — skipping")
            return

        def fmt(secs):
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            return f"{h:02}:{m:02}:{s:02}"

        output = []
        seg_start = 0

        while seg_start < duration:
            seg_end  = min(seg_start + 5, duration)
            seg_path = f"/tmp/seg_{int(seg_start)}.webm"

            result_ffmpeg = subprocess.run([
                'ffmpeg', '-y', '-i', webm_path,
                '-ss', str(seg_start),
                '-t', '5',
                '-c:v', 'libvpx',
                '-c:a', 'libvorbis',
                seg_path
            ], capture_output=True, text=True)

            if not os.path.exists(seg_path):
                print(f"   Segment {int(seg_start)}s not created — skipping")
                seg_start += 5
                continue

            with open(seg_path, "rb") as f:
                response = requests.post(
                    SIGN_API_URL,
                    files={"video": (f"seg_{int(seg_start)}.webm", f, "video/webm")},
                    timeout=120
                )

            os.remove(seg_path)

            if response.status_code == 200:
                result = response.json()
                output.append({
                    "speaker":    "disabled",
                    "start":      fmt(seg_start),
                    "end":        fmt(seg_end),
                    "text":       result["sign"]
                })
                print(f"   {fmt(seg_start)} → {result['sign']} ({result['confidence']*100:.1f}%)")
            else:
                print(f"   Segment {int(seg_start)}s failed: {response.status_code} — {response.json()}")

            seg_start += 5

        base_name   = os.path.basename(webm_path).replace(".webm", "")
        result_path = os.path.join(OUTPUT_DIR, f"{base_name}_signs.json")

        with open(result_path, "w", encoding="utf-8") as out:
            json.dump(output, out, ensure_ascii=False, indent=2)

        print(f"   Saved: {os.path.basename(result_path)}")

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

import os
import glob
import subprocess
import time
import shutil  # Added this to quickly move files without converting
from pathlib import Path
from datetime import datetime

# configurations 
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "converted_output")
DOWNLOADS_DIR = str(Path.home() / "Downloads")
CHECK_INTERVAL = 120  # check every 2 minutes 

#setup output directory 
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f" MEETING WATCHER SERVICE ")
print(f" Monitoring: {DOWNLOADS_DIR}")
print(f" Output to:  {OUTPUT_DIR}")
print(f" Interval:   Every 2 minutes")

def process_files():
    #find all meeting files in Downloads
    search_pattern = os.path.join(DOWNLOADS_DIR, "meeting_*_*.webm")
    found_files = glob.glob(search_pattern)

    if not found_files:
        return False # No files found

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  Found {len(found_files)} new recordings! Processing.")

    for source_path in found_files:
        filename = os.path.basename(source_path)
        
        #  VIDEO OR WEBCAM Just move the .webm file directly (Instantly)
        if "_video" in filename or "_webcam" in filename:
            output_path = os.path.join(OUTPUT_DIR, filename) # Keep the .webm extension
            print(f" Moving: {filename}.")
            
            try:
                shutil.move(source_path, output_path)
                print(f" Saved: {filename}")
            except Exception as e:
                print(f"  Error moving file: {e}")
                
            continue # Move to the next file in the loop

        # AUDIO Convert to .wav using FFmpeg
        elif "_audio" in filename:
            output_filename = filename.replace(".webm", ".wav")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = ['ffmpeg', '-i', source_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '2', '-y', output_path]
            
            print(f" Converting: {filename}.")
            
            try:
                # Run FFmpeg
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f" Saved: {output_filename}")
                
                # Delete Original Audio .webm
                try:
                    os.remove(source_path)
                    print(f"   Deleted original.")
                except:
                    pass
            except Exception as e:
                print(f"  Error: {e}")
        
        else:
            continue

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
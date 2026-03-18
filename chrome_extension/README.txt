Step 1: Download and Extract
    1.Navigate to the LughaCap web landing page and click "Download Extension Now!".
    2.Once downloaded, extract/unzip the LughaCap.zip file to a folder on your computer.
        Note: Remember where you saved this folder, as Chrome will need to reference it in the next steps.

Step 2: Enable Developer Mode in Chrome
    1.Open Google Chrome.
    2.Type chrome://extensions/ into your URL bar and press Enter.
    3.In the top-right corner of the Extensions page, toggle the "Developer mode" switch so it is turned ON.

Step 3: Load the Extension
    1.Click the "Load unpacked" button that appears after enabling Developer mode.
    2.A file browser window will open. Navigate to the extracted LughaCap folder from Step 1.
    3.Select the folder and click "Select Folder", make sure you select the folder containing the manifest.json file, not a folder inside it.
    4.LughaCap should now appear in your list of extensions, and you should see its icon in the Chrome toolbar.

Step 4: Grant the Extension Permissions
    1.Once you select the folder, permissions setup page will appear asking for microphone and camera access. These permissions are necessary for LughaCap to capture audio and video during meetings.
    2. Click **Grant Microphone** and click "Allow" on the browser prompt. (Required to transcribe the meeting).
    3. Click **Grant Camera** and click "Allow" on the browser prompt. (Optional: Only required if you want to use the Sign Language translation features. You can skip this if you only want audio).
        *Note: If you accidentally close the setup tab or need to change the permissions, you can open it again by clicking the LughaCap icon in your toolbar and select "Options".*

Step 5: Use the Extension
    1. Click the Puzzle Piece icon (Extensions menu) in the top right of your Chrome browser.
    2. Find LughaCap in the list.
    3. To record a meeting, navigate to your meeting tab and click the LughaCap icon. The icon will turn red and display "REC".
    4. To stop recording, click the icon again. Your audio and video files will automatically process and download.

To run the full LughaCap pipeline locally
You need to start the Python Processing Watcher.
Prerequisites: FFmpeg: This is mandatory for the Python script to convert the browser's raw .webm files into .wav and .mp4 formats. Ensure FFmpeg is installed and added to your system's Environment Variables (PATH) for the script to access it from any directory.
1. Open a terminal or command prompt.
2. Navigate to the directory where you extracted the LughaCap files.
3. Run the following command to start the processing watcher in the background. This script continuously monitors the recordings in the downloads folder for new .webm files, processes them, and saves the output in the "processed" folder.:
    python process_recordings.py
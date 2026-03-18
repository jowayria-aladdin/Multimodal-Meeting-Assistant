/**
 background server worker for the Chrome Extension
 overall functionality: 
 this script manages the extension's lifecycle and recording states (IDLE, RECORDING, SAVING),
 and communication between the browser action (the icon) and the offscreen document 
 where the actual media capture happens. It ensures files are downloaded correctly
 */

// automatically open the setup page when the extension is first installed
chrome.runtime.onInstalled.addListener((details) => {
    if (details.reason === 'install') {
        chrome.runtime.openOptionsPage();
    }
});
//initialization
// Resets the extension state whenever Chrome starts to prevent the UI from getting stuck in a recording state if the browser crashed.
chrome.runtime.onStartup.addListener(() => {
    chrome.storage.local.set({ recordingState: 'IDLE' });// Reset state to IDLE
    chrome.action.setBadgeText({ text: "" });// Clear any text on the extension icon
});
//main click handler for the extension icon
// This listener handles user clicks on the extension icon. It checks the current recording state and either starts a new recording, stops an ongoing recording, or ignores clicks if a save operation is in progress. It also manages the badge text and color to provide visual feedback on the recording status.
chrome.action.onClicked.addListener(async (tab) => {
// state recovery pull the current state from local storage.
  // because Service Workers can "go to sleep" and lose variables.
  const result = await chrome.storage.local.get('recordingState');
  let currentState = result.recordingState || 'IDLE';

  console.log(`Clicked! Current State: ${currentState}`);
//logic for handling different states when the extension icon is clicked.

//if currently recording, we want to stop the recording process.
  if (currentState === 'RECORDING') {
    console.log("Command: Stop");
    
    // Set to SAVING immediately to block double-clicks to prevent the user from clicking again while the files are being processed and downloaded.
    await chrome.storage.local.set({ recordingState: 'SAVING' });
    chrome.action.setBadgeText({ text: "SAVE" });
    chrome.action.setBadgeBackgroundColor({ color: "#0000FF" });

    try {
      // Send a message to offscreen.js to trigger the stopCapture() function.
        chrome.runtime.sendMessage({ type: 'STOP', target: 'offscreen' });
    } catch (e) {
      // If the offscreen document was closed unexpectedly, reset the extension.
        console.warn("Offscreen unreachable. Force resetting.");
        await chrome.storage.local.set({ recordingState: 'IDLE' });
        chrome.action.setBadgeText({ text: "" });
    }
    return;
  }
//if currently saving, we want to ignore clicks to prevent the user from interrupting the save process. This is important because the saving process can take a few seconds, and we don't want the user to accidentally start a new recording or stop the save process while it's still running.
  if (currentState === 'SAVING') {
    console.warn(" Ignored click: Still saving previous files.");
    return; 
  }
//if currently idle, we want to start a new recording process.
  if (currentState === 'IDLE') {
      console.log("Command: Start");
      chrome.action.setBadgeText({ text: "..." });
// check if an offscreen document (needed for MediaRecorder) already exists. If not, create one. This allows us to reuse the offscreen document across multiple recordings without having to recreate it every time, which can save resources and improve performance.
      const existingContexts = await chrome.runtime.getContexts({});
      const offscreen = existingContexts.find(c => c.contextType === 'OFFSCREEN_DOCUMENT');
      // this document is required to capture tab audio/video in Manifest V3.
      if (!offscreen) {
        await chrome.offscreen.createDocument({
          url: 'offscreen.html',
          reasons: ['USER_MEDIA'],
          justification: 'Recording tab audio'
        });
      }
// get the stream ID for the current active tab to start the capture.
      try {
          const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
          // tell offscreen.js to start recording using the tab's Stream ID.
          chrome.runtime.sendMessage({ type: 'START', target: 'offscreen', data: streamId });

          // SAVE STATE TO DISK, not just in memory, to survive Service Worker restarts.
          // Update state and UI to show the user that recording is active.
          await chrome.storage.local.set({ recordingState: 'RECORDING' });
          
          chrome.action.setBadgeText({ text: "REC" });
          chrome.action.setBadgeBackgroundColor({ color: "#FF0000" });

      } catch (err) {
          console.error(" Capture Error:", err);
          
          // only kill zombies if we are truly stuck in a recording state. If we failed to get a stream ID, but we aren't in a recording state, then something else is wrong and killing the offscreen document won't help.
          //error handling if a zombie stream is stuck, clean it up and reset the extension state. This can happen if the offscreen document is still open and holding onto a media stream that is no longer active, which can prevent new recordings from starting.
          if (err.message.includes("active stream")) {
              console.warn(" Zombie stream detected. Cleaning up.");
              if (offscreen) await chrome.offscreen.closeDocument();
              
              chrome.action.setBadgeText({ text: "RST" });
              chrome.action.setBadgeBackgroundColor({ color: "#FFA500" });
              setTimeout(() => chrome.action.setBadgeText({ text: "" }), 1000);
          } else {
              chrome.action.setBadgeText({ text: "ERR" });
          }
          
// reset to IDLE so the user can try again
          await chrome.storage.local.set({ recordingState: 'IDLE' });
      }
  }
});

//message handler for communication between offscreen.js and background.js. this listener handles messages sent from the offscreen document, which is responsible for capturing media. 
// it listens for three types of messages: 'DOWNLOAD' to trigger a file download, 'STOP_DONE' to reset the state after saving is complete, and 'ERROR' to handle any errors that occur during recording or saving.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  
  // permissions handler listens for a message indicating that permissions are missing. If the offscreen document encounters a permissions issue (e.g., user denied access to capture), it sends this message to the background script, which then resets the state and opens the setup page to prompt the user to grant the necessary permissions.
  if (msg.type === 'PERMISSIONS_REQUIRED') {
      console.warn("Permissions missing. Opening setup page.");
      // Reset the badge and state
      chrome.storage.local.set({ recordingState: 'IDLE' });
      chrome.action.setBadgeText({ text: "" });
      // Automatically pop open setup.html
      chrome.runtime.openOptionsPage(); 
  }

  // download the recorded file using the Chrome Downloads API. The message should include the URL of the recorded media and its file type.
  //  The filename is generated using a timestamp to ensure uniqueness. This allows the user to save the recorded meeting directly to their device.
  //download handler receives the URLs from offscreen.js
  if (msg.type === 'DOWNLOAD') {
    const timestamp = Math.floor(Date.now() / 1000);
    const filename = `meeting_${timestamp}_${msg.fileType}.webm`;
//triggers the Chrome download manager. 
    chrome.downloads.download({
      url: msg.url,
      filename: filename,
      saveAs: false // Automatically save to the default downloads folder
    });
  }

  // stop done handler resets the extension state to IDLE and clears the badge text. This is called by offscreen.js once the recording has been successfully stopped and the files have been processed, 
  // allowing the user to start a new recording if they wish.
  if (msg.type === 'STOP_DONE') {
      console.log("Saving finished. State -> IDLE");
      chrome.storage.local.set({ recordingState: 'IDLE' });
      chrome.action.setBadgeText({ text: "" });
  }

  //error handler logs the error, resets the state to IDLE, and updates the badge text to indicate an error. This allows the user to be aware that something went wrong and gives them the opportunity to try recording again without being stuck in an unusable state.
  //called if the offscreen recorder crashes or encounters an error during recording or saving. It ensures that the extension doesn't remain in a broken state and provides feedback to the user about the error.
  if (msg.type === 'ERROR') {
      console.error(" Offscreen reported error.");
      chrome.storage.local.set({ recordingState: 'IDLE' });
      chrome.action.setBadgeText({ text: "ERR" });
  }
});
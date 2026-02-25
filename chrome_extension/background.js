// Force reset state on load to prevent getting stuck
chrome.runtime.onStartup.addListener(() => {
    chrome.storage.local.set({ recordingState: 'IDLE' });
    chrome.action.setBadgeText({ text: "" });
});

chrome.action.onClicked.addListener(async (tab) => {
  // 1. GET SAVED STATE (Recovers memory if SW went to sleep)
  const result = await chrome.storage.local.get('recordingState');
  let currentState = result.recordingState || 'IDLE';

  console.log(`🖱️ Clicked! Current State: ${currentState}`);

  if (currentState === 'RECORDING') {
    console.log("Command: Stop");
    
    // Set to SAVING immediately to block double-clicks
    await chrome.storage.local.set({ recordingState: 'SAVING' });
    chrome.action.setBadgeText({ text: "SAVE" });
    chrome.action.setBadgeBackgroundColor({ color: "#0000FF" });

    try {
        chrome.runtime.sendMessage({ type: 'STOP', target: 'offscreen' });
    } catch (e) {
        console.warn("Offscreen unreachable. Force resetting.");
        await chrome.storage.local.set({ recordingState: 'IDLE' });
        chrome.action.setBadgeText({ text: "" });
    }
    return;
  }

  if (currentState === 'SAVING') {
    console.warn(" Ignored click: Still saving previous files.");
    return; 
  }

  if (currentState === 'IDLE') {
      console.log("Command: Start");
      chrome.action.setBadgeText({ text: "..." });

      const existingContexts = await chrome.runtime.getContexts({});
      const offscreen = existingContexts.find(c => c.contextType === 'OFFSCREEN_DOCUMENT');
      
      if (!offscreen) {
        await chrome.offscreen.createDocument({
          url: 'offscreen.html',
          reasons: ['USER_MEDIA'],
          justification: 'Recording tab audio'
        });
      }

      try {
          const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
          
          chrome.runtime.sendMessage({ type: 'START', target: 'offscreen', data: streamId });

          // SAVE STATE TO DISK
          await chrome.storage.local.set({ recordingState: 'RECORDING' });
          
          chrome.action.setBadgeText({ text: "REC" });
          chrome.action.setBadgeBackgroundColor({ color: "#FF0000" });

      } catch (err) {
          console.error(" Capture Error:", err);
          
          // Only kill zombies if we are truly stuck
          if (err.message.includes("active stream")) {
              console.warn(" Zombie stream detected. Cleaning up.");
              if (offscreen) await chrome.offscreen.closeDocument();
              
              chrome.action.setBadgeText({ text: "RST" });
              chrome.action.setBadgeBackgroundColor({ color: "#FFA500" });
              setTimeout(() => chrome.action.setBadgeText({ text: "" }), 1000);
          } else {
              chrome.action.setBadgeText({ text: "ERR" });
          }
          
          // Reset state on error
          await chrome.storage.local.set({ recordingState: 'IDLE' });
      }
  }
});

// --- MESSAGE HANDLER ---
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  
  // 1. DOWNLOAD
  if (msg.type === 'DOWNLOAD') {
    const timestamp = Math.floor(Date.now() / 1000);
    const filename = `meeting_${timestamp}_${msg.fileType}.webm`;

    chrome.downloads.download({
      url: msg.url,
      filename: filename,
      saveAs: false
    });
  }

  // 2. STOP DONE (Reset State)
  if (msg.type === 'STOP_DONE') {
      console.log("✅ Saving finished. State -> IDLE");
      chrome.storage.local.set({ recordingState: 'IDLE' });
      chrome.action.setBadgeText({ text: "" });
  }

  // 3. ERROR (Reset State)
  if (msg.type === 'ERROR') {
      console.error(" Offscreen reported error.");
      chrome.storage.local.set({ recordingState: 'IDLE' });
      chrome.action.setBadgeText({ text: "ERR" });
  }
});
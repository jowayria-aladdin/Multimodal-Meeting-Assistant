console.log("✅ Modular Recorder Loaded (Speaker Mode)");

let audioRecorder = null;
let screenRecorder = null;
let webcamRecorder = null;

let audioChunks = [];
let screenChunks = [];
let webcamChunks = [];

let activeStream = null;
let camStream = null;
let activeCtx = null;

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'START') {
        startCapture(msg.data);
    } else if (msg.type === 'STOP') {
        stopCapture();
    }
});

async function startCapture(streamId) {
    audioChunks = [];
    screenChunks = [];
    webcamChunks = [];

    // 1. GET SCREEN
    try {
        activeStream = await navigator.mediaDevices.getUserMedia({
            audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
            video: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId, maxWidth: 1920, maxHeight: 1080 } }
        });
    } catch (err) {
        console.error(" Critical: Could not capture screen.", err);
        chrome.runtime.sendMessage({ type: 'ERROR', message: 'SCREEN_FAIL' });
        return; 
    }

    // 2. GET MIC (With Echo Cancellation)
    let micStream = null;
    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                echoCancellation: true,  //  Must be TRUE since you have no headphones
                noiseSuppression: true, 
                autoGainControl: true 
            }
        });
    } catch (err) {
        console.warn(" Mic denied/unavailable.");
    }

    // 3. GET WEBCAM
    try {
        camStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });
    } catch (err) {
        camStream = null;
    }

    // 4. AUDIO MIXING (The Ticking Fix)
    
    // "latencyHint: 'playback'" is the magic fix for the ticking/glitching noise
    activeCtx = new AudioContext({ latencyHint: 'playback' }); 
    const dest = activeCtx.createMediaStreamDestination();
    
    if (activeStream && micStream) {
        const tabSource = activeCtx.createMediaStreamSource(activeStream);
        const micSource = activeCtx.createMediaStreamSource(micStream);

        // --- VOLUME SETTINGS ---
        
        // Tab: Keep low so it doesn't feed back into mic
        const tabGain = activeCtx.createGain(); 
        tabGain.gain.value = 0.6; 

        // Mic: REDUCED from 8.0 to 3.0
        // 8.0 amplifies "room hiss" (TV noise). 3.0 is plenty loud.
        const micGain = activeCtx.createGain(); 
        micGain.gain.value = 3.0; 
        
        const compressor = activeCtx.createDynamicsCompressor();
        compressor.threshold.value = -20;
        compressor.knee.value = 30;
        compressor.ratio.value = 12;
        compressor.attack.value = 0.003;
        compressor.release.value = 0.25;

        micSource.connect(micGain).connect(compressor);
        tabSource.connect(tabGain);

        const merger = activeCtx.createChannelMerger(2);
        tabGain.connect(merger, 0, 0);       
        compressor.connect(merger, 0, 1);    
        merger.connect(dest);
        
        // Monitor (Hear the meeting)
        tabSource.connect(activeCtx.destination); 
        
    } else {
        const tabSource = activeCtx.createMediaStreamSource(activeStream);
        tabSource.connect(dest);
        tabSource.connect(activeCtx.destination);
    }

    // 5. START RECORDERS
    // Using 1000ms chunks to keep data flow smooth
    audioRecorder = new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 128000 });
    audioRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    audioRecorder.start(1000);

    const screenOnlyStream = new MediaStream([activeStream.getVideoTracks()[0]]);
    screenRecorder = new MediaRecorder(screenOnlyStream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 2500000 });
    screenRecorder.ondataavailable = (e) => { if (e.data.size > 0) screenChunks.push(e.data); };
    screenRecorder.start(1000);

    if (camStream) {
        webcamRecorder = new MediaRecorder(camStream, { mimeType: 'video/webm;codecs=vp9' });
        webcamRecorder.ondataavailable = (e) => { if (e.data.size > 0) webcamChunks.push(e.data); };
        webcamRecorder.start(1000);
    }

    console.log(` Recording Started.`);
}

function stopCapture() {
    console.log(" Stopping.");
    if (!activeStream) {
        chrome.runtime.sendMessage({ type: 'STOP_DONE' });
        return;
    }

    const stopRecorder = (recorder, chunks, type) => {
        return new Promise((resolve) => {
            if (!recorder || recorder.state === 'inactive') { resolve(null); return; }
            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: type === 'audio' ? 'audio/webm' : 'video/webm' });
                resolve(URL.createObjectURL(blob));
            };
            recorder.stop();
        });
    };

    activeStream.getTracks().forEach(t => t.stop());
    if (camStream) camStream.getTracks().forEach(t => t.stop());
    if (activeCtx) activeCtx.close();

    Promise.all([
        stopRecorder(screenRecorder, screenChunks, 'video'),
        stopRecorder(webcamRecorder, webcamChunks, 'video'),
        stopRecorder(audioRecorder, audioChunks, 'audio')
    ]).then(([screenUrl, webcamUrl, audioUrl]) => {
        
        if (screenUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'video', url: screenUrl });
        if (webcamUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'webcam', url: webcamUrl });
        if (audioUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'audio', url: audioUrl });
        
        setTimeout(() => {
            chrome.runtime.sendMessage({ type: 'STOP_DONE' });
        }, 500);
    });
}
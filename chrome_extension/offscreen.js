/**
 offscreen logic
 overall functionality:
 this script runs in a hidden offscreen document. It captures the computer's audio,the user's microphone, and the webcam. 
 it performs realtime audio mixing to ensure the meeting audio and the user's voice are both clear for the RAG system.
 */

console.log(" Modular Recorder Loaded (Speaker Mode)");
// Global variables to hold the MediaRecorder instances and the binary data chunks for audio, screen, and webcam recordings.
//  These variables are used to manage the recording state and store the captured media data until it is ready to be processed and downloaded.
let audioRecorder = null;
let screenRecorder = null;
let webcamRecorder = null;

let audioChunks = [];
let screenChunks = [];
let webcamChunks = [];
// Stream objects to manage active hardware/tab connections
let activeStream = null; // Tab audio/video
let camStream = null;    // Webcam video
let activeCtx = null;    // The Web Audio context for mixing

// communication handler listens for messages from the background script to start or stop recording. When it receives a 'START' message, it triggers the startCapture function with the provided stream ID.
//  When it receives a 'STOP' message, it calls the stopCapture function to end the recording and process the captured media.
// Listens for START or STOP commands from background.js
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'START') {
        startCapture(msg.data);
    } else if (msg.type === 'STOP') {
        stopCapture();
    }
});
// Initiates all captures and sets up the Audio Pipeline.
async function startCapture(streamId) {
    // Reset data arrays for a fresh recording session
    audioChunks = [];
    screenChunks = [];
    webcamChunks = [];

    // get screen stream
    //  using the provided stream ID. This is necessary to capture the audio and video from the active tab. The function uses the MediaDevices API to access the media stream, specifying the appropriate constraints for audio and video capture.
    // If the capture fails  due to permissions issues or an active stream, it catches the error and sends an error message back to the background script, which can then update the UI accordingly.
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

    // get microphone stream
    //  to mix with tab audio. This allows the user's voice to be included in the recording, which is essential for meetings. The function requests access to the user's microphone with specific audio constraints to improve quality, such as echo cancellation and noise suppression. If the user denies access or if there is an issue with the microphone, 
    // it catches the error and logs a warning, but it does not stop the recording process since the tab audio can still be captured. 
    let micStream = null;
    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { 
                echoCancellation: true,  // Prevents the meeting audio from looping back into the mic
                noiseSuppression: true, 
                autoGainControl: true 
            }
        });
    } catch (err) {
        console.warn(" Mic denied/unavailable.");
    }

    // get webcam stream "optional" since not everyone will want it
    // , and it can be a resource hog. If access is denied or if there is no webcam, it simply sets camStream to null and continues without it. This allows the extension to be flexible and work in a variety of environments without requiring all media sources to be available.
    try {
        camStream = await navigator.mediaDevices.getUserMedia({
            audio: false, // We only need the video; audio is handled by the mic stream
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }
        });
    } catch (err) {
        camStream = null;
    }

    // audio mixing setup using the Web Audio API. It creates an audio context and sets up a processing graph to mix the tab audio and microphone input. The tab audio is kept at a moderate level to prevent feedback, while the microphone input is amplified to ensure the user's voice is clear in the recording. A dynamics compressor is used on the microphone input to even out volume levels and reduce background noise. The mixed audio is then sent to a MediaStreamDestination node, which allows it to be recorded by the MediaRecorder. If there are any issues with setting up the audio context or processing graph, it catches the error and logs it, but it does not stop the recording process since the video can still be captured.
    
// 'latencyHint: playback' prevents the "popping/clicking" sounds in recorded audio.
    activeCtx = new AudioContext({ latencyHint: 'playback' }); 
    const dest = activeCtx.createMediaStreamDestination();
    
    if (activeStream && micStream) {
        const tabSource = activeCtx.createMediaStreamSource(activeStream);
        const micSource = activeCtx.createMediaStreamSource(micStream);

        // gain volume settings for tab and mic
        // to ensure a good balance in the recording. The tab audio is reduced to prevent it from overpowering the microphone input, which is amplified to make sure the user's voice is clearly audible. These settings can be adjusted based on user feedback or specific use cases, but they provide a good starting point for most meeting scenarios.
        
// Tab Volume: Lowered to 0.6 so it doesn't drown out the speaker.
        const tabGain = activeCtx.createGain(); 
        tabGain.gain.value = 0.6; 

        // Mic Volume: Boosted to 3.0 to ensure the speaker's voice is clear for ASR.
        const micGain = activeCtx.createGain(); 
        micGain.gain.value = 3.0; 
        //dynamic compressor settings to improve mic audio quality. 
        // The compressor helps to even out the volume of the microphone input, making quiet sounds louder and loud sounds quieter.
        // This is especially useful in meeting recordings where the speaker may move around or speak at varying volumes.
        //  The specific settings for threshold, knee, ratio, attack, and release can be adjusted based on the desired audio characteristics to provide a good starting point for clear and balanced audio.
        
        const compressor = activeCtx.createDynamicsCompressor();
        compressor.threshold.value = -20;
        compressor.knee.value = 30;
        compressor.ratio.value = 12;
        compressor.attack.value = 0.003;
        compressor.release.value = 0.25;
        // Routing: Mic -> Gain -> Compressor -> Merger
        micSource.connect(micGain).connect(compressor);
        // Routing: Tab -> Gain -> Merger
        tabSource.connect(tabGain);

        const merger = activeCtx.createChannelMerger(2);
        tabGain.connect(merger, 0, 0);       
        compressor.connect(merger, 0, 1);    
        merger.connect(dest);
        
        // monitor connects the tab audio to the user's speakers so they can still hear the meeting.
        tabSource.connect(activeCtx.destination); 
        
    } else {
        // Fallback if mic is missing: just record tab audio.
        const tabSource = activeCtx.createMediaStreamSource(activeStream);
        tabSource.connect(dest);
        tabSource.connect(activeCtx.destination);
    }

    //start recorders for audio, screen, and webcam if available. 
    // The MediaRecorder API is used to capture the mixed audio from the Web Audio context, the video from the active tab, and the webcam video if it is available. Each recorder is set up with appropriate MIME types and bitrates to ensure good quality recordings.
    // The ondataavailable event handlers are used to collect the recorded data chunks into arrays, which will later be processed into Blob objects for downloading.
    //  The recorders are started with a timeslice of 1000ms to ensure that data is available in manageable chunks, which helps to prevent data loss and allows for smoother processing.

    // Recorder A: Mixed Audio For ASR/Transcribing
    audioRecorder = new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus', audioBitsPerSecond: 128000 });
    audioRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    audioRecorder.start(1000);
    // Recorder B: Screen Capture (For Visual Reference)
    const screenOnlyStream = new MediaStream([activeStream.getVideoTracks()[0]]);
    screenRecorder = new MediaRecorder(screenOnlyStream, { mimeType: 'video/webm;codecs=vp9', videoBitsPerSecond: 2500000 });
    screenRecorder.ondataavailable = (e) => { if (e.data.size > 0) screenChunks.push(e.data); };
    screenRecorder.start(1000);
    // Recorder C: Webcam Capture (For the Sign Language Model)
    if (camStream) {
        webcamRecorder = new MediaRecorder(camStream, { mimeType: 'video/webm;codecs=vp9' });
        webcamRecorder.ondataavailable = (e) => { if (e.data.size > 0) webcamChunks.push(e.data); };
        webcamRecorder.start(1000);
    }

    console.log(` Recording Started.`);
}
//stops all recorders, cleans up hardware tracks, and sends files to background.js
function stopCapture() {
    console.log(" Stopping.");
    if (!activeStream) {
        chrome.runtime.sendMessage({ type: 'STOP_DONE' });
        return;
    }
    // helper function to stop a recorder and return the data as a URL
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
// Kill all hardware tracks (turns off the green "recording" dots in Chrome)
    activeStream.getTracks().forEach(t => t.stop());
    if (camStream) camStream.getTracks().forEach(t => t.stop());
    if (activeCtx) activeCtx.close();
// Wait for all recorders to finish processing their final chunks
    Promise.all([
        stopRecorder(screenRecorder, screenChunks, 'video'),
        stopRecorder(webcamRecorder, webcamChunks, 'video'),
        stopRecorder(audioRecorder, audioChunks, 'audio')
    ]).then(([screenUrl, webcamUrl, audioUrl]) => {
        // Send the final file URLs to background.js to trigger the downloads
        if (screenUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'video', url: screenUrl });
        if (webcamUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'webcam', url: webcamUrl });
        if (audioUrl) chrome.runtime.sendMessage({ type: 'DOWNLOAD', fileType: 'audio', url: audioUrl });
        // Small delay before resetting the UI to IDLE
        setTimeout(() => {
            chrome.runtime.sendMessage({ type: 'STOP_DONE' });
        }, 500);
    });
}
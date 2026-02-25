/**
 permissions logic
 overall functionality:
 This script handles the actual requests for hardware access. 
 Since the background service worker cannot trigger permission prompts, 
 this  ui driven script ensures the browser registers the user's consent 
 for the microphone (ASR) and camera (Sign Language model).
 */

// Ask for Microphone Only
// This listener handles the click on the "Grant Microphone" button.
document.getElementById('micBtn').addEventListener('click', async () => {
    try {
        await navigator.mediaDevices.getUserMedia({ audio: true });//triggers the browser's native popup.
        // We request 'audio: true' to capture the user's voice for the RAG system.
        alert(" Microphone Permission Granted!");
    } catch (err) {
        alert(" Microphone Denied: " + err.message);
    }
});

// Ask for Camera Only
// This listener handles the click on the "Grant Camera" button.
document.getElementById('camBtn').addEventListener('click', async () => {
    try {
        // We request 'video: true' specifically for the Sign Language model input.
        await navigator.mediaDevices.getUserMedia({ video: true });
        alert(" Camera Permission Granted!");
    } catch (err) {
        alert(" Camera Denied: " + err.message);
    }
});
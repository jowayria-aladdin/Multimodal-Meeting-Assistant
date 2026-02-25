// Ask for Microphone Only
document.getElementById('micBtn').addEventListener('click', async () => {
    try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
        alert(" Microphone Permission Granted!");
    } catch (err) {
        alert(" Microphone Denied: " + err.message);
    }
});

// Ask for Camera Only
document.getElementById('camBtn').addEventListener('click', async () => {
    try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        alert(" Camera Permission Granted!");
    } catch (err) {
        alert(" Camera Denied: " + err.message);
    }
});
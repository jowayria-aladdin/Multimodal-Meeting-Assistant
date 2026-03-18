
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import pickle
import cv2
import mediapipe as mp
import tensorflow as tf
import tempfile, os
from typing import List

app = FastAPI(title="Sign Language API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & labels at startup ──────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "Transformer_20260117_193329_best.keras")
PKL_PATH      = os.path.join(BASE_DIR, "sign_to_class.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(PKL_PATH, "rb") as f:
    sign_to_class = pickle.load(f)

class_to_sign = {v: k for k, v in sign_to_class.items()}
print(f"✓ Model loaded | {len(class_to_sign)} classes")

# ── Response models ────
class Prediction(BaseModel):
    sign: str
    confidence: float

class PredictResponse(BaseModel):
    sign: str
    confidence: float
    top10: List[Prediction]

# ── Preprocessing — same as test.py ────
NUM_FRAMES   = 40
mp_holistic  = mp.solutions.holistic

def extract_keypoints(results):
    pose = (np.array([[lm.x, lm.y, lm.z] for lm in
                      results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 3, dtype=np.float32))
    lh   = (np.array([[lm.x, lm.y, lm.z] for lm in
                      results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
    rh   = (np.array([[lm.x, lm.y, lm.z] for lm in
                      results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
    return np.concatenate([pose, lh, rh])   # (225,)

def adjust_landmarks(arr, center):
    if np.all(arr == 0):
        return arr
    r = arr.reshape(-1, 3)
    return (r - np.tile(center, (len(r), 1))).reshape(-1)

def normalize_landmarks(sequence):
    out = []
    for f in sequence:
        pose = f[0:99];    nose = pose[0:3]
        lh   = f[99:162];  lhw  = lh[0:3]
        rh   = f[162:225]; rhw  = rh[0:3]
        out.append(np.concatenate([
            adjust_landmarks(pose, nose),
            adjust_landmarks(lh,   lhw),
            adjust_landmarks(rh,   rhw),
        ]))
    return np.array(out, dtype=np.float32)

def calculate_velocity_features(seq):
    delta    = np.diff(seq, axis=0)
    velocity = np.concatenate([np.zeros((1, seq.shape[1]), dtype=seq.dtype), delta], axis=0)
    return np.concatenate([seq, velocity], axis=1)   # (40, 450)

def normalize_sequence_length(seq, target=NUM_FRAMES):
    if len(seq) == 0:
        return np.zeros((target, seq.shape[1] if seq.ndim > 1 else 225), dtype=np.float32)
    if len(seq) < target:
        pad = np.repeat(seq[-1:], target - len(seq), axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    elif len(seq) > target:
        idx = np.linspace(0, len(seq) - 1, target, dtype=int)
        seq = seq[idx]
    return seq.astype(np.float32)

def extract_sequence_from_video(video_path: str) -> np.ndarray:
    cap  = cv2.VideoCapture(video_path)
    kps  = []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            kps.append(extract_keypoints(results))
    cap.release()
    return np.array(kps, dtype=np.float32)

# ── Endpoints ─────
@app.get("/health")
def health():
    return {"status": "ok", "classes": len(class_to_sign)}

@app.post("/predict", response_model=PredictResponse)
async def predict(video: UploadFile = File(...)):
    # Save uploaded file to a temp path
    suffix   = os.path.splitext(video.filename)[-1] or ".webm"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name

        # 1. Extract raw keypoints from every frame
        sequence = extract_sequence_from_video(tmp_path)
        if len(sequence) == 0:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        # pad short sequences instead of rejecting them
        if len(sequence) < NUM_FRAMES:
            pad = np.repeat(sequence[-1:], NUM_FRAMES - len(sequence), axis=0)
            sequence = np.concatenate([sequence, pad], axis=0)

        # 2. Preprocess — same pipeline as test.py
        sequence = normalize_sequence_length(sequence, NUM_FRAMES)  # (40, 225)
        sequence = normalize_landmarks(sequence)                     # (40, 225)
        sequence = calculate_velocity_features(sequence)             # (40, 450)

        # 3. Inference
        batch  = np.expand_dims(sequence, axis=0)                    # (1, 40, 450)
        probs  = model.predict(batch, verbose=0)[0]

        top10_idx = np.argsort(probs)[-10:][::-1]
        top10 = [
            Prediction(
                sign=str(class_to_sign.get(int(i), "Unknown")),
                confidence=float(probs[i])
            )
            for i in top10_idx
        ]

        return PredictResponse(
            sign=top10[0].sign,
            confidence=top10[0].confidence,
            top10=top10
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)


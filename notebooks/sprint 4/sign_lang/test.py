import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import time
from PIL import Image, ImageDraw, ImageFont
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# ============= CONFIGURATION =============
import tensorflow as tf

# Add at the top of your script
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU found - running on CPU")

class Config:
    MODEL_PATH = "models/Transformer_20260117_193329_best.keras"  # Update with your model path
    LABELS_PATH = "KARSL-502_Labels.xlsx"  # Update with your labels file
    SIGN_TO_CLASS_PATH = "sign_to_class.pkl"  # Update with your sign_to_class pickle
    NUM_FRAMES = 40
    MP_CONFIDENCE = 0.5
    BUFFER_SIZE = 40  # Must match NUM_FRAMES
    PREDICTION_INTERVAL = 4  # Predict every 4 frames
    
    # Display settings
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480  # Standard webcam aspect ratio (4:3)
    WINDOW_HEIGHT = 900  # Total window height for predictions
    SIDEBAR_WIDTH = 400
    PREDICTION_THRESHOLD = 0.3  # Minimum confidence to display
    TOP_K = 10  # Number of top predictions to show
    
    # Clear button settings
    BUTTON_WIDTH = 120
    BUTTON_HEIGHT = 40
    BUTTON_MARGIN = 15

config = Config()

# ============= LANDMARK EXTRACTION =============
class LandmarkExtractor:
    def __init__(self, min_detection_confidence=0.5):
        self.holistic_model = mp.solutions.holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        self.num_landmarks = (33 + 21 + 21) * 3
    
    def extract(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.holistic_model.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            pose_landmarks = np.zeros(33 * 3, dtype=np.float32)
            lh_landmarks = np.zeros(21 * 3, dtype=np.float32)
            rh_landmarks = np.zeros(21 * 3, dtype=np.float32)
            
            if results.pose_landmarks:
                pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if results.left_hand_landmarks:
                lh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks:
                rh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            
            landmarks = np.concatenate([pose_landmarks, lh_landmarks, rh_landmarks])
            return landmarks
        except Exception as e:
            return np.zeros(self.num_landmarks, dtype=np.float32)
    
    def close(self):
        self.holistic_model.close()

# ============= PREPROCESSING FUNCTIONS =============
def adjust_landmarks(arr, center):
    if np.all(arr == 0):
        return arr
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = arr_reshaped - center_repeated
    return arr_adjusted.reshape(-1)

def normalize_landmarks(sequence):
    seq = sequence.copy()
    if seq.shape[0] == 0:
        return seq
    
    T, num_features = seq.shape
    normalized_seq = []
    
    for t in range(T):
        frame_features = seq[t, :]
        pose = frame_features[0:99]
        nose = pose[0:3]
        pose_adjusted = adjust_landmarks(pose, nose)
        
        lh = frame_features[99:162]
        lh_wrist = lh[0:3]
        lh_adjusted = adjust_landmarks(lh, lh_wrist)
        
        rh = frame_features[162:225]
        rh_wrist = rh[0:3]
        rh_adjusted = adjust_landmarks(rh, rh_wrist)
        
        normalized_frame = np.concatenate([pose_adjusted, lh_adjusted, rh_adjusted])
        normalized_seq.append(normalized_frame)
    
    return np.array(normalized_seq, dtype=np.float32)

def calculate_velocity_features(normalized_sequence):
    T, num_features = normalized_sequence.shape
    delta_features = np.diff(normalized_sequence, axis=0)
    zero_delta = np.zeros((1, num_features), dtype=normalized_sequence.dtype)
    velocity_features = np.concatenate([zero_delta, delta_features], axis=0)
    final_features = np.concatenate([normalized_sequence, velocity_features], axis=1)
    return final_features

def normalize_sequence_length(seq, target_length):
    if len(seq) == 0:
        return np.zeros((target_length, seq.shape[1]), dtype=np.float32)
    
    if len(seq) < target_length:
        pad_len = target_length - len(seq)
        pad = np.repeat(seq[-1:], pad_len, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    elif len(seq) > target_length:
        indices = np.linspace(0, len(seq) - 1, target_length, dtype=int)
        seq = seq[indices]
    
    return seq.astype(np.float32)

# ============= HELPER FUNCTIONS =============
def put_arabic_text(img, text, position, font_size=40, color=(0, 255, 0)):
    """Draw Arabic text on OpenCV image using PIL"""
    text = str(text)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    try:
        reshaped_text = reshape(text)
        bidi_text = get_display(reshaped_text)
    except:
        bidi_text = text
    
    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def is_point_in_button(x, y, button_rect):
    """Check if point (x, y) is inside button rectangle"""
    bx, by, bw, bh = button_rect
    return bx <= x <= bx + bw and by <= y <= by + bh

def draw_button(frame, rect, text, is_hovered=False):
    """Draw a button on the frame"""
    x, y, w, h = rect
    
    # Button colors
    if is_hovered:
        bg_color = (80, 80, 220)
        text_color = (255, 255, 255)
    else:
        bg_color = (60, 60, 180)
        text_color = (230, 230, 230)
    
    # Draw button background with rounded corners effect
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 255), 2)
    
    # Draw text centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

# Mouse callback for button clicks
mouse_x, mouse_y = -1, -1
button_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, button_clicked
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        button_clicked = True

# ============= LOAD MODEL AND LABELS =============
print("Loading model...")
model = load_model(config.MODEL_PATH)
print("✓ Model loaded successfully")

print("Loading labels...")
df_labels = pd.read_excel(config.LABELS_PATH)
with open(config.SIGN_TO_CLASS_PATH, "rb") as f:
    sign_to_class = pickle.load(f)

class_to_sign = {v: k for k, v in sign_to_class.items()}
print(f"✓ Loaded {len(class_to_sign)} sign classes")

# ============= INITIALIZE =============
extractor = LandmarkExtractor(min_detection_confidence=config.MP_CONFIDENCE)
frame_buffer = deque(maxlen=config.BUFFER_SIZE)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set up mouse callback
cv2.namedWindow('Sign Language Recognition')
cv2.setMouseCallback('Sign Language Recognition', mouse_callback)

print("\n" + "="*60)
print("REAL-TIME SIGN LANGUAGE RECOGNITION")
print("="*60)
print("Press 'q' to quit")
print("Press 'l' to toggle landmarks display")
print("Press 'c' or click button to clear buffer")
print("="*60 + "\n")

# Tracking variables
current_prediction = "No prediction"
current_confidence = 0.0
top_k_predictions = []
inference_time = 0.0
fps_history = deque(maxlen=30)
show_landmarks = False
frame_counter = 0

# ============= MAIN LOOP =============
try:
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks = extractor.extract(frame)
        frame_buffer.append(landmarks)
        
        frame_counter += 1
        
        # Make prediction every PREDICTION_INTERVAL frames if buffer is full
        if len(frame_buffer) == config.BUFFER_SIZE and frame_counter % config.PREDICTION_INTERVAL == 0:
            sequence = np.array(list(frame_buffer))
            sequence = normalize_landmarks(sequence)
            sequence = calculate_velocity_features(sequence)
            
            sequence_batch = np.expand_dims(sequence, axis=0)
            inference_start = time.time()
            prediction_probs = model.predict(sequence_batch, verbose=0)[0]
            inference_time = (time.time() - inference_start) * 1000
            
            top_k_indices = np.argsort(prediction_probs)[-config.TOP_K:][::-1]
            top_k_predictions = [(class_to_sign.get(idx, "Unknown"), prediction_probs[idx]) 
                                for idx in top_k_indices]
            
            predicted_class = top_k_indices[0]
            confidence = prediction_probs[predicted_class]
            
            if confidence >= config.PREDICTION_THRESHOLD:
                predicted_sign = class_to_sign.get(predicted_class, "Unknown")
                current_prediction = str(predicted_sign)
                current_confidence = confidence
        
        fps = 1.0 / (time.time() - start_time)
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # ============= DRAW UI =============
        h, w = frame.shape[:2]
        total_width = w + config.SIDEBAR_WIDTH
        
        # Create canvas with extended height for predictions
        canvas_height = max(h, config.WINDOW_HEIGHT)
        base_frame = np.zeros((canvas_height, total_width, 3), dtype=np.uint8)
        display_frame_video = frame.copy()
        
        # Calculate button position (top right of video area)
        button_rect = (
            w - config.BUTTON_WIDTH - config.BUTTON_MARGIN,
            config.BUTTON_MARGIN,
            config.BUTTON_WIDTH,
            config.BUTTON_HEIGHT
        )
        
        # Check if mouse is hovering over button
        is_hovered = is_point_in_button(mouse_x, mouse_y, button_rect)
        
        # Check if button was clicked
        if button_clicked and is_hovered:
            frame_buffer.clear()
            current_prediction = "Buffer cleared"
            current_confidence = 0.0
            top_k_predictions = []
            print("Buffer cleared!")
        
        button_clicked = False  # Reset click flag
        
        # Draw landmarks if enabled
        if show_landmarks:
            current_landmarks = frame_buffer[-1] if len(frame_buffer) > 0 else None
            if current_landmarks is not None:
                landmarks_reshaped = current_landmarks.reshape(75, 3)
                
                for i in range(33):
                    x, y = int(landmarks_reshaped[i][0] * w), int(landmarks_reshaped[i][1] * h)
                    if x > 0 and y > 0:
                        cv2.circle(display_frame_video, (x, y), 3, (255, 0, 0), -1)
                
                for i in range(33, 54):
                    x, y = int(landmarks_reshaped[i][0] * w), int(landmarks_reshaped[i][1] * h)
                    if x > 0 and y > 0:
                        cv2.circle(display_frame_video, (x, y), 3, (0, 255, 0), -1)
                
                for i in range(54, 75):
                    x, y = int(landmarks_reshaped[i][0] * w), int(landmarks_reshaped[i][1] * h)
                    if x > 0 and y > 0:
                        cv2.circle(display_frame_video, (x, y), 3, (0, 165, 255), -1)
        
        base_frame[:h, :w] = display_frame_video
        # Sidebar extends for full canvas height
        base_frame[:, w:] = (40, 40, 40)
        
        # Fill area below video with dark background
        if canvas_height > h:
            base_frame[h:, :w] = (20, 20, 20)
        
        # Draw header on video
        overlay = display_frame_video.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(overlay, 0.6, display_frame_video, 0.4, 0)
        base_frame[:h, :w] = frame_with_overlay
        
        # Draw clear buffer button
        draw_button(base_frame, button_rect, "Clear Buffer", is_hovered)
        
        cv2.putText(base_frame, "Predicted Sign:", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        base_frame = put_arabic_text(base_frame, current_prediction, (20, 50), 
                                    font_size=40, color=(0, 255, 0))
        
        # ============= SIDEBAR - SYSTEM STATS =============
        sidebar_x = w + 10
        y_offset = 20
        
        cv2.putText(base_frame, "System Status", (sidebar_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)
        y_offset += 30
        
        stats = [
            ("FPS", f"{avg_fps:.1f}", (0, 255, 0)),
            ("Inference", f"{inference_time:.1f}ms", (255, 200, 0)),
            ("Buffer", f"{len(frame_buffer)}/{config.BUFFER_SIZE}", (150, 150, 255)),
            ("Pred. Rate", f"1/{config.PREDICTION_INTERVAL} frames", (255, 180, 100)),
            ("Landmarks", "ON" if show_landmarks else "OFF", (255, 150, 150))
        ]
        
        for label, value, color in stats:
            cv2.putText(base_frame, f"{label}:", (sidebar_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(base_frame, value, (sidebar_x + 90, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 22
        
        y_offset += 5
        cv2.line(base_frame, (sidebar_x, y_offset), (total_width - 10, y_offset), 
                (100, 100, 100), 2)
        y_offset += 20
        
        # ============= SIDEBAR - TOP 10 PREDICTIONS =============
        cv2.putText(base_frame, "Top 10 Predictions", (sidebar_x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 2)
        y_offset += 30
        
        predictions_start_y = y_offset
        temp_y = predictions_start_y
        prediction_positions = []
        
        if top_k_predictions:
            for i, (sign, conf) in enumerate(top_k_predictions):
                rank_color = (0, 255, 0) if i == 0 else (200, 200, 200)
                rank_text = f"#{i+1}"
                perc_text = f"{conf*100:.1f}%"
                
                cv2.putText(base_frame, rank_text, (sidebar_x, temp_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, rank_color, 2 if i == 0 else 1)
                cv2.putText(base_frame, perc_text, 
                           (sidebar_x + config.SIDEBAR_WIDTH - 60, temp_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, rank_color, 1)
                
                bar_start_x = sidebar_x + 35
                bar_width_max = config.SIDEBAR_WIDTH - 105
                bar_width = int(bar_width_max * conf)
                
                cv2.rectangle(base_frame, (bar_start_x, temp_y - 10), 
                            (bar_start_x + bar_width_max, temp_y), (60, 60, 60), -1)
                bar_color = (0, 255, 0) if i == 0 else (100, 150, 255)
                cv2.rectangle(base_frame, (bar_start_x, temp_y - 10), 
                            (bar_start_x + bar_width, temp_y), bar_color, -1)
                
                temp_y += 12
                prediction_positions.append((sign, bar_start_x, temp_y, 
                                            rank_color if i == 0 else (255, 255, 255)))
                temp_y += 28
            
            display_frame = base_frame.copy()
            for sign, x_pos, y_pos, color in prediction_positions:
                display_frame = put_arabic_text(display_frame, str(sign), 
                                               (x_pos, y_pos - 13), 
                                               font_size=18, color=color)
        else:
            display_frame = base_frame.copy()
            cv2.putText(display_frame, "Waiting for data...", 
                       (sidebar_x + 20, predictions_start_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.imshow('Sign Language Recognition', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('c'):
            frame_buffer.clear()
            current_prediction = "Buffer cleared"
            current_confidence = 0.0
            top_k_predictions = []
            print("Buffer cleared!")

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("✓ Done")
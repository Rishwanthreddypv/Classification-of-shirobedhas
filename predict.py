import os
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_PATH = r"C:\Pinacle\models\face68_lstm_model.keras"
CLASSES_PATH = r"C:\Pinacle\models\classes.npy"
VIDEO_PATH = r"G:\My Drive\shirobheda_videos\alolitam8.mp4"

MAX_FRAMES = 60

# -------------------------------------------------
# FaceMesh 68 subset indices (same as before)
# -------------------------------------------------
FACE68_IDX = list(range(0,17)) + list(range(70,96)) + list(range(97,137)) \
             + list(range(145,175)) + list(range(374,404)) + list(range(78,88)) + list(range(308,318))
FACE68_IDX = sorted(set(FACE68_IDX))[:68]

# -------------------------------------------------
# Load model + classes
# -------------------------------------------------
print("📦 Loading model + classes...")
model = load_model(MODEL_PATH)

classes = np.load(CLASSES_PATH, allow_pickle=True)
print("✅ Loaded classes:", classes)
print("✅ Model loaded successfully!\n")

# -------------------------------------------------
# Helper functions (MATCH split.py)
# -------------------------------------------------
def normalize_keypoints(seq):
    """
    seq: (T, 204) -> (T, 68, 3)
    framewise centering + scaling (same as split.py)
    """
    seq = np.array(seq, dtype=np.float32)

    if len(seq) == 0:
        return seq

    T, F = seq.shape
    seq = seq.reshape(T, 68, 3)

    # center x,y by mean per frame
    center_xy = np.mean(seq[:, :, :2], axis=1, keepdims=True)
    seq[:, :, :2] = seq[:, :, :2] - center_xy

    # scale x,y by std
    scale_xy = np.std(seq[:, :, :2], axis=(1, 2), keepdims=True) + 1e-6
    seq[:, :, :2] = seq[:, :, :2] / scale_xy

    # z scale
    z = seq[:, :, 2]
    z_scale = np.std(z, axis=1, keepdims=True) + 1e-6
    seq[:, :, 2] = z / z_scale

    return seq.reshape(T, 68 * 3)


def resample_sequence(seq, target_frames=MAX_FRAMES):
    """
    Resample to fixed 60 frames (same as split.py)
    """
    seq = np.array(seq, dtype=np.float32)

    if len(seq) == 0:
        return np.zeros((target_frames, 68 * 3), dtype=np.float32)

    if len(seq) == 1:
        return np.repeat(seq, target_frames, axis=0)

    old_idx = np.linspace(0, 1, len(seq))
    new_idx = np.linspace(0, 1, target_frames)

    f = interp1d(old_idx, seq, axis=0, kind="linear", fill_value="extrapolate")
    return f(new_idx).astype(np.float32)

# -------------------------------------------------
# Extract 68 keypoints from video
# -------------------------------------------------
def extract_face68_keypoints(video_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            pts = []
            for i in FACE68_IDX:
                p = lm[i]
                pts.extend([p.x, p.y, p.z])
        else:
            pts = [0.0] * (68 * 3)

        all_frames.append(pts)

    cap.release()
    face_mesh.close()

    if len(all_frames) == 0:
        raise ValueError("❌ No frames detected in video!")

    arr = np.array(all_frames, dtype=np.float32)      # (T, 204)
    arr = normalize_keypoints(arr)                    # ✅ correct normalization
    arr = resample_sequence(arr, MAX_FRAMES)          # ✅ fixed frames

    return np.expand_dims(arr, axis=0)                # (1, 60, 204)

# -------------------------------------------------
# Predict
# -------------------------------------------------
def predict_shirobheda(video_path):
    print(f"🎥 Processing video: {video_path}")

    X = extract_face68_keypoints(video_path)
    pred = model.predict(X, verbose=0)

    idx = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100.0
    predicted_class = str(classes[idx])

    print("\n🎯 Predicted Shirobheda Class:", predicted_class)
    print(f"📊 Confidence: {confidence:.2f}%")

    return predicted_class, confidence

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    predict_shirobheda(VIDEO_PATH)

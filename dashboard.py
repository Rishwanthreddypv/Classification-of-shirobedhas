# dashboard_predict.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

# UI / video
import cv2
import matplotlib.pyplot as plt

# ML
import tensorflow as tf
from tensorflow.keras.models import load_model

# Try to import your existing extraction utilities if available
EXISTING_EXTRACT_MODULE = None
try:
    # if you have extract.py exposing a function like extract_face68_from_video
    import extract as user_extract
    if hasattr(user_extract, "extract_face68_from_video"):
        EXISTING_EXTRACT_MODULE = user_extract
except Exception:
    EXISTING_EXTRACT_MODULE = None

# Fallback: mediapipe extraction
USE_MEDIAPIPE = True
if EXISTING_EXTRACT_MODULE:
    USE_MEDIAPIPE = False

# mediapipe fallback
def extract_face68_mediapipe(video_path, target_frames=60):
    """
    Extract face 68-ish keypoints from a video using MediaPipe FaceMesh.
    Returns: numpy array shape (target_frames, 204)  -> 68 pts * (x,y) -> 136 but we match 204 if you used more features.
    We'll output 204 by including (x,y,z) for 68 -> 68*3 = 204
    """
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError("mediapipe is required for fallback extraction. Install it or provide your extract.py") from e

    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=False,
                          min_detection_confidence=0.3,
                          min_tracking_confidence=0.3) as face_mesh:
        success, frame = cap.read()
        while success:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # keep first 68 if more (MediaPipe returns 468). We'll sample / pick typical face indices if 468.
                # We'll pick indices for 68 approximation; if fewer, we zero-pad.
                # Simple approach: take the first 68 landmarks
                coords = []
                for i in range(min(68, len(lm))):
                    coords.extend([lm[i].x, lm[i].y, lm[i].z])
                # if less than 68 fill zeros
                if len(coords) < 68*3:
                    coords += [0.0] * (68*3 - len(coords))
                frames.append(coords)
            else:
                frames.append([0.0] * (68*3))
            success, frame = cap.read()
    cap.release()

    # convert to numpy
    arr = np.array(frames)  # (num_frames, 204)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 68*3)
    # Resample/interpolate to target_frames
    n = arr.shape[0]
    if n == 0:
        raise RuntimeError("No frames read from video or no face landmarks detected.")
    if n == target_frames:
        return arr.astype(np.float32)
    # if n < 2, repeat
    if n == 1:
        arr = np.repeat(arr, target_frames, axis=0)
        return arr.astype(np.float32)
    # interpolate along frame axis for each feature
    x = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_frames)
    arr_new = np.zeros((target_frames, arr.shape[1]), dtype=np.float32)
    from scipy.interpolate import interp1d
    f = interp1d(x, arr, axis=0, kind='linear', fill_value="extrapolate")
    arr_new = f(x_new).astype(np.float32)
    return arr_new

# Preprocessing to match training pipeline
def preprocess_keypoints_to_model_input(kp_arr, expected_frames=60):
    """
    kp_arr: (num_frames, 204) -> output shape (1, expected_frames, 204)
    Zero-pad or crop if needed. Also normalization if used in training (we assume raw coordinates).
    """
    arr = kp_arr
    if arr.shape[0] < expected_frames:
        pad = np.zeros((expected_frames - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    elif arr.shape[0] > expected_frames:
        arr = arr[:expected_frames]
    # expand batch
    return np.expand_dims(arr, axis=0).astype(np.float32)

# Load model (path default to your saved model if exists)
MODEL_PATH_DEFAULT = Path("models/face68_lstm_model.keras")

@st.cache_resource
def load_trained_model(path=MODEL_PATH_DEFAULT):
    if not Path(path).exists():
        st.warning(f"Model not found at {path}. Please check path or train & save model.")
        return None
    # load with custom_objects if needed
    model = load_model(str(path))
    return model

# Utility: if dataset present, evaluate model on it
def evaluate_if_dataset(model, dataset_folder="face68_dataset"):
    """
    Expects dataset folder with train/test npy arrays saved by your split.py
    We'll try to load test arrays: X_test.npy and y_test.npy or infer from structure.
    """
    ds = Path(dataset_folder)
    if not ds.exists():
        return None
    # try standard-naming
    try:
        X_test = np.load(str(ds / "X_test.npy"))
        y_test = np.load(str(ds / "y_test.npy"), allow_pickle=True)
        # if y_test is encoded, adapt
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        return {"loss": float(loss), "accuracy": float(acc), "samples": X_test.shape[0]}
    except Exception:
        # try to build from folder layout (test/*/*.npy)
        tests = list((ds / "test").rglob("*.npy"))
        if not tests:
            return None
        # This is dataset-specific; skip robust implementation for now
        return None

# Prediction wrapper
def predict_video_class(model, video_file_path, classes, frames_expected=60):
    # first extract keypoints
    if EXISTING_EXTRACT_MODULE:
        try:
            kp = EXISTING_EXTRACT_MODULE.extract_face68_from_video(video_file_path, target_frames=frames_expected)
        except Exception as e:
            st.error(f"Error using your extract module: {e}\nFalling back to mediapipe.")
            kp = extract_face68_mediapipe(video_file_path, target_frames=frames_expected)
    else:
        kp = extract_face68_mediapipe(video_file_path, target_frames=frames_expected)

    inp = preprocess_keypoints_to_model_input(kp, expected_frames=frames_expected)
    preds = model.predict(inp)[0]  # shape (n_classes,)
    top_idx = int(np.argmax(preds))
    return {"preds": preds.tolist(), "pred_class": classes[top_idx], "pred_idx": top_idx, "confidence": float(preds[top_idx]), "kp_sequence": kp}

# Load class names from training logs or define
DEFAULT_CLASSES = ['Dhutam', 'adhomukham', 'alolitam', 'kampitam',
                   'paravrittam', 'parivahitam', 'sama', 'udvahitam', 'ukshipatam']

# Streamlit UI
st.set_page_config(page_title="Shirobheda Project Dashboard", layout="wide")
st.title("Shirobheda — Model Prediction Dashboard")

# Sidebar: model load and evaluation
st.sidebar.header("Model & Dataset")
model_path = st.sidebar.text_input("Model path", str(MODEL_PATH_DEFAULT))
model = load_trained_model(model_path)

if model is None:
    st.sidebar.warning("No model loaded. Predictions disabled until a model path is provided.")
else:
    st.sidebar.success("Model loaded.")

# Evaluate model if dataset present
if model is not None:
    eval_res = evaluate_if_dataset(model, dataset_folder="face68_dataset")
    if eval_res:
        st.sidebar.markdown("### Evaluation on dataset")
        st.sidebar.metric("Test accuracy", f"{eval_res['accuracy']*100:.2f}%")
        st.sidebar.write(f"Test samples: {eval_res['samples']}")
    else:
        st.sidebar.info("No ready-to-evaluate test dataset found at `face68_dataset` (or files not standard named).")

# Main: model info & metrics display
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Model Info")
    st.write(f"Model path: `{model_path}`")
    if model is not None:
        st.write(model.summary())
    else:
        st.write("No model loaded.")

with col2:
    st.subheader("Project stats")
    # show training snapshot from your logs if available
    st.write("Train samples: 72  ·  Test samples: 18")
    st.write("Saved model: models/face68_lstm_model.keras")

st.markdown("---")
st.header("Upload video & Predict")

uploaded = st.file_uploader("Upload video (mp4 / mov / avi)", type=["mp4","mov","avi","mkv"])
if uploaded is not None:
    # save to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    video_path = tfile.name

    st.video(video_path)

    if model is None:
        st.error("Model not loaded. Provide a valid model path in the sidebar.")
    else:
        with st.spinner("Extracting keypoints & predicting..."):
            try:
                result = predict_video_class(model, video_path, DEFAULT_CLASSES, frames_expected=60)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                result = None

        if result:
            st.success(f"Predicted: **{result['pred_class']}**  (confidence {result['confidence']*100:.2f}%)")
            # show bar chart of probabilities
            probs = np.array(result['preds'])
            dfp = pd.DataFrame({"class": DEFAULT_CLASSES, "prob": probs})
            st.subheader("Predicted probabilities")
            st.bar_chart(dfp.set_index("class"))

            # Show top-3
            top3_idx = np.argsort(probs)[-3:][::-1]
            st.write("Top-3 predictions:")
            for i in top3_idx:
                st.write(f"- {DEFAULT_CLASSES[i]}: {probs[i]*100:.2f}%")

            # Show keypoint animation (simple)
        st.subheader("Face keypoints (sampled frames)")
        kp_seq = result["kp_sequence"]  # (frames, 204)

        # Show up to 8 sampled frames
        nshow = min(8, kp_seq.shape[0])
        idxs = np.linspace(0, kp_seq.shape[0] - 1, nshow, dtype=int)
        cols = st.columns(4)

        for i, idx in enumerate(idxs):
            coords = kp_seq[idx].reshape(68, 3)
            x = coords[:, 0]
            y = coords[:, 1]

            fig, ax = plt.subplots(figsize=(2.4, 2.4))
            ax.scatter(x, -y, s=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"frame {idx+1}", fontsize=8)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-1.1, 0.1)
            ax.set_aspect('equal')

            cols[i % 4].pyplot(fig)
            plt.close(fig)


            # Save to predictions.csv
            out_row = {
                "timestamp": datetime.now().isoformat(),
                "filename": uploaded.name,
                "pred_class": result["pred_class"],
                "confidence": float(result["confidence"]),
                "probs": json.dumps(result["preds"])
            }
            csv_path = Path("predictions.csv")
            if not csv_path.exists():
                df_init = pd.DataFrame([out_row])
                df_init.to_csv(csv_path, index=False)
            else:
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([out_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
            st.success(f"Saved prediction to `{csv_path}`")

st.markdown("---")
st.header("Manage saved predictions")
pred_csv = Path("predictions.csv")
if pred_csv.exists():
    dfp = pd.read_csv(pred_csv)
    st.dataframe(dfp)
    st.download_button("Download predictions CSV", data=pred_csv.read_bytes(), file_name="predictions.csv")
else:
    st.info("No predictions.csv found yet. Make a prediction to create it.")

st.markdown("---")
st.write("Notes:")
st.write("""
- The extraction expects 68 face landmarks × (x,y,z)→ 204 features per frame (same as training).
- If your training used a different ordering/normalization, replace the `extract_face68_mediapipe` or connect your `extract.extract_face68_from_video` function to ensure consistent preprocessing.
- For best results, place your saved model at `models/face68_lstm_model.keras` or change the model path in sidebar.
""")

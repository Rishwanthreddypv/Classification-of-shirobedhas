import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# -------------------------------
# Paths
# -------------------------------
input_folder = r"G:\My Drive\shirobheda_videos"      # videos
output_folder = r"C:\Pinacle\face68_keypoints"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 68-point subset indices
# -------------------------------
FACE68_IDX = list(range(0,17)) + list(range(70,96)) + list(range(97,137)) \
             + list(range(145,175)) + list(range(374,404)) + list(range(78,88)) + list(range(308,318))
FACE68_IDX = sorted(set(FACE68_IDX))[:68]

# -------------------------------
# Initialize FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------
# Extraction loop
# -------------------------------
videos = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
print(f"Found {len(videos)} videos. Extracting 68-face keypoints...\n")

for v in tqdm(videos):
    video_path = os.path.join(input_folder, v)
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            pts = []
            for i in FACE68_IDX:
                p = lm[i]
                pts.extend([p.x, p.y, p.z])
        else:
            pts = [0.0] * (68 * 3)

        frames.append(pts)

    cap.release()

    # ✅ FIX: save clean filename always (.npy)
    base = os.path.splitext(v)[0]
    out_path = os.path.join(output_folder, base + ".npy")
    np.save(out_path, np.array(frames, dtype=np.float32))

face_mesh.close()
print("\n✅ Saved 68-face keypoints in:", output_folder)

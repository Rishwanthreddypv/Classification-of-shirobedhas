import os
import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# ✅ SETTINGS (EDIT PATHS ONLY)
# ============================================================

# Folder in Google Drive (via Google Drive for Desktop)
INPUT_FOLDER = r"G:\My Drive\shirobheda_videos"   # ✅ change this

# Output folder in your PC
OUTPUT_FOLDER = r"C:\Pinacle\Argumented_Dataset"  # ✅ change this

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ✅ Augmentation parameters you asked
BLUR_KSIZE = 13      # must be odd
ROT_ANGLE = 3       # degrees
BRIGHT_FACTOR = 1.25


# ============================================================
# ✅ AUGMENTATION FUNCTIONS
# ============================================================

def apply_blur(frame, ksize=9):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

def apply_rotation(frame, angle=3):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_brightness(frame, factor=1.25):
    frame = frame.astype(np.float32)
    frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
    return frame


# ============================================================
# ✅ PROCESS SINGLE VIDEO
# ============================================================

def augment_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fallback FPS (some videos give fps = 0)
    if fps is None or fps == 0:
        fps = 25

    # Output always mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not out.isOpened():
        print(f"❌ VideoWriter failed: {output_path}")
        cap.release()
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ Apply all 3 augmentations
        frame = apply_blur(frame, BLUR_KSIZE)
        frame = apply_rotation(frame, ROT_ANGLE)
        frame = apply_brightness(frame, BRIGHT_FACTOR)

        out.write(frame)

    cap.release()
    out.release()
    return True


# ============================================================
# ✅ MAIN LOOP
# ============================================================

valid_ext = (".mp4", ".mov")

videos = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_ext)]
print(f"✅ Total videos found: {len(videos)}")

failed = []

for vid in tqdm(videos):
    input_path = os.path.join(INPUT_FOLDER, vid)
    base = os.path.splitext(vid)[0]

    # output file naming
    output_path = os.path.join(OUTPUT_FOLDER, f"{base}_aug.mp4")

    ok = augment_video(input_path, output_path)
    if not ok:
        failed.append(vid)

print("\n🎉 DONE! Augmented dataset saved in:", OUTPUT_FOLDER)

if failed:
    print("\n❌ These videos failed (mostly codec issue):")
    for f in failed:
        print(" -", f)

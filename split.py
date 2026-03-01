import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
input_folder = r"C:\Pinacle\face68_keypoints"
output_folder = r"C:\Pinacle\face68_dataset"

train_folder = os.path.join(output_folder, "train")
test_folder = os.path.join(output_folder, "test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

MAX_FRAMES = 60

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def normalize_keypoints(seq):
    """
    Proper normalization for FaceMesh sequences.
    seq: (T, 204)  -> 68 keypoints * (x,y,z)

    Applies:
    - framewise centering (translation invariance)
    - framewise scaling using std (scale invariance)
    """
    seq = np.array(seq, dtype=np.float32)

    # if empty
    if len(seq) == 0:
        return seq

    T, F = seq.shape
    seq = seq.reshape(T, 68, 3)

    # center x,y by mean per frame
    center_xy = np.mean(seq[:, :, :2], axis=1, keepdims=True)  # (T,1,2)
    seq[:, :, :2] = seq[:, :, :2] - center_xy

    # scale x,y
    scale_xy = np.std(seq[:, :, :2], axis=(1, 2), keepdims=True) + 1e-6  # (T,1,1)
    seq[:, :, :2] = seq[:, :, :2] / scale_xy

    # z scale
    z = seq[:, :, 2]
    z_scale = np.std(z, axis=1, keepdims=True) + 1e-6
    seq[:, :, 2] = z / z_scale

    return seq.reshape(T, 68 * 3)


def resample_sequence(seq, target_frames=60):
    """
    Resample a sequence of frames to fixed frames using interpolation.
    """
    seq = np.array(seq, dtype=np.float32)

    if len(seq) == 0:
        return np.zeros((target_frames, 68 * 3), dtype=np.float32)

    if len(seq) == 1:
        return np.repeat(seq, target_frames, axis=0)

    old_idx = np.linspace(0, 1, len(seq))
    new_idx = np.linspace(0, 1, target_frames)

    f = interp1d(old_idx, seq, axis=0, kind="linear", fill_value="extrapolate")
    out = f(new_idx).astype(np.float32)
    return out


# ------------------------------------------------------------
# Build Dataset
# ------------------------------------------------------------
files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

def get_class_name(filename):
    name = filename.replace(".npy", "")
    for i, c in enumerate(name):
        if c.isdigit():
            return name[:i].lower()
    return name.lower()

class_dict = {}
for f in files:
    cls = get_class_name(f)
    class_dict.setdefault(cls, []).append(f)

print(f"📂 Found {len(class_dict)} classes:")
for c in class_dict:
    print("   -", c, ":", len(class_dict[c]))

# ------------------------------------------------------------
# Normalize + Resample + Split
# ------------------------------------------------------------
for cls, cls_files in class_dict.items():
    train_files, test_files = train_test_split(cls_files, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_folder, cls), exist_ok=True)
    os.makedirs(os.path.join(test_folder, cls), exist_ok=True)

    for f in tqdm(train_files, desc=f"Processing train/{cls}"):
        arr = np.load(os.path.join(input_folder, f)).astype(np.float32)
        arr = normalize_keypoints(arr)
        arr = resample_sequence(arr, target_frames=MAX_FRAMES)
        np.save(os.path.join(train_folder, cls, f), arr)

    for f in tqdm(test_files, desc=f"Processing test/{cls}"):
        arr = np.load(os.path.join(input_folder, f)).astype(np.float32)
        arr = normalize_keypoints(arr)
        arr = resample_sequence(arr, target_frames=MAX_FRAMES)
        np.save(os.path.join(test_folder, cls, f), arr)

print("\n✅ Dataset built successfully!")
print("Train/Test saved in:", output_folder)

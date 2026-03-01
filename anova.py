import numpy as np
import os
from pathlib import Path
from scipy.stats import f_oneway

DATASET = Path("face68_dataset/train")

# choose keypoint index
KP = 30  # nose tip (example)
AXIS = 0 # 0 = X, 1 = Y, 2 = Z

def load_all_samples(folder):
    samples = []
    for f in Path(folder).glob("*.npy"):
        arr = np.load(str(f))
        kp = arr[:, KP*3 + AXIS]  # extract 1D trajectory
        samples.append(kp)
    return samples

classes = sorted(os.listdir(DATASET))

# build list of samples per class for ANOVA
samples_by_class = []

for cls in classes:
    class_data = load_all_samples(DATASET / cls)
    # flatten all sequences into one long vector for this class
    merged = np.concatenate(class_data)
    samples_by_class.append(merged)

# run ANOVA
F, p = f_oneway(*samples_by_class)

print("\n====== ANOVA RESULTS ======")
print(f"Keypoint tested: {KP} (axis {AXIS})")
print(f"F-statistic: {F:.4f}")
print(f"p-value: {p:.6f}")

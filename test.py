# in Python REPL or a small script
import numpy as np
from utils.feature_extractor import extract_features_from_keypoints_sequence
kps = np.load("kps/dhutam_good1.npy")   # shape (T,N,2)
feats = extract_features_from_keypoints_sequence([f for f in kps])
print("features shape:", feats.shape)    # (T,F)
np.save("templates/Dhutam/ex1.npy", feats)

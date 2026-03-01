import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Classes used in your training (keep exact order)
classes = [
    'Dhutam',
    'adhomukham',
    'alolitam',
    'kampitam',
    'paravrittam',
    'parivahitam',
    'sama',
    'udvahitam',
    'ukshipatam'
]

# Recreate label encoder
le = LabelEncoder()
le.fit(classes)

# Make sure folder exists
os.makedirs(r"C:\Project\models", exist_ok=True)

# Save it
with open(r"C:\Project\models\label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Label encoder saved successfully!")

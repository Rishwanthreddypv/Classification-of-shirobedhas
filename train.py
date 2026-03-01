import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Paths
# -----------------------------
DATA_ROOT = r"C:\Pinacle\face68_dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

MODEL_DIR = r"C:\Pinacle\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Hyperparameters
# -----------------------------
MAX_FRAMES = 60
BATCH = 4
EPOCHS = 200
LR = 1e-4

# -----------------------------
# Load dataset
# -----------------------------
def load_split(folder):
    X, y = [], []

    for cls in sorted(os.listdir(folder)):
        path = os.path.join(folder, cls)
        if not os.path.isdir(path):
            continue

        for f in os.listdir(path):
            if f.endswith(".npy"):
                arr = np.load(os.path.join(path, f)).astype(np.float32)

                # ✅ enforce fixed frames
                if arr.shape[0] != MAX_FRAMES:
                    continue

                X.append(arr)
                y.append(cls)

    return np.array(X, dtype=np.float32), np.array(y)


print("Loading data...")
X_train, y_train = load_split(TRAIN_DIR)
X_test,  y_test  = load_split(TEST_DIR)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

y_train_cat = to_categorical(y_train_enc)
y_test_cat  = to_categorical(y_test_enc)

n_classes = len(le.classes_)
print("Classes:", le.classes_)

# ✅ Save class order for inference later
np.save(os.path.join(MODEL_DIR, "classes.npy"), le.classes_)

# -----------------------------
# Build model
# -----------------------------
input_dim = X_train.shape[2]

model = Sequential([
    Masking(mask_value=0., input_shape=(MAX_FRAMES, input_dim)),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.30),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation='relu'),
    Dropout(0.25),

    Dense(n_classes, activation='softmax')
])

opt = Adam(learning_rate=LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train (NO EarlyStopping)
# -----------------------------
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=1
)

# -----------------------------
# ✅ Final Accuracies (Train / Val / Test)
# -----------------------------
final_train_acc = history.history["accuracy"][-1]
final_val_acc   = history.history["val_accuracy"][-1]

test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

print("\n================ FINAL RESULTS ================")
print(f"✅ Final Training Accuracy   : {final_train_acc:.4f}")
print(f"✅ Final Validation Accuracy : {final_val_acc:.4f}")
print(f"✅ Final Testing Accuracy    : {test_acc:.4f}")
print("================================================\n")

# -----------------------------
# Evaluate Predictions
# -----------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)

print("Classification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# -----------------------------
# ✅ Confusion Matrix HEATMAP
# -----------------------------
cm = confusion_matrix(y_test_enc, y_pred)

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot Accuracy/Loss
# -----------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(['train', 'val'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(['train', 'val'])

plt.tight_layout()
plt.show()

# -----------------------------
# Save model
# -----------------------------
save_path = os.path.join(MODEL_DIR, "face68_lstm_model.keras")
model.save(save_path)

print(f"\n✅ Saved model at: {save_path}")
print(f"✅ Saved classes at: {os.path.join(MODEL_DIR, 'classes.npy')}")

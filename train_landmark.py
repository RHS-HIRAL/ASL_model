import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

# --- Config ---
EPOCHS = 25
BATCH_SIZE = 32

# --- Load Data ---
# Assumes landmarks.npy (shape: [num_samples, num_features]) and labels.npy (shape: [num_samples,])
landmarks = np.load('landmarks.npy')
labels = np.load('labels.npy')

num_classes = len(np.unique(labels))
input_dim = landmarks.shape[1]

# --- Train/Val Split ---
X_train, X_val, y_train, y_val = train_test_split(
    landmarks, labels, test_size=0.2, random_state=42, stratify=labels
)

# --- Model Architecture ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.summary()

# --- Compile Model ---
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Training ---
csv_logger = CSVLogger('landmark_training_log.csv', append=False)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[csv_logger]
)

# --- Save Model ---
model.save('sign_landmark_model.h5')
print('Landmark-based model training complete and saved as "sign_landmark_model.h5"')

# --- Plot Training Curves ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

# Save the plots
plt.xlabel('Epochs')
plt.tight_layout()
plt.savefig('landmark_training_curves.png')
plt.show() 
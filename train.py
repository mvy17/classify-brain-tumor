import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --------------------------
# Paths
# --------------------------
PROJECT_ROOT = r"C:\Users\yashwanth\Music\brain tumor"
DATASET_DIR = r"C:\Users\yashwanth\Music\brain tumor\Testing"

# --------------------------
# Image generators with validation split
# --------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# --------------------------
# Automatically detect number of classes
# --------------------------
num_classes = train_gen.num_classes
print("Detected classes:", train_gen.class_indices)
print("Number of classes:", num_classes)

# --------------------------
# Model definition
# --------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # output layer matches dataset
])

# --------------------------
# Compile model
# --------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------
# Train model
# --------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    verbose=1
)

# --------------------------
# Save model
# --------------------------
model.save(os.path.join(PROJECT_ROOT, "brain_tumor_model1.h5"))
print("Model saved!")

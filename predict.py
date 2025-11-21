import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import pandas as pd

# Load model
model = tf.keras.models.load_model("brain_tumor_model1.h5")

# Class names
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Folders to predict
base_path = r"C:\Users\yashwanth\Music\brain tumor\Testing"

folders = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

results = []

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    index = np.argmax(pred)
    confidence = np.max(pred)
    return class_names[index], confidence


for folder in folders:
    folder_path = os.path.join(base_path, folder)

    print(f"\n📁 Processing folder: {folder_path}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            
            img_path = os.path.join(folder_path, filename)

            predicted_class, confidence = predict_image(img_path)

            print(f"{filename} → {predicted_class} ({confidence*100:.2f}%)")

            results.append({
                "image": filename,
                "actual_folder": folder,
                "predicted_class": predicted_class,
                "confidence": confidence
            })


# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("prediction_results.csv", index=False)

print("\n✅ Prediction complete!")
print("📄 Results saved to: prediction_results.csv")

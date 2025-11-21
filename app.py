import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import werkzeug
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --------------------------
# Configuration
# --------------------------
PROJECT_ROOT = r"C:\Users\yashwanth\Music\brain tumor"
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, "templates")
STATIC_FOLDER = os.path.join(PROJECT_ROOT, "static")
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = "supersecretkey123"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# --------------------------
# Load Model
# --------------------------
MODEL_PATH = os.path.join(PROJECT_ROOT, "brain_tumor_model1.h5")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


# --------------------------
# Prediction Function
# --------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    predicted_idx = np.argmax(preds)
    predicted_class = class_names[predicted_idx]
    confidence = preds[predicted_idx] * 100

    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

    return predicted_class, confidence


# --------------------------
# Login
# --------------------------
USERNAME = "admin"
PASSWORD = "123456"

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("username") == USERNAME and request.form.get("password") == PASSWORD:
            session["user"] = USERNAME
            return redirect(url_for("home"))
        return render_template("login.html", error="Invalid username or password.")
    
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# --------------------------
# Pages
# --------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/contact")
def contact():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("contact.html")


# --------------------------
# Predict Endpoint
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"})

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print(f"Saved image at: {save_path}")

    # Prediction
    predicted_class, confidence = predict_image(save_path)

    return jsonify({
        "result": f"{predicted_class} ({confidence:.2f}%)",
        "image_url": f"/static/uploads/{filename}"
    })


# --------------------------
# Run Server
# --------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

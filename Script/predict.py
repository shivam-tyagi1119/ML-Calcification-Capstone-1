"""
===============================================================================
PREDICTION SERVICE
Vehicle Damage Classification – Insurance Claims
===============================================================================
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
from flask import Flask, request, jsonify

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = "../Notebook/best_model_lr_0.01_dropout_0.3.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["no_damage", "minor_damage", "major_damage"]

# -------------------------------------------------
# Load Model (once at startup)
# -------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

app = Flask(__name__)

# -------------------------------------------------
# Image Preprocessing (MATCHES TRAINING)
# -------------------------------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image = np.array(image)

    image = tf.keras.applications.resnet50.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image_tensor = preprocess_image(file.read())

    predictions = model.predict(image_tensor)
    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    result = {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": round(confidence, 4),
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(predictions[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }

    return jsonify(result), 200

# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP"}), 200

# -------------------------------------------------
# Local Execution Only
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=False)

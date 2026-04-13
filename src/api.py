from flask import Flask, request, jsonify, render_template
import os
try:
    import tensorflow as tf
except ImportError:
    tf = None
import numpy as np
import cv2

app = Flask(__name__, template_folder='../templates')

# Preload the model once the server starts
MODEL_PATH = "models/medical_ai_model.h5"
if tf is not None and os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Successfully loaded AI Model for API!")
else:
    model = None
    if tf is None:
        print("Warning: TensorFlow is not installed. Running in mock UI mode for demonstration.")
    else:
        print("Warning: Model not found. The predict route will fail. Train the model first.")

@app.route('/', methods=['GET'])
def index():
    """Renders the HTML web interface for easy file uploads."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint that accepts a medical image and returns a diagnosis prediction.
    """
    if model is None:
        return jsonify({"Error": "Model not trained yet."}), 500

    if 'image' not in request.files:
        return jsonify({"Error": "No 'image' file provided."}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"Error": "Empty file name."}), 400

    try:
        # Read the file to a buffer, decode to an OpenCV image as grayscale
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if image is None:
             return jsonify({"Error": "Failed to decode image file."}), 400

        # Preprocess the image to match exactly what the model was trained on
        image = cv2.resize(image, (256, 256))
        
        # Normalization
        image = image / 255.0  

        if model is None:
            # Provide mock prediction for UI screenshot purposes if model/tf is absent
            import random
            prediction_prob = random.uniform(0.1, 0.9)
        else:
            # Reshape for the CNN (Batch Size, Height, Width, Channels) => (1, 256, 256, 1)
            image = image.reshape(1, 256, 256, 1)
            # Run Prediction
            prediction_prob = model.predict(image)[0][0]
        
        # 0.5 threshold for Binary Classification
        prediction_label = "Pneumonia Detected" if prediction_prob > 0.5 else "Normal (No Pneumonia)"

        return jsonify({
            "Prediction": prediction_label,
            "Confidence": float(prediction_prob) if prediction_prob > 0.5 else float(1.0 - prediction_prob)
        })

    except Exception as e:
        return jsonify({"Error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask API Server...")
    app.run(debug=True, port=5000)

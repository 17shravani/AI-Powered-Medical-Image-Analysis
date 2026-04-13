import os
import tensorflow as tf
from src.preprocess import load_and_preprocess_data
from src.model import create_cnn_model

def train_model(data_dir="data", model_save_path="models/medical_ai_model.h5", epochs=5):
    """
    Trains the CNN model on the dataset and saves the weights.
    """
    if not os.path.exists("models"):
        os.makedirs("models")

    # Load preprocessed data
    print("Initializing Data Generators...")
    train_data, test_data = load_and_preprocess_data(data_dir)

    if train_data.samples == 0:
        print("Error: No training data found. Please run data_generation.py first.")
        return None

    # Load model architecture
    print("Building Model Architecture...")
    model = create_cnn_model()

    # Compile the model
    # We use Adam optimizer, Binary Crossentropy loss (since it's a binary classification 0 or 1),
    # and we track accuracy.
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print(f"Starting Training for {epochs} epochs...")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"\nModel training complete! Saved to {model_save_path}")

    return history

if __name__ == "__main__":
    train_model()

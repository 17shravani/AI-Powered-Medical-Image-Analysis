import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from src.preprocess import load_and_preprocess_data

def evaluate_model(data_dir="data", model_path="models/medical_ai_model.h5"):
    """
    Evaluates the loaded model against the test dataset and highlights the accuracy and confusion matrix.
    Saves plots to the artifacts directory.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    print("Loading Preprocessed Test Data...")
    _, test_data = load_and_preprocess_data(data_dir)

    if test_data.samples == 0:
        print("Error: No test data found. Please run data_generation.py first.")
        return

    print("Loading Model...")
    model = tf.keras.models.load_model(model_path)

    print("Evaluating Model on Test Data...")
    y_true = test_data.classes
    # Get predictions
    y_pred_probs = model.predict(test_data)
    y_pred = np.round(y_pred_probs).astype(int).flatten()

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- MODEL EVALUATION RESULTS ---")
    print(f"Accuracy Score: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    # Save Confusion Matrix Plot
    save_confusion_matrix(cm, ["Normal", "Pneumonia"])

def save_confusion_matrix(cm, classes, output_path="outputs/confusion_matrix.png"):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    evaluate_model()

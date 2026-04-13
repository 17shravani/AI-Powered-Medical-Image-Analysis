from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape=(256, 256, 1)):
    """
    Builds a custom Convolutional Neural Network (CNN) 
    designed for binary classification (Normal vs Pneumonia).
    """
    model = Sequential([
        # First Convolutional Layer to extract basic features (e.g., edges)
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Layer to extract higher-level features
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 2D arrays for the fully connected layers
        Flatten(),
        
        # Fully Connected Layer representing combinations of features
        Dense(128, activation='relu'),
        
        # Dropout layer to prevent overfitting by randomly dropping neurons
        Dropout(0.5),
        
        # Output Layer: Single node with sigmoid activation for binary output (0 or 1)
        Dense(1, activation='sigmoid')
    ])

    return model

if __name__ == "__main__":
    model = create_cnn_model()
    model.summary()

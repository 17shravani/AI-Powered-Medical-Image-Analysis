import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir="data", img_size=(256, 256), batch_size=16):
    """
    Prepares images for training by resizing, normalizing,
    and applying data augmentation.
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    # Image Data Generator with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=15,       # Rotate images randomly by up to 15 degrees
        width_shift_range=0.1,   # Shift image horizontally
        height_shift_range=0.1,  # Shift image vertically
        zoom_range=0.1,          # Randomly zoom into the image
        horizontal_flip=True     # Flip images horizontally for variation
    )

    # Basic normalization for testing (no augmentation for testing!)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="binary"
    )

    print("Loading Testing Data...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )
    
    return train_generator, test_generator

if __name__ == "__main__":
    train_data, test_data = load_and_preprocess_data()

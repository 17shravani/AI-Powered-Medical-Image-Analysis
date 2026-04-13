import os
import cv2
import numpy as np

def generate_dummy_data(base_dir="data", num_samples_per_class=20):
    """
    Generates dummy grayscale images to simulate a medical dataset.
    This allows students to test the code immediately without downloading large datasets.
    """
    classes = ["NORMAL", "PNEUMONIA"]
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    for split_dir in [train_dir, test_dir]:
        for cls in classes:
            path = os.path.join(split_dir, cls)
            os.makedirs(path, exist_ok=True)
            
            for i in range(num_samples_per_class):
                # Generate a 256x256 image with random noise to simulate an X-ray texture
                img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
                
                # If pneumonia, add some "white cloudiness" (simulating infection)
                if cls == "PNEUMONIA":
                    overlay = np.ones((256, 256), dtype=np.uint8) * 150
                    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                
                # If normal, add more clear dark regions (simulating clear lungs)
                elif cls == "NORMAL":
                    img = cv2.GaussianBlur(img, (25, 25), 0)

                img_path = os.path.join(path, f"sample_{i}.jpeg")
                cv2.imwrite(img_path, img)
    
    print(f"Dummy dataset generated in '{base_dir}' folder.")

if __name__ == "__main__":
    generate_dummy_data()

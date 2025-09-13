import numpy as np
import cv2
import os

def generate_dummy_images(base_path, num_images_per_digit=5):
    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        os.makedirs(digit_path, exist_ok=True)
        for i in range(num_images_per_digit):
            # Create a blank 28x28 grayscale image
            img = np.zeros((28, 28), dtype=np.uint8)
            # Optionally, add a simple pattern for more realistic dummy data
            # For example, draw a white square in the middle
            # cv2.rectangle(img, (7, 7), (20, 20), 255, -1)
            
            img_name = f"dummy_digit_{digit}_{i}.png"
            cv2.imwrite(os.path.join(digit_path, img_name), img)
    print(f"Generated {num_images_per_digit} dummy images for each digit in {base_path}")

if __name__ == '__main__':
    generate_dummy_images("data/school_digits")



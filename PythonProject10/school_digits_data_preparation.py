import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_school_digits_data(base_path, test_size=0.2, random_state=42):
    images = []
    labels = []
    for digit in range(10):
        digit_path = os.path.join(base_path, str(digit))
        for img_name in os.listdir(digit_path):
            img_path = os.path.join(digit_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Convert to grayscale
            img = cv2.resize(img, (28, 28)) # Resize to 28x28
            img = img.astype("float32") / 255.0 # Normalize to [0, 1]
            img = np.expand_dims(img, -1) # Reshape for CNN input (28, 28, 1)
            images.append(img)
            labels.append(digit)

    X = np.array(images)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    print("School digits data preparation complete.")
    print(f"X_school_train shape: {X_train.shape}")
    print(f"y_school_train shape: {y_train.shape}")
    print(f"X_school_test shape: {X_test.shape}")
    print(f"y_school_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train_school, y_train_school, X_test_school, y_test_school = prepare_school_digits_data("/home/ubuntu/digit_classifier_project/data/school_digits")



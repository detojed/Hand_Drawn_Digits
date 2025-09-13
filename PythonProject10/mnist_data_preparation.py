
import tensorflow as tf
import numpy as np

def prepare_mnist_data():
    # 3. Download the MNIST dataset using keras.datasets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 4. Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 5. Reshape images for CNN input: (28, 28, 1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    print("MNIST data preparation complete.")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = prepare_mnist_data()
    # You can add code here to save the processed data if needed



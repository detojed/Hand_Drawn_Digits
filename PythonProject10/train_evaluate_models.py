import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

from mnist_data_preparation import prepare_mnist_data
from school_digits_data_preparation import prepare_school_digits_data
from cnn_model import build_cnn_model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n--- Training {model_name} ---")
    history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    
    # Save model
    model_path = os.path.join("models", f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate
    print(f"\n--- Evaluating {model_name} ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join("reports", f"confusion_matrix_{model_name}.png"))
    plt.close()
    print(f"Confusion matrix saved for {model_name}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("reports", f"training_history_{model_name}.png"))
    plt.close()
    print(f"Training history plots saved for {model_name}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

if __name__ == '__main__':
    # Prepare data
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = prepare_mnist_data()
    X_train_school, y_train_school, X_test_school, y_test_school = prepare_school_digits_data("/home/ubuntu/digit_classifier_project/data/school_digits")

    all_results = {}

    # Model 1: Only MNIST data
    model_mnist = build_cnn_model()
    results_mnist = train_and_evaluate(model_mnist, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, "mnist_only_model")
    all_results["mnist_only"] = results_mnist

    # Model 2: Only school digits data
    model_school = build_cnn_model()
    results_school = train_and_evaluate(model_school, X_train_school, y_train_school, X_test_school, y_test_school, "school_only_model")
    all_results["school_only"] = results_school

    # Model 3: Combined dataset
    X_train_combined = np.concatenate((X_train_mnist, X_train_school), axis=0)
    y_train_combined = np.concatenate((y_train_mnist, y_train_school), axis=0)
    X_test_combined = np.concatenate((X_test_mnist, X_test_school), axis=0)
    y_test_combined = np.concatenate((y_test_mnist, y_test_school), axis=0)

    model_combined = build_cnn_model()
    results_combined = train_and_evaluate(model_combined, X_train_combined, y_train_combined, X_test_combined, y_test_combined, "combined_model")
    all_results["combined"] = results_combined

    print("\n--- All Training and Evaluation Complete ---")
    print("Summary of Results:")
    for model_type, metrics in all_results.items():
        print(f"  {model_type.replace('_', ' ').title()} Model:")
        for metric, value in metrics.items():
            print(f"    {metric.replace('_', ' ').title()}: {value:.4f}")

    # Save summary of performance metrics to a JSON file
    import json
    with open(os.path.join("reports", "performance_summary.json"), "w") as f:
        json.dump(all_results, f, indent=4)
    print("Performance summary saved to reports/performance_summary.json")



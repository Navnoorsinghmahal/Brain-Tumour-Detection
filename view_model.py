import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import numpy as np


def load_and_view_model(model_path, history_path=None, X_test=None, y_test=None):
    # Load the model from the file
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Print the summary of the model
    print("\nModel Summary:")
    model.summary()

    # Optionally, view the training history if a history file is provided
    if history_path:
        try:
            with open(history_path, 'rb') as file:
                history = pickle.load(file)
            print("\nPlotting training history...")
            plot_metrics(history)
        except FileNotFoundError:
            print(f"History file not found: {history_path}")

    # Optionally, evaluate the model on test data if provided
    if X_test is not None and y_test is not None:
        print("\nEvaluating model on test data...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")


def plot_metrics(history):
    # Plot training and validation loss
    plt.figure()
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Define the paths to your model and history files
    model_path = 'models/cnn-parameters-improvement-10-0.84.keras'  # Update this path
    history_path = None  # Update this path if you have a history file
    X_test = None  # Update with your test data if you want to evaluate the model
    y_test = None  # Update with your test labels if you want to evaluate the model

    load_and_view_model(model_path, history_path, X_test, y_test)

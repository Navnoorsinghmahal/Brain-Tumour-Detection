import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, \
    Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Load the model from the file
model_path = 'models/cnn-parameters-improvement-10-0.84.keras'  # Update this to the correct filename
model = load_model(model_path)


# Data Preparation & Preprocessing
def crop_brain_contour(image, plot=False):
    gray = image.convert('L')  # Convert image to grayscale
    gray = gray.filter(ImageFilter.GaussianBlur(5))  # Apply Gaussian blur
    thresh = gray.point(lambda p: p > 45 and 255)  # Apply thresholding
    thresh = np.array(thresh)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        new_image = image.crop((extLeft[0], extTop[1], extRight[0], extBot[1]))
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.subplot(1, 2, 2)
            plt.imshow(new_image)
            plt.title('Cropped Image')
            plt.savefig('cropped_image_comparison.png')
            plt.close()
        return new_image
    return image  # Return original if no contours found


def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size
    for directory in dir_list:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                image = Image.open(file_path)
                image = crop_brain_contour(image, plot=False)
                image = image.resize((image_width, image_height))
                image = np.array(image) / 255.0
                X.append(image)
                y.append([1] if 'yes' in directory else [0])
    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    return X, y


def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((4, 4), name='max_pool0')(X)
    X = MaxPooling2D((4, 4), name='max_pool1')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='BrainDetectionModel')
    return model


IMG_WIDTH, IMG_HEIGHT = (240, 240)
augmented_path = 'augmented_data/'
augmented_yes = augmented_path + 'yes'
augmented_no = augmented_path + 'no'
X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))
X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)

model = build_model((IMG_WIDTH, IMG_HEIGHT, 3))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/brain_tumor_detection_cnn')
checkpoint = ModelCheckpoint("models/cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}.keras",
                             monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard, checkpoint]
)


# Plot and save metrics
def save_metrics_plots(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()


save_metrics_plots(history)

# Evaluate the model on the test set
best_model = load_model('models/cnn-parameters-improvement-09-0.87.keras')
loss, acc = best_model.evaluate(X_test, y_test)
print(f"Test Loss = {loss}")
print(f"Test Accuracy = {acc}")


# Make predictions and save misclassifications
def save_misclassifications(X, y_true, y_pred, num_samples=5):
    misclassified_idx = np.where(y_true != y_pred)[0]
    for i in range(min(num_samples, len(misclassified_idx))):
        idx = misclassified_idx[i]
        plt.figure()
        plt.imshow(X[idx])
        plt.title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        plt.savefig(f'misclassification_{i}.png')
        plt.close()


y_test_prob = best_model.predict(X_test)
y_test_pred = (y_test_prob > 0.5).astype(int)
save_misclassifications(X_test, y_test, y_test_pred)

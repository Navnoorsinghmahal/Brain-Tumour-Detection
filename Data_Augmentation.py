import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np


def augment_data(file_dir, n_generated_samples, save_to_dir):
    # Ensure save_to_dir exists
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    # Create ImageDataGenerator object with augmentation parameters
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Process each image file in the directory
    for filename in os.listdir(file_dir):
        try:
            image_path = os.path.join(file_dir, filename)
            image = Image.open(image_path)

            # Convert RGBA images to RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image = np.array(image)

            # Add channel dimension if missing
            if len(image.shape) == 2:  # Grayscale image
                image = np.expand_dims(image, axis=-1)
                image = np.repeat(image, 3, axis=-1)  # Convert grayscale to RGB
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
                image = np.array(image.convert('RGB'))  # Convert RGBA to RGB

            image = np.expand_dims(image, axis=0)

            save_prefix = 'aug_' + os.path.splitext(filename)[0]
            i = 0
            for batch in data_gen.flow(image, batch_size=1, save_to_dir=save_to_dir,
                                       save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i >= n_generated_samples:
                    break
        except Exception as e:
            print(f"Error processing {filename}: {e}")


def main():
    # Define paths
    dataset_dir = 'dataset'
    augmented_data_path = 'augmented_data/'

    print("Dataset directory:", os.path.abspath(dataset_dir))
    print("Augmented data path:", os.path.abspath(augmented_data_path))

    # Augment images
    augment_data(file_dir=os.path.join(dataset_dir, 'no'),
                 n_generated_samples=6, save_to_dir=os.path.join(augmented_data_path, 'no'))
    augment_data(file_dir=os.path.join(dataset_dir, 'yes'),
                 n_generated_samples=9, save_to_dir=os.path.join(augmented_data_path, 'yes'))


if __name__ == "__main__":
    main()

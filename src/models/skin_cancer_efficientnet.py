# -*- coding: utf-8 -*-
"""skin_cancer_efficientnet.ipynb

Automatically generated by Colaboratory.


"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
# Setting paths
TRAIN_IMAGE_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Training_Data"
TRAIN_MASK_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Training_GroundTruth"
TEST_IMAGE_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_Data"
TEST_MASK_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_GroundTruth"
# Get the list of training image and mask filenames
training_images = set(os.listdir(TRAIN_IMAGE_PATH))
training_masks = {mask_file.replace('_Segmentation.png', '.jpg') for mask_file in os.listdir(TRAIN_MASK_PATH)}

# Identify and print the unmatched training images
unmatched_training_images = training_images - training_masks
print("Unmatched training images:", unmatched_training_images)

# If you wish to remove the unmatched images, uncomment and run the following:
for image_file in unmatched_training_images:
     os.remove(os.path.join(TRAIN_IMAGE_PATH, image_file))

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Setting paths
TRAIN_IMAGE_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Training_Data"
TRAIN_MASK_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Training_GroundTruth"
TEST_IMAGE_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_Data"
TEST_MASK_PATH = "/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_GroundTruth"

def custom_data_generator(image_dir, mask_dir, batch_size, image_size=(128, 128), shuffle=True, augment=False):
    # Fetch all the image and mask paths
    image_files = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir)])
    mask_files = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)])

    total = len(image_files)
    indices = np.arange(total)

    while True:
        # Shuffle indices after each epoch
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_indices = indices[start:end]

            batch_images = []
            batch_masks = []
            for i in batch_indices:
                # Load images and masks, then preprocess them
                img = load_img(image_files[i], target_size=image_size)
                img = img_to_array(img) / 255.0

                mask = load_img(mask_files[i], target_size=image_size, color_mode="grayscale")
                mask = img_to_array(mask) / 255.0

                batch_images.append(img)
                batch_masks.append(mask)

            yield np.array(batch_images), np.array(batch_masks)

# Create generators
train_generator = custom_data_generator(TRAIN_IMAGE_PATH, TRAIN_MASK_PATH, 32)
val_generator = custom_data_generator(TEST_IMAGE_PATH, TEST_MASK_PATH, 32, shuffle=False)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

input_shape = (128, 128, 3)
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)

# Create U-Net-like structure with EfficientNetB0 as encoder
inputs = Input(shape=input_shape)
x = base_model(inputs, training=True)
x = UpSampling2D()(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D()(x)
outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)

model = Model(inputs, outputs)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

model.compile(optimizer=Adam(learning_rate=0.0001), loss=binary_crossentropy, metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1)
early_stop = EarlyStopping(patience=10, restore_best_weights=True, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=898 // 32,
    validation_data=val_generator,
    validation_steps=379 // 32,
    epochs=50,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

# 6. Save the model
model.save("skin_cancer_efficientnet_model.h5")

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)

print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained EfficientNet model
efficientnet_model = load_model('efficientnet_best_model.h5')

# Test dataset directory
test_images_dir = '/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_Data/'

# Define image dimensions for EfficientNet
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Load and preprocess the test images
test_image_files = os.listdir(test_images_dir)
X_test = np.zeros((len(test_image_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
for n, image_file in enumerate(test_image_files):
    image_path = os.path.join(test_images_dir, image_file)
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    X_test[n] = img_to_array(img)

# Predict using the EfficientNet model
efficientnet_predictions = efficientnet_model.predict(X_test)

# Save the predictions as images
save_dir = '/content/drive/MyDrive/EfficientNet/predicted_masks_efficientnet/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i, pred_mask in enumerate(efficientnet_predictions):
    # Squeeze out the last dimension to get a 2D array
    mask_2d = np.squeeze(pred_mask, axis=-1)

    # Convert the mask values to the range [0, 255] and then to uint8 type
    mask_to_save = (mask_2d * 255).astype(np.uint8)

    # Convert the numpy array to a PIL image
    mask_img = Image.fromarray(mask_to_save)

    # Extract the base name of the original test image
    base_name = os.path.splitext(test_image_files[i])[0]

    # Create the filename for the predicted mask
    mask_filename = os.path.join(save_dir, base_name + "_Segmentation.png")

    # Save the PIL image
    mask_img.save(mask_filename)

import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# Paths for the predicted masks and ground truth masks
predicted_masks_path = '/content/drive/MyDrive/EfficientNet/predicted_masks_efficientnet/'
ground_truth_path = '/content/drive/MyDrive/EfficientNet/ISBI2016_ISIC_Part1_Test_GroundTruth/'

def compute_iou(mask1, mask2):
    """Compute Intersection over Union (IoU) of two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def compute_dice(mask1, mask2):
    """Compute Dice Coefficient of two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    return 2. * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))

ious = []
dices = []

for mask_file in os.listdir(predicted_masks_path):
    predicted_mask = imread(os.path.join(predicted_masks_path, mask_file))
    ground_truth_mask_original = imread(os.path.join(ground_truth_path, mask_file))

    # Resize the ground truth mask to 128x128
    ground_truth_mask = resize(ground_truth_mask_original, (128, 128), mode='constant', preserve_range=True)

    # Binarize the masks for metric computation
    predicted_mask_binarized = (predicted_mask > 127).astype(np.uint8)
    ground_truth_mask_binarized = (ground_truth_mask > 127).astype(np.uint8)

    iou = compute_iou(predicted_mask_binarized, ground_truth_mask_binarized)
    dice = compute_dice(predicted_mask_binarized, ground_truth_mask_binarized)

    ious.append(iou)
    dices.append(dice)

# Compute average IoU and Dice Coefficient
avg_iou = np.mean(ious)
avg_dice = np.mean(dices)

print(f"Average IoU: {avg_iou:.4f}")
print(f"Average Dice Coefficient: {avg_dice:.4f}")

import os
import numpy as np
from skimage.io import imread

# Directory paths
predicted_masks_path = '/content/drive/MyDrive/EfficientNet/predicted_masks_efficientnet/'

# Get list of predicted mask files
mask_files = os.listdir(predicted_masks_path)

# Lists to store results
coverage_percentages = []
risk_categories = []
interpretations = []

# Iterate through each predicted mask
for mask_file in mask_files:
    mask_path = os.path.join(predicted_masks_path, mask_file)
    mask = imread(mask_path)

    # Binarize the mask
    mask_binarized = (mask > 127).astype(np.uint8)

    # Compute coverage percentage
    coverage = (np.sum(mask_binarized) / (mask_binarized.shape[0] * mask_binarized.shape[1])) * 100
    coverage_percentages.append(coverage)

    # Determine risk category and interpretation
    if coverage < 5:
        risk_categories.append("Low Risk")
        interpretations.append("The analysis indicates a low presence of abnormal tissue. While it appears likely benign, always consider a professional dermatologist's advice.")
    elif coverage >= 5 and coverage <= 15:
        risk_categories.append("Medium Risk")
        interpretations.append("The model has identified some regions of potential concern. It's recommended to consult a dermatologist for a more detailed examination.")
    else:
        risk_categories.append("High Risk")
        interpretations.append("The analysis indicates a significant presence of abnormal tissue. Immediate consultation with a dermatologist is advised.")



for i, mask_file in enumerate(mask_files):
    print(f"Image: {mask_file}")
    print(f"Coverage Percentage: {coverage_percentages[i]:.2f}%")
    print(f"Risk Category: {risk_categories[i]}")
    print(f"Interpretation: {interpretations[i]}")
    print("-" * 50)  # to separate the results for each image

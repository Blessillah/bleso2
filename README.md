# bleso2
# -*- coding: utf-8 -*-
"""
# Smart Agriculture Platform: Crop Disease and Pest Detection Model

This notebook demonstrates the implementation of a supervised learning model (Convolutional Neural Network - CNN)
for classifying crop images to detect diseases or pests. This is a crucial component of the
'Smart Agriculture Platform' aimed at addressing SDG 2: Zero Hunger by enabling early intervention.

## Concepts Covered:
- Supervised Learning (Classification)
- Computer Vision and Image Preprocessing
- Convolutional Neural Networks (CNNs) using Keras
- Data Augmentation
- Model Training and Evaluation (Accuracy, Confusion Matrix)

## Dataset (Placeholder):
For a real project, you would use a dataset like PlantVillage, which contains images of healthy
and diseased plant leaves. For this example, we'll simulate a small dataset.
"""

# 1. Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

print("Libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# --- 2. Data Simulation (Placeholder for Image Data) ---
# In a real project, you would load images from directories.
# Example:
# base_dir = 'path/to/your/plant_disease_dataset'
# train_dir = os.path.join(base_dir, 'train')
# val_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')

# For this demo, we'll simulate a small dataset structure and image loading.
# Assume 3 classes: 'Healthy', 'Disease_A', 'Disease_B'
num_classes = 3
image_size = (64, 64) # Smaller size for faster demo
batch_size = 32

# Simulate image paths and labels
def create_synthetic_image_data(num_samples_per_class, image_shape, num_classes):
    X_data = []
    y_data = []
    class_names = ['Healthy', 'Disease_A', 'Disease_B']
    for i in range(num_classes):
        for _ in range(num_samples_per_class):
            # Simulate a random image (e.g., grayscale for simplicity, or RGB)
            # In a real scenario, this would be actual image data (numpy array from image file)
            img = np.random.rand(image_shape[0], image_shape[1], 3) * 255 # RGB image
            X_data.append(img.astype(np.uint8))
            y_data.append(i) # Assign numerical label

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data, class_names

num_samples = 150 # Total 450 images (150 per class)
X_simulated, y_simulated, class_names = create_synthetic_image_data(num_samples, image_size, num_classes)

print(f"\nSimulated Image Data Shape: {X_simulated.shape}")
print(f"Simulated Labels Shape: {y_simulated.shape}")
print(f"Class Names: {class_names}")

# Split the simulated data into training and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_simulated, y_simulated, test_size=0.2, random_state=42, stratify=y_simulated
)

# Convert labels to one-hot encoding for Keras
y_train = to_categorical(y_train_raw, num_classes=num_classes)
y_test = to_categorical(y_test_raw, num_classes=num_classes)

print(f"\nX_train_raw shape: {X_train_raw.shape}, y_train shape: {y_train.shape}")
print(f"X_test_raw shape: {X_test_raw.shape}, y_test shape: {y_test.shape}")

# --- 3. Image Preprocessing and Data Augmentation ---

# ImageDataGenerator for scaling and augmentation
# Rescale pixel values to [0, 1]
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,       # Randomly rotate images by 20 degrees
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Apply shear transformation
    zoom_range=0.2,          # Randomly zoom into images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Fill newly created pixels after rotation or shift
)

test_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for test data (no augmentation)

# Flow from numpy arrays
train_generator = train_datagen.flow(X_train_raw, y_train, batch_size=batch_size)
test_generator = test_datagen.flow(X_test_raw, y_test, batch_size=batch_size)

print("\nImage data generators created with augmentation for training.")

# --- 4. Build Your Model (Convolutional Neural Network - CNN) ---

# Define the CNN model architecture
model = Sequential([
    # Input layer: Expects images of image_size (64x64) with 3 color channels (RGB)
    Input(shape=(image_size[0], image_size[1], 3)),

    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1'),
    Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
    MaxPooling2D((2, 2), name='pool1'),
    Dropout(0.25), # Dropout for regularization to prevent overfitting

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
    MaxPooling2D((2, 2), name='pool2'),
    Dropout(0.25),

    # Flatten the 3D output to 1D for the Dense layers
    Flatten(name='flatten'),

    # Fully Connected (Dense) Layers
    Dense(128, activation='relu', name='dense1'),
    Dropout(0.5), # Higher dropout for the dense layer
    Dense(num_classes, activation='softmax', name='output_layer') # Output layer with softmax for multi-class classification
])

# Compile the model
# Optimizer: Adam is a good general-purpose optimizer.
# Loss function: Categorical Crossentropy for multi-class classification with one-hot encoded labels.
# Metrics: Accuracy to monitor performance during training.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n--- Model Summary ---")
model.summary()

# --- 5. Model Training ---

# Train the model using the data generators
# steps_per_epoch: Number of batches to draw from the generator before declaring one epoch finished.
# validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
# epochs: Number of complete passes through the training dataset.
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train_raw) // batch_size,
    epochs=50, # Reduced epochs for faster demo; increase for better performance
    validation_data=test_generator,
    validation_steps=len(X_test_raw) // batch_size,
    verbose=1
)

print("\nModel training complete!")

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 6. Model Evaluation ---

# Evaluate the model on the test data
print("\n--- Model Evaluation on Test Set ---")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions for the test set
# It's better to predict on the raw test data after scaling it, not through the generator,
# to ensure consistent order for confusion matrix.
X_test_scaled = X_test_raw / 255.0 # Manually scale test images
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1) # Convert probabilities to class labels

# Generate confusion matrix
cm = confusion_matrix(y_test_raw, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Generate classification report
print("\n--- Classification Report ---")
print(classification_report(y_test_raw, y_pred, target_names=class_names))

# --- Example of making a single prediction ---
print("\n--- Single Prediction Example ---")

# Take a random image from the test set for demonstration
sample_index = random.randint(0, len(X_test_raw) - 1)
sample_image = X_test_raw[sample_index]
true_label_idx = y_test_raw[sample_index]
true_label_name = class_names[true_label_idx]

# Preprocess the single image: add batch dimension and scale
processed_sample_image = np.expand_dims(sample_image / 255.0, axis=0)

# Make prediction
prediction_probs = model.predict(processed_sample_image)[0]
predicted_label_idx = np.argmax(prediction_probs)
predicted_label_name = class_names[predicted_label_idx]

print(f"True Label: {true_label_name}")
print(f"Predicted Label: {predicted_label_name}")
print(f"Prediction Probabilities: {prediction_probs}")

# Display the sample image
plt.figure(figsize=(4, 4))
plt.imshow(sample_image.astype(np.uint8)) # Display original uint8 image
plt.title(f"Actual: {true_label_name}\nPredicted: {predicted_label_name}")
plt.axis('off')
plt.show()

"""
## Conclusion

This notebook provides a foundational CNN model for crop disease detection.
By accurately identifying diseases and pests early, farmers can apply targeted treatments,
reduce crop loss, and minimize the use of harmful chemicals, directly contributing
to sustainable agriculture and SDG 2: Zero Hunger.

## Next Steps / Stretch Goals:
- **Real Dataset:** Replace synthetic data with a real, larger dataset like PlantVillage.
- **Transfer Learning:** Utilize pre-trained models (e.g., VGG16, ResNet) for better performance with less data.
- **Model Optimization:** Fine-tune hyperparameters, experiment with different CNN architectures.
- **Deployment:** Integrate the model into a mobile application for on-field diagnosis.
- **Multi-label Classification:** If an image can have multiple diseases, adjust the output layer and loss function.
"""

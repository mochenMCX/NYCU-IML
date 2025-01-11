import numpy as np
import argparse
import cv2
import csv
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tqdm import tqdm


output_file = 'output.csv'
# Build the model
model = Sequential()

# Convolutional Block 1
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Convolutional Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Convolutional Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes for emotion detection

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
tensorboard = TensorBoard(log_dir="logs", histogram_freq=1)

model.summary()
total_params = model.count_params()
print(f"Total parameters: {total_params}")

model_size = total_params * 4 / (1024 * 1024)
print(f"Estimated model size: {model_size:.2f} MB")

test_dir = 'data/Images/test'
model.load_weights('best_model.h5')
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "label"])  # Write header

    for image_file in tqdm(os.listdir(test_dir)):
        image_path = os.path.join(test_dir, image_file)

        # Load, preprocess, and predict
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error reading image: {image_path}")
            continue

        resized_image = cv2.resize(image, (48, 48)) / 255.0
        cropped_img = np.expand_dims(np.expand_dims(resized_image, axis=-1), axis=0)

        prediction = model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))
        writer.writerow([os.path.splitext(image_file)[0], maxindex])

print(f"Predictions saved to {output_file}")
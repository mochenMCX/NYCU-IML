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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display/test")
mode = ap.parse_args().mode

# Parameters
train_dir = 'data/Images/train'
val_dir = 'data/Images/train'
test_dir = 'data/Images/test'
output_file = 'output.csv'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 100

# Check for GPU
if tf.config.experimental.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("No GPU found, using CPU")

# Data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

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

# Train the model
if mode == "train":
    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001, decay=1e-5),
            metrics=['accuracy'])

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(48, 48), batch_size=batch_size,
        color_mode="grayscale", class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(48, 48), batch_size=batch_size,
        color_mode="grayscale", class_mode='categorical')

    model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        validation_data=val_generator,
        validation_steps=num_val // batch_size,
        epochs=num_epoch,
        callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard]
    )

    model.save_weights('model_final.h5')

# Display (Prediction Mode)
elif mode == "display":
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

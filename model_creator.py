import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

# Paths to training and testing datasets
train_dir = 'archive/train'  # Replace with your training data path
test_dir = 'archive/test'    # Replace with your testing data path

# Hyperparameters
img_width, img_height = 48, 48  # Standard for emotion recognition
batch_size = 32
epochs = 100  # Increased epochs for better training

# Data Augmentation for training set
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,  # Increased zoom range
    brightness_range=[0.8, 1.2],  # Added brightness variation
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for testing set
datagen_test = ImageDataGenerator(rescale=1./255)

# Load images from folders with labels
train_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = datagen_test.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    BatchNormalization(),  # Added Batch Normalization
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # L2 regularization
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dropout(0.5),  # Dropout to prevent overfitting
    
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Increased layer size, added L2
    Dropout(0.5),  # Increased Dropout to 0.5
    
    Dense(7, activation='softmax')  # 7 classes for each emotion
])

# Compile the model with learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('emotion_cnn_model_v2.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

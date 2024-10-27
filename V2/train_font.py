import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--img_height', type=int, default=96, help='Height of the input images')
parser.add_argument('--img_width', type=int, default=192, help='Width of the input images')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes in the dataset')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
parser.add_argument('--output', type=str, default="captcha_font_classifier_final.keras", help="Where to store the model")

# Parse the arguments
args = parser.parse_args()

# Parameters
data_dir = args.data_dir
batch_size = args.batch_size
img_height = args.img_height
img_width = args.img_width
num_classes = args.num_classes
epochs = args.epochs

# Parameters
batch_size = 64  
img_height = 96
img_width = 192  
num_classes = 2   
epochs = 50       

# Create training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,      # 20% for validation
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'     
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale'
)

class_names = train_ds.class_names
print("Class Names:", class_names)

# Normalize the images to [0, 1]
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation to improve generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),           # ±10 degrees
    layers.RandomZoom(0.1),               # 10% zoom
    layers.RandomTranslation(0.1, 0.1),   # 10% horizontally and vertically
    layers.RandomFlip("horizontal"),
    layers.GaussianNoise(0.01),           # Minimal noise
])

# Build the CNN model with recommended hyperparameters
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 1)),
    
    data_augmentation,
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout rate of 0.5
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Dropout rate of 0.3
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with recommended hyperparameters
optimizer = optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Define callbacks
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_captcha_font_classifier.keras",
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-6
)

early_stopping_cb = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb]
)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_ds)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Save the trained model
model.save('captcha_font_classifier_final.keras')

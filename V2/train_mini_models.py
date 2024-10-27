import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Data Preparation

# Parameters
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 96
IMAGE_CHANNELS = 3
MIN_CAPTCHA_LENGTH = 4  # Minimum number of characters in captcha
MAX_CAPTCHA_LENGTH = 6  # Maximum number of characters in captcha
BATCH_SIZE = 64
EPOCHS = 50
SYMBOLS_FILE = 'symbols.txt'
DATA_DIR = 'V2/data/final_4_6_cleaned_wild_crazy'  # Replace with your captcha images directory

# Derived Parameters
NUM_CHAR_CLASSES = MAX_CAPTCHA_LENGTH - MIN_CAPTCHA_LENGTH + 1  # Number of possible captcha lengths

# Load symbol set
with open(SYMBOLS_FILE, 'r') as f:
    symbols = f.read().strip()
num_symbols = len(symbols)
symbol_to_num = {s: i for i, s in enumerate(symbols)}
num_classes = num_symbols + 1  # +1 for the blank/padding character

# Function to encode labels
def encode_label(label):
    # Remove any trailing '_number' pattern
    label = re.sub(r'_\d+$', '', label)
    # Replace '~' with '\' before encoding
    label = label.replace('~', '\\')
    encoded = [symbol_to_num[char] for char in label]
    # Pad with a special 'blank' symbol (num_symbols) up to MAX_CAPTCHA_LENGTH
    while len(encoded) < MAX_CAPTCHA_LENGTH:
        encoded.append(num_symbols)
    return encoded

# Load data
images = []
labels = []
num_chars = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.png'):
        # Extract label from filename (assuming filename is label.png)
        label_str = os.path.splitext(filename)[0]
        label_length = len(label_str)
        # Ensure label length is within specified bounds
        if MIN_CAPTCHA_LENGTH <= label_length <= MAX_CAPTCHA_LENGTH:
            label_encoded = encode_label(label_str)
            images.append(os.path.join(DATA_DIR, filename))
            labels.append(label_encoded)
            num_chars.append(label_length)

images = np.array(images)
labels = np.array(labels)
num_chars = np.array(num_chars)

# Split into training and validation
train_images, val_images, train_labels, val_labels, train_num_chars, val_num_chars = train_test_split(
    images, labels, num_chars, test_size=0.2, random_state=42
)

# Data Generator
def data_generator(image_paths, labels, num_chars, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        channel_shift_range=10.,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    while True:
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_indices = indices[start:end]
            batch_images = []
            batch_labels = []
            batch_num_chars = []
            for i in batch_indices:
                img = tf.keras.preprocessing.image.load_img(
                    image_paths[i], target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
                )
                img = tf.keras.preprocessing.image.img_to_array(img)
                batch_images.append(img)
                batch_labels.append(labels[i])
                batch_num_chars.append(num_chars[i])
            batch_images = np.array(batch_images, dtype='float32') / 255.0
            batch_labels = np.array(batch_labels)
            batch_num_chars = np.array(batch_num_chars)
            # One-hot encode labels for each character position
            batch_labels_ohe = {
                f'char_{i+1}': to_categorical(batch_labels[:, i], num_classes=num_classes)
                for i in range(MAX_CAPTCHA_LENGTH)
            }
            # One-hot encode number of characters
            batch_num_chars_adjusted = batch_num_chars - MIN_CAPTCHA_LENGTH
            batch_labels_ohe['num_chars'] = to_categorical(
                batch_num_chars_adjusted, num_classes=NUM_CHAR_CLASSES
            )
            yield (batch_images, batch_labels_ohe)

# Define output_signature
output_signature = (
    tf.TensorSpec(shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32),
    {
        f'char_{i+1}': tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        for i in range(MAX_CAPTCHA_LENGTH)
    } | {'num_chars': tf.TensorSpec(shape=(None, NUM_CHAR_CLASSES), dtype=tf.float32)}
)

train_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_labels, train_num_chars, BATCH_SIZE),
    output_signature=output_signature
)
val_gen = tf.data.Dataset.from_generator(
    lambda: data_generator(val_images, val_labels, val_num_chars, BATCH_SIZE),
    output_signature=output_signature
)


# 2. Model Architecture

# Input layer
input_tensor = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

# Convolutional layers
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_tensor)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)

# Flatten
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# Output layers for each character position
outputs = []
for i in range(MAX_CAPTCHA_LENGTH):
    outputs.append(
        layers.Dense(num_classes, activation='softmax', name=f'char_{i+1}')
        (x)
    )

# Output layer for number of characters
num_chars_output = layers.Dense(
    NUM_CHAR_CLASSES, activation='softmax', name='num_chars'
)(x)
outputs.append(num_chars_output)

# Define the model
model = models.Model(inputs=input_tensor, outputs=outputs)

# 3. Compile the Model

# Define loss for each output
losses = {
    f'char_{i+1}': 'categorical_crossentropy' for i in range(MAX_CAPTCHA_LENGTH)
}
losses['num_chars'] = 'categorical_crossentropy'

# Define loss weights (optional, can be adjusted)
loss_weights = {
    f'char_{i+1}': 1.0 for i in range(MAX_CAPTCHA_LENGTH)
}
loss_weights['num_chars'] = 1.0

# Compile the model
model.compile(
    optimizer=optimizers.Adam(),
    loss=losses,
    loss_weights=loss_weights,
    metrics={
        **{f'char_{i+1}': 'accuracy' for i in range(MAX_CAPTCHA_LENGTH)},
        'num_chars': 'accuracy'
    }
)

model.summary()

# 4. Training

# Calculate steps per epoch
steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

# Create generators
# train_gen = data_generator(train_images, train_labels, train_num_chars, BATCH_SIZE)
# val_gen = data_generator(val_images, val_labels, val_num_chars, BATCH_SIZE)

# Fit the model
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps
)

# 5. Evaluation and Prediction

# Function to decode predictions
def decode_predictions(preds):
    char_preds = []
    for i in range(MAX_CAPTCHA_LENGTH):
        pred = np.argmax(preds[f'char_{i+1}'], axis=-1)
        char_preds.append(pred)
    num_chars_pred = np.argmax(preds['num_chars'], axis=-1) + MIN_CAPTCHA_LENGTH
    result = []
    for i in range(len(num_chars_pred)):
        captcha = ''
        for j in range(num_chars_pred[i]):
            if char_preds[j][i] < num_symbols:
                captcha += symbols[char_preds[j][i]]
            else:
                # Handle blank/padding if necessary
                pass
        result.append(captcha)
    return result

# Example Prediction
def predict_captcha(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    preds = model.predict(img)
    decoded = decode_predictions(preds)
    return decoded[0]

# Example usage
# captcha_text = predict_captcha('path_to_some_captcha.png')
# print(captcha_text)

# Save the model
model.save('updated_wild_crazy_model_4_6.h5')

# Load the model
# model = models.load_model('captcha_model.h5')

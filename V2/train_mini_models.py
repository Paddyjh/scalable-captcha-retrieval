import os
import numpy as np
import tensorflow as tf
import re
import argparse
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='CAPTCHA Solver Training')
parser.add_argument('--min_captcha_length', type=int, default=4, help='Minimum number of characters in CAPTCHA')
parser.add_argument('--max_captcha_length', type=int, default=6, help='Maximum number of characters in CAPTCHA')
parser.add_argument('--data_dir', type=str, help='Directory of CAPTCHA images')
parser.add_argument('--output_model_prefix', type=str, help='Prefix for the output model name')
args = parser.parse_args()

# Assign command-line arguments to variables
MIN_CAPTCHA_LENGTH = args.min_captcha_length
MAX_CAPTCHA_LENGTH = args.max_captcha_length
DATA_DIR = args.data_dir
OUTPUT_MODEL = f"{args.output_model_prefix}_{MIN_CAPTCHA_LENGTH}_{MAX_CAPTCHA_LENGTH}.h5"

# Parameters
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 96
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 35
SYMBOLS_FILE = 'symbols.txt'

# Derived Parameters
NUM_CHAR_CLASSES = MAX_CAPTCHA_LENGTH - MIN_CAPTCHA_LENGTH + 1

# Load symbol set
with open(SYMBOLS_FILE, 'r') as f:
    symbols = f.read().strip()
num_symbols = len(symbols)
symbol_to_num = {s: i for i, s in enumerate(symbols)}
num_classes = num_symbols + 1  # +1 for the blank/padding character

# Function to encode labels
def encode_label(label):
    label = re.sub(r'_\d+$', '', label)
    label = label.replace('~', '\\')
    encoded = [symbol_to_num[char] for char in label]
    while len(encoded) < MAX_CAPTCHA_LENGTH:
        encoded.append(num_symbols)
    return encoded

# Load data
images = []
labels = []
num_chars = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.png'):
        label_str = os.path.splitext(filename)[0]
        label_length = len(label_str)
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
            batch_labels_ohe = {
                f'char_{i+1}': to_categorical(batch_labels[:, i], num_classes=num_classes)
                for i in range(MAX_CAPTCHA_LENGTH)
            }
            batch_num_chars_adjusted = batch_num_chars - MIN_CAPTCHA_LENGTH
            batch_labels_ohe['num_chars'] = to_categorical(
                batch_num_chars_adjusted, num_classes=NUM_CHAR_CLASSES
            )
            yield (batch_images, batch_labels_ohe)

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

# Model Architecture
input_tensor = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_tensor)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)

outputs = []
for i in range(MAX_CAPTCHA_LENGTH):
    outputs.append(
        layers.Dense(num_classes, activation='softmax', name=f'char_{i+1}')(x)
    )

num_chars_output = layers.Dense(
    NUM_CHAR_CLASSES, activation='softmax', name='num_chars'
)(x)
outputs.append(num_chars_output)

model = models.Model(inputs=input_tensor, outputs=outputs)

losses = {
    f'char_{i+1}': 'categorical_crossentropy' for i in range(MAX_CAPTCHA_LENGTH)
}
losses['num_chars'] = 'categorical_crossentropy'

loss_weights = {
    f'char_{i+1}': 1.0 for i in range(MAX_CAPTCHA_LENGTH)
}
loss_weights['num_chars'] = 1.0

model.compile(
    optimizer=optimizers.Adam(),
    loss=losses,
    loss_weights=loss_weights,
    metrics={
        **{f'char_{i+1}': 'accuracy' for i in range(MAX_CAPTCHA_LENGTH)},
        'num_chars': 'accuracy'
    }
)

steps_per_epoch = len(train_images) // BATCH_SIZE
validation_steps = len(val_images) // BATCH_SIZE

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=validation_steps
)

model.save(OUTPUT_MODEL)

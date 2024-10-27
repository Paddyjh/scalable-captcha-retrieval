import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import argparse

# Parameters
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 96
SYMBOLS_FILE = 'symbols.txt'

# Load symbol set
with open(SYMBOLS_FILE, 'r') as f:
    symbols = f.read().strip()
num_symbols = len(symbols)
symbol_to_num = {s: i for i, s in enumerate(symbols)}

# Function to decode predictions
def decode_predictions(preds, min_captcha_length, max_captcha_length):
    char_preds = []
    num_chars_pred = np.argmax(preds[-1], axis=-1) + min_captcha_length  # The last output is for the number of characters

    # Extract predictions for each character position
    for i in range(max_captcha_length):
        if i < len(preds) - 1:  # Ignore the last element (num_chars)
            pred = np.argmax(preds[i], axis=-1)
            char_preds.append(pred)
        else:
            # Pad with blanks if no more predictions available
            char_preds.append(np.full(preds[0].shape[0], num_symbols))

    # Decode characters for each CAPTCHA in the batch
    result = []
    for j in range(len(num_chars_pred)):
        captcha = ''
        for k in range(num_chars_pred[j]):
            if char_preds[k][j] < num_symbols:
                captcha += symbols[char_preds[k][j]]
            else:
                # Skip blank/padding characters
                pass
        result.append(captcha)

    return result

# Function to load and preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

# Function to predict CAPTCHA text
def predict_captcha(model, image_path, min_captcha_length, max_captcha_length):
    img = preprocess_image(image_path)
    preds = model.predict(img)
    decoded = decode_predictions(preds, min_captcha_length, max_captcha_length)
    return decoded[0]

# Function to classify all images in a directory
def classify_directory(model_path, image_dir, min_captcha_length, max_captcha_length):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    predictions = {}

    # Iterate through each image in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            captcha_text = predict_captcha(model, image_path, min_captcha_length, max_captcha_length)
            predictions[filename] = captcha_text
            print(f"{filename}: {captcha_text}")

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict CAPTCHA text from images using a trained model.')
    parser.add_argument('--min_captcha_length', type=int, help='Minimum CAPTCHA length', required=True)
    parser.add_argument('--max_captcha_length', type=int, help='Maximum CAPTCHA length', required=True)
    parser.add_argument('--model_path', type=str, help='Path to the trained model', required=True)
    parser.add_argument('--image_dir', type=str, help='Directory containing CAPTCHA images', required=True)
    parser.add_argument('--predictions_file', type=str, help='Filename for saving predictions (without .txt)', required=True)

    args = parser.parse_args()

    # Run classification
    predictions = classify_directory(args.model_path, args.image_dir, args.min_captcha_length, args.max_captcha_length)

    # Save predictions to a file
    with open(f'{args.predictions_file}.txt', 'w') as f:
        for filename, text in predictions.items():
            f.write(f"{filename}: {text}\n")

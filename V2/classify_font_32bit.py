import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os
import shutil
import argparse

# Configuration
MODEL_PATH = 'captcha_font_classifier_final.keras'
# INPUT_DIR = 'patrick-files/captchas'               
# OUTPUT_DIR = 'patrick-files/captcha-categorized-by-font'         
IMAGE_HEIGHT = 96
IMAGE_WIDTH = 192
COLOR_MODE = 'grayscale'                           

# # Just going to hard code these but change as necessary 
# # these should essentially be the directory names 
# CLASS_NAMES = ['WildCrazy', 'cRAZYsTYLE']

def parse_arguments():
    parser = argparse.ArgumentParser(description='Classify CAPTCHA images by font.')
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        required=True,
        help='Path to the input directory containing CAPTCHA images.'
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='Path to the output directory to store classified images.'
    )
    parser.add_argument(
        '-m', '--model_path',
        type=str,
        default=MODEL_PATH,
        help=f'Path to the trained model file (default: {MODEL_PATH}).'
    )
    
    parser.add_argument(
        '-c', '--class-name-1',
        type=str,
        help=f'First font class name.'
    )
        
    parser.add_argument(
        '-d', '--class-name-2',
        type=str,
        help=f'Second font class name .'
    )
    return parser.parse_args()



def load_trained_model(model_path):
    try:
        interpreter = tflite.Interpreter(model_path)
        interpreter.allocate_tensors()
        print(f"Model loaded successfully from {model_path}.")
        return interpreter
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_image(img_path):
    try:
        img = Image.open(img_path)
        if COLOR_MODE == 'grayscale':
            img = img.convert('L')  # Convert to grayscale
        else:
            img = img.convert('RGB')  # Convert to RGB

        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.array(img)

        if COLOR_MODE == 'grayscale':
            img_array = img_array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 1))

        img_array = img_array / 255.0  # Normalize to [0,1]

        # Expand dimensions to match model's expected input shape (batch_size, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def classify_images(interpreter, class_names, input_dir, output_dir):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Create output directories if they don't exist
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # Iterate through all PNG files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            img_path = os.path.join(input_dir, filename)
            img_array = preprocess_image(img_path)

            if img_array is None:
                continue  # Skip this image due to preprocessing error

            # Predict the class
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            predicted_class = class_names[int(prediction[0][0] > 0.5)]  # Threshold at 0.5

            # Define source and destination paths
            src = img_path
            dest = os.path.join(output_dir, predicted_class, filename)

            try:
                shutil.move(src, dest)
                print(f"Moved {filename} to {predicted_class}/")
            except Exception as e:
                print(f"Error moving file {filename}: {e}")

def main():
    # Parse command-line arguments
    args = parse_arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir
    model_path = args.model_path
    class_names = [args.class_name_1, args.class_name_2]

    # Load the model
    interpreter = load_trained_model(model_path)

    # Verify class names
    print(f"Class names: {class_names}")

    # Classify and organize images
    classify_images(interpreter, class_names, input_dir, output_dir)
    print("Classification and organization complete.")


if __name__ == "__main__":
    main()

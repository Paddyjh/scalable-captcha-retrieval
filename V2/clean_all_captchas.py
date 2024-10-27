import argparse
import os
import cv2
import numpy as np
from scipy.ndimage import median_filter

def remove_circles(img):
    hough_circle_locations = cv2.HoughCircles(
        img,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=5,
        minRadius=0,
        maxRadius=2
    )
    if hough_circle_locations is not None:
        circles = hough_circle_locations[0]
        for circle in circles:
            x, y, r = map(int, circle[:3])
            img = cv2.circle(img, center=(x, y), radius=r, color=(255), thickness=2)
    return img

# def median_blur_rectangular(image, k_height, k_width):
#     padded_image = cv2.copyMakeBorder(
#         image,
#         k_height // 2,
#         k_height // 2,
#         k_width // 2,
#         k_width // 2,
#         cv2.BORDER_REFLECT,
#         value=0
#     )
#     output = np.zeros_like(image)
#     for y in range(image.shape[0]):
#         for x in range(image.shape[1]):
#             window = padded_image[y:y + k_height, x:x + k_width]
#             output[y, x] = np.median(window)
#     return output

def median_blur_rectangular(image, k_height, k_width):
    if not (isinstance(k_height, int) and isinstance(k_width, int)):
        raise TypeError("Kernel dimensions must be integers.")
    if k_height <= 0 or k_width <= 0:
        raise ValueError("Kernel dimensions must be positive.")

    # Apply the median filter using SciPy
    return median_filter(image, size=(k_height, k_width), mode='reflect')

def remove_noise(orig_img, display=False):
    img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    img = median_blur_rectangular(img, 5, 1)
    img = median_blur_rectangular(img, 1, 3)
    img = remove_circles(img)  
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    min_size = 100
    output_image = np.full(img.shape, 255, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output_image[labels == i] = 0

    if display:
        cv2.imshow('Original Image', orig_img)
        cv2.imshow('Processed Image', output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_image

def preprocess(img):
    return remove_noise(img, display=False)

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for idx, filename in enumerate(os.listdir(input_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Unable to read image '{img_path}'. Skipping.")
                continue
            
            clean_img = preprocess(img)
            clean_output_path = os.path.join(output_dir, filename)
            cv2.imwrite(clean_output_path, clean_img)
            
            if idx % 100 == 0:
                print(f'Progress: Processed {idx} images in "{input_dir}"')

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess captchas in a single directory and save cleaned images to the specified output directory.'
    )
    parser.add_argument(
        '--captcha-dir',
        help='Directory containing captchas to preprocess',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save cleaned images',
        type=str,
        required=True
    )
    args = parser.parse_args()

    if not os.path.isdir(args.captcha_dir):
        print(f"Error: The directory '{args.captcha_dir}' does not exist or is not a directory.")
        exit(1)

    print(f"Processing directory: {args.captcha_dir}")
    process_directory(args.captcha_dir, args.output_dir)
    print(f"Finished processing directory: {args.captcha_dir}")

if __name__ == "__main__":
    main()

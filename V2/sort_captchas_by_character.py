import argparse
import os
import cv2
import numpy as np

# Please note that if you set visuailize to True then you need to pass in a unique index into this function (otherwise files will get overwritten)
def segment(cleaned, two_char_min, three_char_min, four_char_min, index=0, visualization_dir='visualizations'):
    os.makedirs(visualization_dir, exist_ok=True)

    # Convert to binary and prepare markers for watershed
    _, thresh = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=5)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR), markers)

    # Find contours and count characters
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    character_count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 6 and h > 6:  # Filter noise
            if w > two_char_min and w <= three_char_min:
                character_count += 2
            elif w > three_char_min and w <= four_char_min:
                character_count += 3
            elif w > four_char_min:
                character_count += 4
            else:
                character_count += 1

    return character_count



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captcha-dir', help='Where to read the captchas', type=str)
    parser.add_argument('--font', help='The capctcha font used', type=str)
    args = parser.parse_args()

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to split")
        exit(1)
        
    
    two_char_min = [40]
    three_char_min = [66]
    four_char_min = [101]
    
    one_three = 0
    two_four = 0
    three_four = 0
    four_six = 0
    
    font_name = args.font
    
    for i in range(len(two_char_min)):
        correctly_segmented = 0
        for idx, filename in enumerate(os.listdir(args.captcha_dir)):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                
                img_path = os.path.join(args.captcha_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Update this line to use character count instead of segmented images
                number_of_segmented_characters = segment(img, index=idx, two_char_min=two_char_min[i], three_char_min=three_char_min[i], four_char_min=four_char_min[i])
  
                if number_of_segmented_characters == 1 or number_of_segmented_characters == 2:
                    cv2.imwrite(f'V2/data/model_data/{font_name}/1_3/{filename}', img)
                elif number_of_segmented_characters == 3:
                    cv2.imwrite(f'V2/data/model_data/{font_name}/2_4/{filename}', img)
                elif number_of_segmented_characters == 4:
                    cv2.imwrite(f'V2/data/model_data/{font_name}/3_5/{filename}', img)
                elif number_of_segmented_characters == 5 or number_of_segmented_characters == 6:
                    cv2.imwrite(f'V2/data/model_data/{font_name}/4_6/{filename}', img)
  

 
                  
                # if number_of_segmented_characters == 1 or number_of_segmented_characters == 2:
                #     cv2.imwrite(f'V2/data/model_data/{font_name}/1_3', filename)
                # elif number_of_segmented_characters == 3:
                #     cv2.imwrite(f'V2/data/model_data/{font_name}/2_4', filename)
                # elif number_of_segmented_characters == 4:
                #     cv2.imwrite(f'V2/data/model_data/{font_name}/3_5', filename)
                # elif number_of_segmented_characters == 5 or number_of_segmented_characters == 6:
                #     cv2.imwrite(f'V2/data/model_data/{font_name}/5_6', filename)



if __name__ == "__main__":
    main()

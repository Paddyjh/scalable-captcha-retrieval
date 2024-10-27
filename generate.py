# Initial code provided by Ciaran McGoldrick
#!/usr/bin/env python3

import os
import numpy
import random
import cv2
import argparse
import captcha.image
import preprocess_testing

def generate_image(captcha_symbols, length, output_dir, captcha_generator, vary_size, min_length):
    
    # Checks if captcha length should vary and if so generates a random value between min_length and max length else use length param
    if vary_size:
        captcha_length = random.randint(min_length, length)  
    else:
        captcha_length = length
        
    random_str = ''.join([random.choice(captcha_symbols)
                            for j in range(captcha_length)])
    filename_str = random_str
    if "\\" in filename_str:
        filename_str = filename_str.replace("\\", "~")
    image_path = os.path.join(output_dir, filename_str+'.png')
    
    if os.path.exists(image_path):
        version = 1
        while os.path.exists(os.path.join(output_dir, filename_str + '_' + str(version) + '.png')):
            version += 1
        image_path = os.path.join(
            output_dir, filename_str + '_' + str(version) + '.png')

    image = numpy.array(captcha_generator.generate_image(random_str))
    cv2.imwrite(image_path, image)

def generate(width, height, length, count, output_dir, symbols, font, vary_size, min_length):
    captcha_generator = captcha.image.ImageCaptcha(
        width=width, height=height, fonts=[font])
    symbols_file = open(symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(output_dir):
        print("Creating output directory " + output_dir)
        os.makedirs(output_dir)

    for i in range(count):
        generate_image(captcha_symbols, length, output_dir, captcha_generator, vary_size, min_length)
        if (i+1) % 100 == 0 or (i+1) == count: 
            print(f"Generated {i+1}/{count} CAPTCHAs")
        
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument(
        '--length', help='Length of captchas in characters', type=int)
    parser.add_argument(
        '--count', help='How many captchas to generate', type=int)
    parser.add_argument(
        '--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument(
        '--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument(
        '--font', help='Path for font to use', type=str)
    parser.add_argument(
        '--vary-char-size', help='Vary the number of captcha characters (default: False)', action='store_true', default=False)
    parser.add_argument(
        '--min-length', help='Minimum length of characters in captchas', type=int, default=1)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    if args.font is None:
        print("Please specify the font")
        exit(1)
    generate(args.width, args.height, args.length, args.count, args.output_dir, args.symbols, args.font, args.vary_char_size, args.min_length)


if __name__ == '__main__':
    main()

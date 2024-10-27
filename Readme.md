# Scalable Computing Group 001/002 Attempt 2 - By Patrick Haughey 20336099

## Instructions to replicate my solution can be found below
*please note you can find the command associated with each step below this section
1. Generating test/validation data: This step involves generating 25,000 captchas for each of the eight models that we will train. I recommend generating 25,0000 captchas per model. Each model will be a font-specific classifier for captchas of varying character lengths as follows (1-3, 2-4, 3-5, 4-6).
  1.1 This step involves generating the 25,000 captchas for each model. Each of these 25,000 captchas should be stored in a unique directory relating to their model
  1.2 Once all the captchas have been generated, they must be cleaned; you should repeat the command in 1.2 for each directory you made relating to each model in step 1.1. Each of these 25,000 batches of cleaned captchas should be stored in its own model-specific folder
2. You will now generate eight models using the command in section 2 below. You should execute this command for each of the 8 models. You should specify the appropriate max and min length of each captcha model, including the specific directory of the captchas for the given model created in step 1.2.
3. Once you have trained all eight of your models in step 2, you can begin to classify the assignment-specific captchas.
  3.1
    3.1.1 Carry out the command in part 3.1 below twice, once for each font. This will create 50,000 captchas of varying character lengths for each font type.
    3.1.2 You can then train the font identification model as outlined in detail in 3.1.2 below.
    3.1.3 Finally, you can break the assignment-specific captchas into their given fonts by using the command in part 3.1.3 below.
  3.2 Once we have separated the captchas by font, we can then preprocess them using the command in 3.2 below.
  3.3 Once we have cleaned the files, we can then use the command in 3.4 to classify all the captchas for a given font into their given character ranges (i.e which models they will be fed into). All the captchas will be put into a directory based on their number of characters. This will make passing them into the given model with the same range easier for classification.
  3.4 Once we have categorised all the captchas by character length, we can pass each of these directories into the classification code by using the command in 3.4. Ensure to pass in the correct number of minimum and maximum characters along with the correct associated model for that character range and font. This process should be repeated until all the directories have been classified. Please also ensure that you provide unique names for the outputted text file and ensure that all of these .txt files are stored in one shared directory to make the next processing step easier.
  3.5 When you have classified all the different character range directories and have the output .txt files all in the same directory, pass this directory into the command in part 3.5 below. This will create a 'combined_submission.csv', which you can then submit to submit to get your score. This file is the order collection of all the results of the output.txt files in the directory.

### 1 Generating test/validation data

#### 1.1 Captcha Generation
The first step is to generate 25,000 captchas (or however many captchas you decide) for each model.
Parameters:
width: int, sets the width of the captchas generated in pixels (do not change)
height: int, sets the height of the captchas generated in pixels (do not change)
length: int, the max number of characters in the generated captchas
count: int, number of captchas to be generated
output-dir: string, path to the output directory to save captchas
symbols: string, the path to the symbol set
font: string, path to font file for captcha generation 
vary-char-size: boolean (is present true if not included set to false), sets the number of characters generated to vary between captchas (withing the min length and length specified)
min-length: int, the minimum number of characters a captcha should have

```
python3 generate.py --width=192 --height=96 --length=max_number_of_characters --count=number_of_captchas --output-dir=location-of-output/directory --symbols=symbols.txt --font=location_of_font/font.ttf --vary-char-size --min-length=minimum_character_length
```

#### 1.2 Captcha Pre-Processing 
Once you have generated the captcha sets for each model, you then need to clean these captchas (i.e. perform pre-processing to remove noise). This can be done using the python command below:
Parameters:
captcha-dir: string, the path to the directory containing captchas to be cleaned.
output-dir: string, path the directory where the cleaned captchas should be stored.

```
python3 clean_all_captchas.py --captcha-dir=directory_containg_uncleaned_cpatchas --output-dir=location_of_cleaned_captchas
```

### 2. Train the models
Parameters: 
min_captcha_length: int, the minimum number of characters that the model will identify in characters
max_captcha_length: int, the maximum number of characters that the model will identify in characters
data_dir: string, path to the directory containing training captchas
output_model_prefix: string, filename of the model before the file type.
```
python3 train_mini_models.py --min_captcha_length=3 --max_captcha_length=5 --data_dir='/path/to/data' --output_model_prefix='model_name'
```

### 3. Classify the Captchas

#### 3.1 Classify captchas by font ( this was already discussed in submission 1)

##### 3.1.1 Generate Data to Train Font Classifier
I recommend generating 50,000 captchas for each font (100,000 total). Each captcha should have a width of 192 and a height of 96, and captchas should be generated using a random number of characters from 1 to 6 (to most accurately represent the data provided by Ciaran). **Note: ensure you include the '--vary-char-size' and do not modify width, height or length.**
```
python3 generate.py --width=192 --height=96 --font=user-files/fonts/FontName.ttf --output-dir=user-files/font-training-data/FontName --symbols=symbols.txt --count=50000 --length=6 --vary-char-size
```
**Run the above command twice (once for each font) and ensure you name the output directory as the font name**


##### 3.1.2 Train font identification model
This trains for both fonts simultaneously. I have created a Convolutional Neural Network with preselected hyperparameters (please note these parameters can be changed by being passed as arguments in the command line, but they have my default values). You must pass in a directory that contains two subdirectories in which each of these directories contains examples of each font. The names of these subdirectories should be the respective font names. Note that if you use follow the instructions above, this should be done automatically. 
```
python3 train_font.py --data_dir=user-files/font-training-data --output=user-files/font-identification-model
```
**Note: You can specify the hyperparameters if you would like, but I have set default values that I used for my model. The optional commands are: **

'--batch_size', type=int, default=64, help='Batch size for training'
'--img_height', type=int, default=96, help='Height of the input images'
'--img_width', type=int, default=192, help='Width of the input images'
'--num_classes', type=int, default=2, help='Number of classes in the dataset'
'--epochs', type=int, default=50, help='Number of epochs for training'

#### 3.1.3 Classify captchas based on fonts

Once you have trained your model, you can call the classify_font_32bit.py class and pass in the directory of your captchas. This will generate a new directory that will contain two sub-directories (one for each of your captchas categorised by font). You must also include the class name for each font; this will be the name of each font. I have also put a print statement in the training model so that you can identify the order of the fonts/ their respective names. These names should just be the names of the respective directories containing the font training data.  

```
python3 classify_font_32bit.py --input_dir=user-files/captchas --output_dir=user-files/captcha-categorised --class-name-1=FontName1 --class-name-2=FontName2 --model=user-files/font-identification-model.tflite
```


#### 3.2 Captcha Pre-Processing 
Repeat the steps outlined in 1.2 to clean the captchas that you want to classify.

#### 3.3 Sort captchas by character number
Parameters:
captcha-dir: string, location of captchas to be categorised by character length
font: string, name of the font being categorised

```
python3 sort_captchas_by_character.py --captcha-dir=location_of_captchas --font=name_of_font
```

#### 3.4 Captcha Classification
Parameters: 
min_captcha_length: int, the minimum number of characters in the given captchas
max_captcha_length: int, the maximum number of characters in the given captchas
model_path: string, the path to the trained model suited for this classification type (i.e for a given font and character range)
image_dir: string, path to the captchas to be classified
predictions_file: string, name and path to output .txt file

```
python3 new_classify.py --min_captcha_length=4 --max_captcha_length=6 --model_path=path/to/model.h5 --image_dir=path/to/images --predictions_file=V2/data/predictions/predictions_output
```

#### 3.5 Create Submission file
Parameters: 
dir: string, a directory containing all the output.txt files output from the classify code above.

```
python3 script_name.py --dir=V2/all_txt_files
```




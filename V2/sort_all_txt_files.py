import glob
import csv
import sys
import os
import argparse

# Take in directory conatining all the .txt files 
parser = argparse.ArgumentParser(description="Combine and sort .txt files into a single CSV file.")
parser.add_argument("--dir", required=True, help="Directory containing .txt files to be combined.")
args = parser.parse_args()

# Step 1: Define the list of .txt files from the specified directory
directory_path = args.dir
txt_files = glob.glob(os.path.join(directory_path, "*.txt"))

# Step 2: Create a new .csv file and write all lines from each .txt file into it
with open("combined_submission.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Loop through each .txt file
    for file in txt_files:
        with open(file, "r") as f:
            lines = f.readlines()
            # Write each line as a row in the CSV
            for line in lines:
                writer.writerow([line.strip()])

# Step 3: Sort the contents of the combined.csv file in 0-9, a-f order
with open("combined_submission.csv", "r") as csvfile:
    lines = csvfile.readlines()

# Sorting lines in 0-9, a-f order
sorted_lines = sorted(lines, key=lambda x: x.strip().lower())

# Write the sorted lines back to the csv file
with open("combined_submission.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    for line in sorted_lines:
        writer.writerow([line.strip()])

print("Combined and sorted .csv file created as 'combined_submission.csv'.")

import os
import json

# Define paths
base_folder = "data/roofNTNU/"
source_folder = os.path.join(base_folder, "roof_clouds_normed/")
output_folder = os.path.join(base_folder, "train_test_split/")
train_json = os.path.join(output_folder, "shuffled_train_file_list.json")
test_json = os.path.join(output_folder, "shuffled_test_file_list.json")

# Create the required folder structure
os.makedirs(os.path.join(output_folder, "points_train_n"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "labels_train_n"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "points_test_n"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "labels_test_n"), exist_ok=True)

def process_files(file_list, points_folder, labels_folder):
    """
    Process the files to extract points and labels and save them to the respective folders.
    """
    for file_name in file_list:
        input_file = os.path.join(source_folder, file_name+'.txt')
        
        # Output file paths
        points_file = os.path.join(points_folder, file_name+'.txt')
        labels_file = os.path.join(labels_folder, file_name+'.txt')
        
        with open(input_file, 'r') as infile, \
             open(points_file, 'w') as points_out, \
             open(labels_file, 'w') as labels_out:
            
            for line in infile:
                values = line.strip().split()
                if len(values) >= 4:  # Ensure the line has enough values
                    # Write the first 3 values to the points file
                    points_out.write(" ".join(values[:3]) + "\n")
                    # Write the second last value to the labels file
                    label = int(values[-2]) + 1
                    labels_out.write(str(label) + "\n")

# Load the JSON files
with open(train_json, 'r') as f:
    train_files = json.load(f)

with open(test_json, 'r') as f:
    test_files = json.load(f)

# Process training files
process_files(train_files, 
              os.path.join(output_folder, "points_train_n"), 
              os.path.join(output_folder, "labels_train_n"))

# Process testing files
process_files(test_files, 
              os.path.join(output_folder, "points_test_n"), 
              os.path.join(output_folder, "labels_test_n"))

print("Processing complete. Files saved in the respective folders.")
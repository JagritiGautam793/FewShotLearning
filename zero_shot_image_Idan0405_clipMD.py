import os
import torch
import csv
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set up the pipeline
pipe = pipeline("zero-shot-image-classification", model="Idan0405/ClipMD", trust_remote_code=True, device="cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {pipe.device}")

# Define the candidate labels corresponding to the MRI classes
candidate_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Define the path to your dataset
dataset_path = "alzheimer_mri/Dataset"

# Prepare to store results
results = []
y_true = []
y_pred = []



# Loop through the subdirectories and images
for class_dir in tqdm(os.listdir(dataset_path), desc="Processing directories"):
    class_path = os.path.join(dataset_path, class_dir)

    # Skip if it's not a directory
    if not os.path.isdir(class_path):
        continue

    # Loop through each image in the class directory
    for filename in tqdm(os.listdir(class_path), desc=f"Processing files in {class_dir}", leave=False):
        file_path = os.path.join(class_path, filename)

        # Process only image files
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Adjust based on your image formats
            try:
                # Load the image
                image = Image.open(file_path).convert("RGB")

                # Perform inference using the pipeline
                predictions = pipe(image, candidate_labels=candidate_labels)

                # Get the best prediction
                best_prediction = predictions[0]
                best_label = best_prediction['label']
                best_score = best_prediction['score']

                results.append({"filename": filename, "true_label": class_dir, "predicted_label": best_label, "score": best_score})
                y_true.append(class_dir)
                y_pred.append(best_label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Save results to a CSV file
output_file = "results_clip_md_optimized.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["filename", "true_label", "predicted_label", "score"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Load the CSV file into a DataFrame
df = pd.read_csv(output_file)

df['predicted_label'] = df['predicted_label'].str.replace(' ', '_')


# Extract the true labels and predicted labels
true_labels = df['true_label'].str.strip()
predicted_labels = df['predicted_label'].str.strip()

# Print unique values in both columns to check for mismatches
print("Unique values in true labels:", true_labels.unique())
print("Unique values in predicted labels:", predicted_labels.unique())

# Calculate the accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate the F1 score (weighted average of F1 score to handle multi-class)
f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

# Calculate precision (weighted)
precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)

# Calculate recall (weighted)
recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)

# Display the results
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

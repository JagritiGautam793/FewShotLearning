import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import csv
from tqdm import tqdm
from transformers import ViTForImageClassification

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

huggingface_token = "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"

# Load the fine-tuned ViT model
model_name = "fawadkhan/ViT_FineTuned_on_ImagesOASIS"
model = ViTForImageClassification.from_pretrained(model_name, use_auth_token=huggingface_token).to(device)

# Define the candidate labels corresponding to the MRI classes
candidate_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# Define the path to your dataset
dataset_path = "alzheimer_mri/Dataset"

# Define your transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the size the model expects
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

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
                # Load and preprocess the image
                image = Image.open(file_path).convert("RGB")
                inputs = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

                # Perform inference
                with torch.no_grad():
                    outputs = model(inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1)

                # Get the predicted class
                predicted_class_idx = probs.argmax().item()
                predicted_label = candidate_labels[predicted_class_idx]
                confidence_score = probs[0][predicted_class_idx].item()

                results.append({"filename": filename, "true_label": class_dir, "predicted_label": predicted_label, "score": confidence_score})
                y_true.append(class_dir)
                y_pred.append(predicted_label)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Save results to a CSV file
output_file = "results_vit_oasis.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["filename", "true_label", "predicted_label", "score"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")

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

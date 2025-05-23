import torch
from torchvision import transforms
from PIL import Image
import os
import csv
from tqdm import tqdm
from timm import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Define the candidate labels corresponding to the MRI classes
candidate_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]
# Load a pre-trained Vision Transformer model
model = create_model('vit_base_patch16_224', pretrained=True, num_classes=len(candidate_labels))
model = model.to(device)
model.eval()  # Set the model to evaluation mode



# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to match the input size of the model
    transforms.ToTensor(),            # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Define the path to your dataset
dataset_path = "alzheimer_mri/Dataset"

# Prepare to store results
results = []
y_true = []
y_pred = []

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    print(f"Dataset path '{dataset_path}' does not exist.")
else:
    print(f"Found dataset path: '{dataset_path}'")

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
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on your image formats
                try:
                    # Load and preprocess the image
                    image = Image.open(file_path).convert("RGB")  # Ensure image is in RGB format
                    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

                    # Perform inference
                    with torch.no_grad():
                        logits = model(image)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()

                    # Store the best prediction
                    best_idx = probs.argmax()
                    best_label = candidate_labels[best_idx]
                    best_score = probs[0][best_idx]

                    results.append({"filename": filename, "true_label": class_dir, "predicted_label": best_label, "score": best_score})
                    y_true.append(class_dir)
                    y_pred.append(best_label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")


    # Save results to a CSV file
output_file = "resultsVit.csv"
with open(output_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["filename", "true_label", "predicted_label", "score"])
    writer.writeheader()
    writer.writerows(results)

print(f"Results saved to {output_file}")

# Compute accuracy and other metrics
if len(y_true)==len(y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    # Save evaluation metrics to a file
    metrics_file = "evaluation_metrics_vIt.txt"
    with open(metrics_file, mode="w") as metrics:
        metrics.write(f"Accuracy: {accuracy:.4f}\n")
        metrics.write(f"Precision: {precision:.4f}\n")
        metrics.write(f"Recall: {recall:.4f}\n")
        metrics.write(f"F1 Score: {f1:.4f}\n")

    print(f"Evaluation metrics saved to {metrics_file}")

    # Also print metrics to the console
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
else:
    print("No true or predicted labels found for evaluation.")



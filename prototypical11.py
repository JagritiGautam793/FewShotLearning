import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import pandas as pd  

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):  
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Organize samples by class
        self.samples_by_class = {cls: [] for cls in self.classes}
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples_by_class[class_name].append(img_path)

    def get_random_samples(self, n_way, n_support, n_query):
        """Get random samples for few-shot task."""
        # Randomly select n_way classes
        selected_classes = random.sample(self.classes, n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        # For each selected class
        for label, class_name in enumerate(selected_classes):
            # Get all samples for this class
            class_samples = self.samples_by_class[class_name]
            # Randomly select support and query samples
            selected_samples = random.sample(class_samples, n_support + n_query)
            
            # Process support samples
            for path in selected_samples[:n_support]:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                support_images.append(img)
                support_labels.append(label)
            
            # Process query samples
            for path in selected_samples[n_support:]:
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                query_images.append(img)
                query_labels.append(label)
        
        # Convert to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels

class PrototypicalNetwork(nn.Module):
    def __init__(self):  
        super(PrototypicalNetwork, self).__init__()
        # Load pretrained ResNet18 and modify for medical images
        resnet = resnet18(pretrained=True)
        # Modify first conv layer for medical images
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Add final embedding layer
        self.embedding = nn.Linear(512, 64)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

    def compute_prototypes(self, support_images, support_labels):
        """Compute class prototypes from support set."""
        # Get embeddings for support set
        z_support = self.forward(support_images)
        n_way = len(torch.unique(support_labels))
        
        # Compute prototypes
        prototypes = torch.zeros(n_way, z_support.size(-1)).to(z_support.device)
        for i in range(n_way):
            mask = support_labels == i
            prototypes[i] = z_support[mask].mean(0)
        
        return prototypes

    def compute_loss(self, prototypes, query_embeddings, query_labels):
        """Compute prototypical networks loss."""
        # Compute distances to prototypes
        dists = torch.cdist(query_embeddings, prototypes)
        
        # Convert distances to probabilities
        log_p_y = F.log_softmax(-dists, dim=1)
        
        # Compute cross entropy loss
        loss = F.nll_loss(log_p_y, query_labels)
        
        # Compute accuracy
        _, predictions = log_p_y.max(1)
        acc = torch.mean((predictions == query_labels).float())
        
        return loss, acc

def train_episode(model, optimizer, support_images, support_labels, query_images, query_labels):
    """Train model on a single episode."""
    model.train()
    optimizer.zero_grad()
    
    # Compute prototypes from support set
    prototypes = model.compute_prototypes(support_images, support_labels)
    
    # Get embeddings for query set
    query_embeddings = model(query_images)
    
    # Compute loss and accuracy
    loss, acc = model.compute_loss(prototypes, query_embeddings, query_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item(), acc.item()

def evaluate(model, dataset, device, n_episodes=100, n_way=4, n_support=5, n_query=15):
    """Evaluate model on multiple episodes."""
    model.eval()
    all_accuracies = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            # Get random episode
            support_images, support_labels, query_images, query_labels = \
                dataset.get_random_samples(n_way, n_support, n_query)
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Compute prototypes
            prototypes = model.compute_prototypes(support_images, support_labels)
            
            # Get query embeddings
            query_embeddings = model(query_images)
            
            # Compute distances and get predictions
            dists = torch.cdist(query_embeddings, prototypes)
            _, predictions = (-dists).max(1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
            all_accuracies.append((predictions == query_labels).float().mean().item())
    
    return np.mean(all_accuracies), all_predictions, all_labels

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parameters
    n_way = 4  # Number of classes per episode
    n_support = 5  # Number of support examples per class
    n_query = 15  # Number of query examples per class
    n_episodes = 1000  # Number of episodes for training
    
    # Transform for medical images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = MedicalImageDataset("../alzheimer_mri/Dataset", transform=transform)
    
    # Create model and optimizer
    model = PrototypicalNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_acc = 0.0
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Get random episode
        support_images, support_labels, query_images, query_labels = \
            dataset.get_random_samples(n_way, n_support, n_query)
        
        # Move to device
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)
        
        # Train on episode
        loss, acc = train_episode(model, optimizer, 
                                    support_images, support_labels,
                                    query_images, query_labels)
        
        if (episode + 1) % 100 == 0:
            print(f"\nEpisode {episode+1}/{n_episodes}")
            print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    acc, predictions, labels = evaluate(model, dataset, device)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # Save metrics
    metrics_file = "evaluation_metrics.txt"
    with open(metrics_file, mode="w") as metrics:
        metrics.write(f"Final Accuracy: {accuracy:.4f}\n")
        metrics.write(f"Final Precision: {precision:.4f}\n")
        metrics.write(f"Final Recall: {recall:.4f}\n")
        metrics.write(f"Final F1 Score: {f1:.4f}\n")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'medical_proto_net2.pth')

    # ---------------------- Save test results to CSV ---------------------- #
    
    # Convert predictions and labels back to class names
    predicted_classes = [dataset.classes[pred] for pred in predictions]
    real_classes = [dataset.classes[label] for label in labels]

    # Create a DataFrame for the results
    results_df = pd.DataFrame({
        'Real Class': real_classes,
        'Predicted Class': predicted_classes
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv('test_results_proto11.csv', index=False)

if __name__ == "__main__":
    main()

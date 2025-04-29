# Alzheimer's MRI Classification using Few-Shot and Zero-Shot Learning

This project implements multiple approaches for classifying Alzheimer's MRI scans using both few-shot learning and zero-shot learning techniques. The project includes three different implementation approaches to tackle the classification problem.

## Project Overview

The project contains three main implementations:

1. **Few-Shot Learning using Prototypical Networks**

   - Uses ResNet18 as backbone
   - Implements episodic training
   - Supports N-way, K-shot learning

2. **Zero-Shot Learning using CLIP-MD**

   - Utilizes Idan0405/ClipMD model
   - Performs direct zero-shot classification

3. **Zero-Shot Learning using ViT with OASIS**
   - Uses fine-tuned Vision Transformer model
   - Trained on OASIS dataset

## Requirements

```
torch
torchvision
transformers
pillow
numpy
pandas
scikit-learn
tqdm
```

## Dataset Structure

The project expects an Alzheimer's MRI dataset with the following structure:

```
alzheimer_mri/Dataset/
    ├── Mild_Demented/
    ├── Moderate_Demented/
    ├── Non_Demented/
    └── Very_Mild_Demented/
```

## Implementation Details

### 1. Few-Shot Learning (FewshotLearning.py)

- Implements Prototypical Networks for few-shot learning
- Uses pre-trained ResNet18 as feature extractor
- Includes custom MedicalImageDataset class
- Performs episodic training
- Saves model weights and evaluation metrics

### 2. Zero-Shot CLIP-MD (zero_shot_image_Idan0405_clipMD.py)

- Leverages pre-trained CLIP-MD model
- Performs zero-shot classification
- Generates detailed classification metrics

### 3. Zero-Shot ViT-OASIS (zero_shot_image_OASIS.py)

- Uses ViT model fine-tuned on OASIS dataset
- Requires Hugging Face authentication
- Provides comprehensive evaluation metrics

## Output Files

The project generates several output files:

- `medical_proto_net2.pth`: Trained model weights for few-shot learning
- `evaluation_metrics.txt`: Performance metrics for few-shot learning
- `test_results_proto11.csv`: Detailed results from few-shot learning
- `results_clip_md_optimized.csv`: Results from CLIP-MD classification
- `results_vit_oasis.csv`: Results from ViT-OASIS classification

## Usage

1. Few-Shot Learning:

```bash
python FewshotLearning.py
```

2. Zero-Shot with CLIP-MD:

```bash
python zero_shot_image_Idan0405_clipMD.py
```

3. Zero-Shot with ViT-OASIS:

```bash
python zero_shot_image_OASIS.py
```

## Evaluation Metrics

Each approach provides the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

Results are saved in CSV format for detailed analysis.

## Note on Model Selection

- Use Few-Shot Learning when you have limited labeled data and need to learn from few examples
- Use Zero-Shot CLIP-MD when you want to leverage medical domain knowledge without training
- Use Zero-Shot ViT-OASIS when you want to benefit from OASIS dataset pre-training

## Contributing

Feel free to submit issues and enhancement requests!

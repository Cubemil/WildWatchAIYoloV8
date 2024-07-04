import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, confusion_matrix, ConfusionMatrixDisplay

# Paths to the dataset folders and model
test_images_directory_path = "./content/datasets/animalDataset/test/images"
test_annotations_directory_path = "./content/datasets/animalDataset/test/labels"
model_path = "./models/wildwatch_yolov8_01.pt"

# Load the model
model = YOLO(model_path)

# Function to read ground truth labels
def parse_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            labels.append(class_id)
    return labels

# Initialize lists to hold true and predicted labels
true_labels = []
predicted_labels = []

# Iterate over test images and make predictions
for image_name in os.listdir(test_images_directory_path):
    if not image_name.endswith('.jpg'):
        continue  # skip non-image files if any
    image_path = os.path.join(test_images_directory_path, image_name)
    label_path = os.path.join(test_annotations_directory_path, image_name.replace('.jpg', '.txt'))
    
    # Read ground truth labels
    if os.path.exists(label_path):
        true_labels.extend(parse_labels(label_path))
    
    # Make prediction
    results = model(image_path)
    predictions = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else np.array([-1])  # Extract predicted class IDs or append -1 if no detections
    
    # Add predicted labels to the list
    predicted_labels.extend(predictions)

# Filter out -1 values from true_labels and predicted_labels to align their lengths
filtered_true_labels = [label for label in true_labels if label != -1]
filtered_predicted_labels = [label for label in predicted_labels if label != -1]

# Compute confusion matrix
cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels, labels=list(range(len(model.names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.names)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close(fig)

# Compute and plot normalized confusion matrix
cm_normalized = confusion_matrix(filtered_true_labels, filtered_predicted_labels, labels=list(range(len(model.names))), normalize='true')
disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=model.names)

fig, ax = plt.subplots(figsize=(10, 10))
disp_normalized.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Normalized Confusion Matrix')
plt.savefig('normalized_confusion_matrix.png')
plt.close(fig)

# Compute and plot precision for each class
precision = precision_score(filtered_true_labels, filtered_predicted_labels, labels=list(range(len(model.names))), average=None, zero_division=0)
precision_dict = {model.names[i]: precision[i] for i in range(len(precision))}

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(precision_dict.keys(), precision_dict.values())
plt.xlabel('Class')
plt.ylabel('Precision')
plt.title('Precision for Each Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('precision_per_class.png')
plt.close(fig)

print("Evaluation completed. Plots saved as 'confusion_matrix.png', 'normalized_confusion_matrix.png', and 'precision_per_class.png'.")

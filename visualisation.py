import supervision as sv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score

# Paths to the dataset folders and yaml file
train_images_directory_path = "./content/datasets/animalDataset/train/images"
train_annotations_directory_path = "./content/datasets/animalDataset/train/labels"
valid_images_directory_path = "./content/datasets/animalDataset/valid/images"
valid_annotations_directory_path = "./content/datasets/animalDataset/valid/labels"
test_images_directory_path = "./content/datasets/animalDataset/test/images"
test_annotations_directory_path = "./content/datasets/animalDataset/test/labels"
data_yaml_path = "./content/datasets/animalDataset/data.yaml"

# Create the test dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=test_images_directory_path, 
    annotations_directory_path=test_annotations_directory_path,
    data_yaml_path=data_yaml_path
)

# Load the model
model = YOLO("./models/wildwatchyolov8_v2_finetuning05.pt")

# Callback function for model predictions
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

# Evaluate and get the confusion matrix
confusion_matrix = sv.ConfusionMatrix.benchmark(
    dataset=dataset,
    callback=callback,
)

plt = confusion_matrix.plot(normalize=True)
plt.savefig('normalized_confusion_matrix.png')

"""
# Handle division by zero in normalization
row_sums = confusion_matrix.sum(axis=1)
normalized_cm = np.array([
    confusion_matrix[i] / row_sums[i] if row_sums[i] != 0 else np.zeros_like(confusion_matrix[i])
    for i in range(len(row_sums))
])

# Plot and save the normalized confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(normalized_cm, cmap='Blues')
fig.colorbar(cax)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.savefig('normalized_confusion_matrix.png')
plt.close(fig)

# Inspect the structure of the dataset
print("Inspecting dataset structure:")
for data in dataset:
    print(data)
    break  # Print only the first element to see its structure

# Extract true labels and predicted labels
true_labels = []
predicted_labels = []
for data in dataset:
    image, labels = data  # Adjust indexing based on the structure
    detections = callback(image)
    true_labels.extend(labels)
    predicted_labels.extend(detections.labels)

# Calculate precision for each class
precision = precision_score(true_labels, predicted_labels, average=None)
precision_dict = {f'Class {i}': p for i, p in enumerate(precision)}

# Save precision values as a bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(precision_dict.keys(), precision_dict.values())
plt.xlabel('Class')
plt.ylabel('Precision')
plt.title('Precision for Each Class')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('precision_per_class.png')
plt.close(fig)

print("Normalized confusion matrix and precision per class have been saved as PNG files.")

"""
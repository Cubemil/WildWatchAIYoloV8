import os
import yaml
from PIL import Image
import numpy as np

def load_test_data(yaml_path):
    """
    Load test data based on annotations from a YAML file.
    
    Args:
    - yaml_path (str): Path to the YAML file with annotations.
    
    Returns:
    - images (list of np.array): Loaded images.
    - labels (list): Corresponding labels of the images.
    """
    # Load YAML file
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    print(f"YAML content: {data}")
    
    test_images_dir = os.path.join(data['path'], data['test'])
    images = []
    labels = []
    
    # Read images and labels
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        images.append(img)
        
        # Assuming labels are in the same folder with same name but .txt extension
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        with open(label_path, 'r') as label_file:
            label_data = label_file.read().strip()
            label = int(label_data)  # Adjust according to your label format
            labels.append(label)
    
    return images, labels

# Adjust the paths as necessary
yaml_path = './content/datasets/animalDataset/data.yaml'

from ultralytics import YOLO
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

# Load our custom model
model = YOLO("./models/wildwatch_yolov8_01.pt")

# Load test dataset
test_images, test_labels = load_test_data(yaml_path)

# Make predictions on the test set
results = model.predict(test_images)

# Print model outputs for inspection
print(f"Model output: {results}")

# Assume results contain list of dictionaries with 'labels' and 'predictions' keys
# Adjust according to your actual model's output format
y_true = test_labels
y_pred = [result[0]['class'] for result in results]  # Adjust according to your result structure

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
map_score = average_precision_score(y_true, y_pred)

# Create a DataFrame for easy display and further processing
metrics_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'mAP'],
    'Score': [precision, recall, f1, map_score]
})

# Visualize metrics with matplotlib
import matplotlib.pyplot as plt

def plot_metrics(metrics_df):
    fig, ax = plt.subplots()
    ax.bar(metrics_df['Metric'], metrics_df['Score'], color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax.set_title('Model Evaluation Metrics')
    ax.set_ylabel('Score')
    plt.ylim(0, 1)  # Assuming scores are between 0 and 1
    plt.show()

plot_metrics(metrics_df)

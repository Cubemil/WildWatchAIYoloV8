import streamlit as st
import supervision as sv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from PIL import Image
import io

# Paths to the dataset folders and yaml file
test_images_directory_path = "./content/datasets/animalDataset/test/images"
test_annotations_directory_path = "./content/datasets/animalDataset/test/labels"
data_yaml_path = "./content/datasets/animalDataset/data.yaml"

# Load the model
model = YOLO("./models/wildwatchyolov8_v2_finetuning05.pt")

# Callback function for model predictions
def callback(image: np.ndarray) -> sv.Detections:
    result = model(image)[0]
    return sv.Detections.from_ultralytics(result)

# Streamlit app setup
st.title("WildWatchAI's fine-tuned YOLOv8 model")
st.sidebar.title("Navigation")
options = ["Home", "Evaluation", "Visualizations"]
choice = st.sidebar.radio("Go to", options)

if choice == "Home":
    st.subheader("Upload an image to detect animals")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Running inference...")
        results = callback(image)
        st.image(results.img, caption='Detected Animals', use_column_width=True)

elif choice == "Evaluation":
    st.subheader("Model Evaluation Metrics")
    st.write("Computing confusion matrix and precision scores...")

    # Create the test dataset
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=test_images_directory_path, 
        annotations_directory_path=test_annotations_directory_path,
        data_yaml_path=data_yaml_path
    )

    # Evaluate and get the confusion matrix
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset,
        callback=callback,
    )

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 10))
    confusion_matrix.plot(normalize=True)
    st.pyplot(plt.gcf())

    # Extract true labels and predicted labels
    true_labels = []
    predicted_labels = []
    for data in dataset:
        image, labels = data.image, data.labels  # Adjust indexing based on the structure
        detections = callback(image)
        true_labels.extend(labels)
        predicted_labels.extend(detections.labels)

    # Calculate precision for each class
    precision = precision_score(true_labels, predicted_labels, average=None)
    precision_dict = {f'Class {i}': p for i, p in enumerate(precision)}

    # Display precision values as a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(precision_dict.keys(), precision_dict.values())
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision for Each Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif choice == "Visualizations":
    st.subheader("Model Visualizations")
    st.write("Visualizations go here")

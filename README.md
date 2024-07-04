# YOLOv8 WildWatchAI

## Introduction

This repository contains the code and documentation for fine-training the YOLOv8 model on our custom dataset.
This project demonstrates the process of annotating a custom dataset, training the YOLOv8 model, evaluating its performance, and deploying the trained model in a Streamlit application.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset consists of self-annotated images of various animals. The images were annotated by handwith bounding boxes. The images were augmented and split into Train/Validation/Test subsets and evenly distributed by RoboFlow's web interface. The annotations were exported in the YOLO format, which includes the class label and the bounding box coordinates.

## Model Training

1. **Prepared the Dataset**: annotation, augmentation, polishing, splitting and exporting
2. **Installed Dependencies in our virtual environment**: installed the necessary libraries and dependencies, including PyTorch, YOLOv8, ultralytics, streamlit etc.
3. **Configured model**: Set up the yolov8 configuration file with the appropriate parameters for our custom dataset, such as the number of classes, input image size, and training hyperparameters. (Did this in the web GUI on the final runs)
4. **Train the Model**: Ran training script (or web training with google collab) to train the model. Epochs: ~500

```python
from yolov8 import YOLO

# load model
model = YOLO('./models/wildwatch_yolov8_X.pt') # X => model version

# train model on dataset
model.train(data='./content/datasets/wildAnimals', epochs=50, batch_size=16, img_size=640)
```

## Evaluation

After training the model, we used confusion matrices to visualize the performance of different versions of the model.

1. **Generated Predictions**: Ran the model on the test set to generate predictions.
2. **Computed Metrics**: Calculated evaluation metrics such as precision, recall, and F1-score. (sklearn)
3. **Visualized Confusion Matrix**: Created confusion matrices to visualize the performance of the model. (normalised and by actual TP/FP/TN/FN-counts)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

predictions = model.predict('./content/datasets/wildAnimals/test/X.png')

cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```

## Deployment

The trained model is deployed in a Streamlit application for real-time animal detection. 

1. **Set Up Streamlit App**: Created a Streamlit app to load the trained model and provide an interface for users to upload and predict images.
2. **Loaded and Ran Model**: Integrated the model to run inference on the uploaded images and display the results.

```python
import streamlit as st
from yolov8 import YOLO

model = YOLO('path/to/trained/model')

st.title('Animal Detection with YOLOv8')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    
    results = model.predict(image)
    
    st.image(results.img, caption='Detected Animals', use_column_width=True)
```

## Getting Started

To get started with this project, clone the repository and follow the instructions below:

1. **Clone the Repository**:

    ```sh
    git clone https://github.com/your-username/yolov8-animal-detection.git
    cd yolov8-animal-detection
    ```

2. **Set up a virtual environment**
    - *Our method:* Create in VSCode and activate by running .venv\Scripts\activate

3. **Install Dependencies**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Train the Model**: Follow the [Model Training](#model-training) section to train the YOLOv8 model on your custom dataset.

5. **Evaluate the Model**: Use the [Evaluation](#evaluation) section to evaluate the performance of the trained model.

6. **Run the Streamlit App**:

    ```sh
    streamlit run app.py
    ```
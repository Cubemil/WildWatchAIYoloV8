import streamlit as st
from utils.load_model import load_yolo_model
from utils.inference import run_inference
from utils.visualization_helpers import display_confusion_matrix

model_path = "../../models/wildwatchyolov8_v2_finetuning05.pt"

st.title("WildWatchAI's fine-tuned yolov8 model")

st.sidebar.title("Navigation")
options = ["Home", "Evaluation", "Visualizations"]
choice = st.sidebar.radio("Go to", options)

if choice == "Home":
    st.subheader("Upload an image to detect animals")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Running inference...")
        model = load_yolo_model(model_path)
        results = run_inference(model, image)
        st.image(results.img, caption='Detected Animals', use_column_width=True)

elif choice == "Evaluation":
    st.subheader("Model Evaluation Metrics")
    # Add code to display evaluation metrics

elif choice == "Visualizations":
    st.subheader("Model Visualizations")
    display_confusion_matrix()

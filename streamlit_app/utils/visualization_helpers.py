import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def display_confusion_matrix():
    # Assuming confusion matrix data is loaded
    cm = ...  # load your confusion matrix data
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    st.pyplot(plt.gcf())

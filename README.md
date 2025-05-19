#🌿 Plant-disease-diagnonis

A Streamlit-based web application that classifies plant leaf diseases using deep learning and explains predictions using XAI techniques like LIME and Grad-CAM. It also segments the leaf to assess severity levels of infections.

---

## 📂 Dataset

This project uses the open-source [PlantVillage dataset](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) from Kaggle, which contains 38 labelled of healthy and diseased plant leaves.

---

## 🔍 Project Highlights

- 🌱 **Leaf Disease Classification** using **InceptionV3** (Transfer Learning)
- 🧠 **Explainable AI (XAI)** techniques:
  - **LIME**: Highlights key superpixels that influenced the prediction
  - **Grad-CAM**: Generates a heatmap over disease-affected regions
- 🎯 **Segmentation** using **Parametric Segmentation** to isolate the leaf and analyze infection severity
- **Treatment Recommendation** using **Disease Ontology Graph** based on the severity level of the plant
- **Disease Spread prediction** using **weekly based time series data**
- **AI Support** using **LLM API key** which is used to clarify farmers queries
- **Text-to-Speech** which is helpful to understand what disease affected the plant by hearing voice instead of reading all informations
- 📊 **Interactive Web Interface** built with **Streamlit**


---

## 🎥 Demo Video

📽️ [Click here to watch the demo](https://github.com/RAVEENRAJC/Plant-disease-diagnonis/blob/main/Plant%20Disease%20Diagnosis%20testing%20-.mp4)
---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- LIME
- Grad-CAM (via custom backprop)
- Streamlit
- NumPy, Matplotlib, PIL

---

## 🧪 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/RAVEENRAJC/Plant-disease-diagnosis.git
   cd Plant-disease-diagnosis

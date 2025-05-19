import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
import pandas as pd
import networkx as nx
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import requests
import tempfile
from matplotlib.figure import Figure
from io import BytesIO

# Create a placeholder for temporary files
if not os.path.exists("temp_report_images"):
    os.makedirs("temp_report_images")

# Set page configuration
st.set_page_config(page_title="Plant Disease Diagnosis", layout="wide")

# ------------------------------
# GLOBAL SETTINGS & PARAMETERS
# ------------------------------
IMG_WIDTH, IMG_HEIGHT = 299, 299   # Inception V3 input dimensions

# Session state initialization for global variables
if 'results' not in st.session_state:
    st.session_state.results = {}

# ------------------------------
# UTILITY FUNCTIONS (COMMON)
# ------------------------------
@st.cache_resource
def load_model(model_path):
    """Load TensorFlow model with caching"""
    return tf.keras.models.load_model(model_path, compile=False)

def get_class_names_from_directory(directory_path):
    """Extract class names from directory"""
    classes = sorted([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
    class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
    return {v: k for k, v in class_indices.items()}

def preprocess_image(image_file):
    """Preprocess image for model inference"""
    # Handle uploaded files or paths
    if isinstance(image_file, str):
        img = load_img(image_file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    else:
        # For uploaded files
        img = load_img(image_file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def lime_predict_function(images, model_instance):
    """Function for LIME explainer"""
    img_batch = images.astype("float32") / 255.0
    return model_instance.predict(img_batch)

def generate_gradcam_heatmap(model, img_array, class_index, last_conv_layer_name='mixed10'):
    """Generate Grad-CAM heatmap"""
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))

def segment_leaf(image_file):
    """Segment leaf from background"""
    # Handle uploaded files or paths
    if isinstance(image_file, str):
        img = cv2.imread(image_file)
    else:
        # For uploaded files
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(image_file.read())
        temp_file.close()
        img = cv2.imread(temp_file.name)
        os.unlink(temp_file.name)
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range for green leaf segmentation
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented_img = cv2.bitwise_and(img, img, mask=mask)
    return mask, segmented_img, img

# ------------------------------
# FAITHFULNESS SCORE FUNCTION (Grad-CAM and LIME)
# ------------------------------
def mask_important_regions(image, mask, mode='zero'):
    """Mask important regions identified by XAI methods"""
    if mode == 'zero':
        masked_image = image.copy()
        masked_image[mask > 0] = 0
    elif mode == 'blur':
        masked_image = cv2.GaussianBlur(image, (5, 5), 0)
    return masked_image

def compute_faithfulness(image_file, model):
    """Compute faithfulness scores for XAI methods"""
    img_array = preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    original_confidence = predictions[0][predicted_class]
    
    # Grad-CAM faithfulness
    heatmap = generate_gradcam_heatmap(model, img_array, predicted_class)
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    _, gradcam_mask = cv2.threshold(heatmap_uint8, 150, 255, cv2.THRESH_BINARY)
    masked_img = mask_important_regions(np.uint8(img_array[0] * 255), gradcam_mask)
    masked_img_array = np.expand_dims(masked_img / 255.0, axis=0)
    masked_confidence = model.predict(masked_img_array)[0][predicted_class]
    gradcam_faithfulness = (original_confidence - masked_confidence) / original_confidence
    
    # LIME faithfulness
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.uint8(img_array[0] * 255),
                                         lambda x: model.predict(x.astype("float32") / 255.0),
                                         top_labels=1, hide_color=0, num_samples=500,
                                         segmentation_fn=lambda x: slic(x, n_segments=100, compactness=10))
    top_features = explanation.local_exp[predicted_class][:10]
    lime_mask = np.zeros((IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)
    for feature, weight in top_features:
        lime_mask[explanation.segments == feature] = 255
    masked_img_lime = mask_important_regions(np.uint8(img_array[0] * 255), lime_mask)
    masked_img_lime_array = np.expand_dims(masked_img_lime / 255.0, axis=0)
    masked_confidence_lime = model.predict(masked_img_lime_array)[0][predicted_class]
    lime_faithfulness = (original_confidence - masked_confidence_lime) / original_confidence
    
    return gradcam_faithfulness, lime_faithfulness

# ------------------------------
# SEVERITY ANALYSIS FUNCTIONS & CHARTS
# ------------------------------
def calculate_severity(leaf_mask, heatmap):
    """Calculate disease severity based on heatmap coverage of leaf area"""
    # Set threshold to detect affected pixels
    heatmap_threshold = 0.6
    
    # Create binary mask for affected areas
    heatmap_mask = (heatmap > heatmap_threshold).astype(np.uint8)
    
    # Calculate overlap of heatmap with leaf
    heatmap_coverage = np.sum(heatmap_mask * (leaf_mask > 0))
    
    # Calculate total leaf pixels
    leaf_coverage = np.sum(leaf_mask > 0)
    
    # Handle case with no leaf detected
    if leaf_coverage == 0:
        return "Unknown", 0.0

    # Calculate coverage ratio as percentage
    coverage_ratio = (heatmap_coverage / leaf_coverage) * 100

    # Determine severity category based on coverage ratio
    if coverage_ratio >= 50:
        severity = "Severe"
    elif 20 <= coverage_ratio < 50:
        severity = "Moderate"
    elif 0 < coverage_ratio < 20:
        severity = "Mild"
    else:
        severity = "Healthy"  # Added Healthy category for 0% coverage

    return severity, coverage_ratio

def create_severity_donut_chart(severity, coverage_percent):
    """Create a donut chart representing disease severity"""
    fig = Figure(figsize=(6, 6))
    ax = fig.subplots()
    
    # Set color and fill percentage based on severity
    if severity == "Healthy":
        color = "limegreen"
        fill_percent = 100
        inner_text = "Healthy 🌿"
    elif severity == "Mild":
        color = "yellow"
        fill_percent = 30
        inner_text = "Mild 🍃"
    elif severity == "Moderate":
        color = "orange"
        fill_percent = 65
        inner_text = "Moderate 🍂"
    elif severity == "Severe":
        color = "red"
        fill_percent = 90
        inner_text = "Severe 🔥"
    else:
        color = "gray"
        fill_percent = 10
        inner_text = "Unknown ❓"
    
    wedges, _ = ax.pie(
        [fill_percent, 100-fill_percent], 
        colors=[color, 'lightgray'],
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white')
    )
    
    ax.text(0, 0, inner_text, ha='center', va='center', fontsize=18, fontweight='bold')
    ax.set_title("Disease Severity Analysis", fontsize=16, pad=20)
    plt.close(fig)  # Close the figure to avoid displaying it twice
    
    # Save figure for PDF report
    fig.savefig("temp_report_images/severity_donut.png")
    
    return fig

# ------------------------------
# LIME XAI IMPLEMENTATION
# ------------------------------
def predict_and_display_with_lime(image_file, model):
    """Generate LIME explanation for model prediction"""
    img_array = preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # LIME explanation
    explainer = lime_image.LimeImageExplainer()
    resized_img = cv2.resize(np.uint8(img_array[0] * 255), (IMG_WIDTH, IMG_HEIGHT))
    
    # Create a wrapper function that includes the model
    def predict_fn(images):
        return lime_predict_function(images, model)
    
    explanation = explainer.explain_instance(
        resized_img,
        predict_fn,  # Use the wrapper function
        top_labels=3,
        hide_color=0,
        num_samples=500
    )
    
    top_labels = explanation.top_labels
    if predicted_class not in top_labels:
        st.warning(f"Predicted class {predicted_class} not in LIME's top labels. Using top label instead.")
        predicted_class = top_labels[0]

    lime_image_overlay, mask = explanation.get_image_and_mask(
        label=predicted_class,
        positive_only=True,
        hide_rest=False,
        num_features=10,
        min_weight=0.01
    )
    
    return lime_image_overlay, mask

# ------------------------------
# KNOWLEDGE GRAPH & DISEASE ONTOLOGY
# ------------------------------
def display_knowledge_graph(selected_plant, ontology_csv):
    """Display knowledge graph for selected plant"""
    # Load ontology data
    df = pd.read_csv(ontology_csv, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    plant_df = df[df["Plant Affected"].str.lower() == selected_plant.lower()]
    if plant_df.empty:
        st.warning(f"No disease data found for {selected_plant}.")
        return None
    
    G = nx.DiGraph()
    G.add_node(selected_plant)
    for _, row in plant_df.iterrows():
        disease = row["Disease"]
        category = row["Category"]
        symptom1 = row["Symptom1"]
        symptom2 = row["Symptom2"]
        remedy1 = row["Remedy1"]
        remedy2 = row["Remedy2"]
        cause = row["Cause"]
        preventive = row["Preventive Measures"]
        severity_level = row["Severity Level"]
        G.add_node(disease)
        G.add_edge(selected_plant, disease)
        if pd.notna(severity_level):
            G.add_node(severity_level)
            G.add_edge(disease, severity_level)
        for symptom in [symptom1, symptom2]:
            if pd.notna(symptom):
                G.add_node(symptom)
                G.add_edge(disease, symptom)
        if pd.notna(cause):
            G.add_node(cause)
            G.add_edge(disease, cause)
        for remedy in [remedy1, remedy2]:
            if pd.notna(remedy):
                G.add_node(remedy)
                G.add_edge(disease, remedy)
        if pd.notna(preventive):
            G.add_node(preventive)
            G.add_edge(disease, preventive)
    
    fig = Figure(figsize=(18, 14))
    ax = fig.subplots()
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=2500, font_size=10, font_weight="bold", arrowsize=10)
    ax.set_title("Disease Ontology and Knowledge Graph")
    
    # Save for PDF report
    fig.savefig("temp_report_images/ontology_graph.png")  
    plt.close(fig)
    
    return fig

# ------------------------------
# REMEDY RECOMMENDATION FUNCTIONS
# ------------------------------
def get_treatment_recommendation(disease, severity, treatment_excel):
    """Get treatment recommendation based on disease and severity"""
    try:
        treatment_df = pd.read_excel(treatment_excel, sheet_name="plant_disease_treatments(data)")
        treatment_row = treatment_df[(treatment_df['Disease'] == disease) & (treatment_df['Severity'] == severity)]
        if not treatment_row.empty:
            return treatment_row.iloc[0]['Treatment']
        else:
            return "No specific treatment found for this disease and severity level."
    except FileNotFoundError:
        return "Treatment database not found. Please check the path to 'plant_disease_treatments.xlsx'."
    except Exception as e:
        return f"Error retrieving treatment information: {str(e)}"

# ------------------------------
# DISEASE SPREAD PREDICTION USING LSTM
# ------------------------------
def predict_disease_spread(disease, severity, lstm_model_path, time_series_csv):
    """Predict disease spread using LSTM model"""
    lstm_model = tf.keras.models.load_model(lstm_model_path, compile=False)
    time_series_df = pd.read_csv(time_series_csv)
    
    if disease not in time_series_df['Disease'].unique():
        return "No spread data available"
    
    disease_data = time_series_df[time_series_df['Disease'] == disease]
    severity_index = {"Mild": 0, "Moderate": 1, "Severe": 2}
    
    if severity not in severity_index:
        return "Invalid severity level"
    
    start_index = severity_index[severity]
    if len(disease_data) < start_index + 10:
        st.warning(f"Not enough time-series data for {disease} with severity {severity}.")
        return "Insufficient data"
    
    input_sequence = disease_data.iloc[start_index:start_index+10, -2:].values
    input_sequence = np.expand_dims(input_sequence, axis=0)
    
    expected_features = lstm_model.input_shape[-1]
    if input_sequence.shape[-1] != expected_features:
        st.warning(f"Feature mismatch! Model expects {expected_features}, but got {input_sequence.shape[-1]}")
        return "Feature dimension mismatch"
    
    spread_prediction = lstm_model.predict(input_sequence)
    severity_growth, confidence = spread_prediction.flatten()
    
    # Display a pie chart for spread prediction score
    fig = Figure(figsize=(1.5, 1.5))
    ax = fig.subplots()
    sizes = [severity_growth, 5 - severity_growth]  # Assuming scale 0-5
    ax.pie(sizes, labels=["Predicted Spread", ""], colors=["purple", "lightgray"], autopct='%1.1f%%',textprops={'fontsize': 8})
    ax.set_title("Disease Spread Prediction")
    
    # Save for PDF report
    fig.savefig("temp_report_images/spread_prediction.png")
    plt.close(fig)
    
    return (severity_growth, confidence), fig

# ------------------------------
# PDF REPORT GENERATION
# ------------------------------
def generate_pdf_report(results, report_filename="Plant_Disease_Report.pdf"):
    """Generate PDF report with analysis results"""
    try:
        from fpdf import FPDF
        
        class PDFReport(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Plant Disease Diagnosis Report', ln=True, align='C')
                self.ln(10)
                
            def chapter_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.cell(0, 10, title, ln=True)
                self.ln(4)
                
            def chapter_body(self, body_text):
                self.set_font('Arial', '', 12)
                self.multi_cell(0, 10, body_text)
                self.ln()
                
            def add_image_on_new_page(self, image_path, title, w=160):
                self.add_page()
                self.chapter_title(title)
                if os.path.exists(image_path):
                    # Center the image
                    img_x = (210 - w) / 2  # 210 is A4 width in mm
                    self.image(image_path, x=img_x, w=w)
                else:
                    self.cell(0, 10, "Image not available", ln=True)
                self.ln()
        
        # Make sure the temp folder exists
        temp_folder = "temp_report_images"
        os.makedirs(temp_folder, exist_ok=True)
        
        # Initialize the PDF
        pdf = PDFReport()
        pdf.add_page()
        
        # Add diagnosis summary
        pdf.chapter_title("Diagnosis Summary")
        summary_text = f"""
        Predicted Plant Disease: {results.get('predicted_class', 'Unknown')}
        Disease Severity: {results.get('severity', 'Unknown')} (Coverage: {results.get('coverage_ratio', 0):.1f}%)
        Recommended Treatment: {results.get('treatment', 'No treatment information available')}
        """
        if isinstance(results.get('spread_prediction'), tuple):
            severity_growth, confidence = results['spread_prediction'][0]
            summary_text += f"\nPredicted Spread Severity: {severity_growth:.2f} (Scale 0-5) with {confidence*100:.2f}% confidence"
        pdf.chapter_body(summary_text)
        
        # XAI Faithfulness Scores on first page
        pdf.chapter_title("XAI Faithfulness Scores")
        faithfulness_text = f"""
        Grad-CAM Faithfulness Score: {results.get('gradcam_faith', 0):.4f}
        LIME Faithfulness Score: {results.get('lime_faith', 0):.4f}
        
        These scores indicate how well the explainability methods identify the important regions in the image for classification.
        Higher score means better explanation quality.
        """
        pdf.chapter_body(faithfulness_text)
        
        # Add each image on new pages
        
        # Original image on new page
        orig_img_path = os.path.join(temp_folder, "original.png")
        if os.path.exists(orig_img_path):
            pdf.add_image_on_new_page(orig_img_path, "Plant Leaf Image")
        
        # Segmented image on new page
        seg_img_path = os.path.join(temp_folder, "segmented.png")
        if os.path.exists(seg_img_path):
            pdf.add_image_on_new_page(seg_img_path, "Segmented Leaf")
        
        # Grad-CAM heatmap on new page
        heatmap_path = os.path.join(temp_folder, "heatmap.png")
        if os.path.exists(heatmap_path):
            pdf.add_image_on_new_page(heatmap_path, "Grad-CAM Heatmap")
        
        # LIME overlay on new page
        lime_path = os.path.join(temp_folder, "lime_overlay.png")
        if os.path.exists(lime_path):
            pdf.add_image_on_new_page(lime_path, "LIME Explanation")
        
        # Severity Donut Chart on new page
        donut_path = os.path.join(temp_folder, "severity_donut.png")
        if os.path.exists(donut_path):
            pdf.add_image_on_new_page(donut_path, "Disease Severity Analysis")
        
        # Knowledge Graph on new page (larger size)
        graph_path = os.path.join(temp_folder, "ontology_graph.png")
        if os.path.exists(graph_path):
            pdf.add_image_on_new_page(graph_path, "Disease Ontology and Knowledge Graph", w=180)
        
        # Spread Prediction on new page
        spread_path = os.path.join(temp_folder, "spread_prediction.png")
        if os.path.exists(spread_path):
            pdf.add_image_on_new_page(spread_path, "Disease Spread Prediction")
        
        # Output the PDF
        pdf.output(report_filename)
        return True
    except ImportError:
        st.error("FPDF module not found. PDF report generation will be skipped.")
        return False
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return False
# ------------------------------
# TEXT TO SPEECH AND TRANSLATION MODULE
# ------------------------------
def translate_text(text, target_language='ta'):
    try:
        from googletrans import Translator
        translator = Translator()
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.warning(f"Translation failed: {str(e)}")
        return text

def text_to_speech(text, language='en', filename="treatment_audio.mp3"):
    try:
        from gtts import gTTS
        os.makedirs("temp_report_images", exist_ok=True)
        audio_path = os.path.join("temp_report_images", filename)

        if not text.strip():
            st.warning("No text provided for speech synthesis.")
            return None

        tts = gTTS(text=text, lang=language, slow=False, tld='co.in' if language == 'ta' else 'com')
        tts.save(audio_path)
        return audio_path

    except Exception as e:
        st.error(f"TTS failed: {str(e)}")
        # Fallback for English only
        if language == 'en':
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.save_to_file(text, audio_path)
                engine.runAndWait()
                return audio_path
            except Exception as fallback_e:
                st.error(f"Fallback TTS failed: {fallback_e}")
        return None

def play_audio(audio_file):
    """Play audio file using streamlit's built-in audio player"""
    try:
        if os.path.exists(audio_file):
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.error(f"Audio file not found: {audio_file}")
    except Exception as e:
        st.error(f"Audio playback error: {str(e)}")
# ------------------------------
# LLM CHAT ASSISTANT
# ------------------------------
def chat_with_llm(api_key, user_message, disease_info):
    """Connect to LLM API and get response"""
    if not api_key:
        return "Please provide an API key in the sidebar configuration."
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Add disease info context to the prompt
    context = f"""
    I'm analyzing a plant leaf with the following information:
    - Detected disease: {disease_info.get('disease', 'Unknown')}
    - Severity: {disease_info.get('severity', 'Unknown')}
    - Plant type: {disease_info.get('plant', 'Unknown')}
    
    My question is: {user_message}
    """
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": context}
        ]
    }
    
    try:
        st.info("Connecting to OpenRouter API...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        else:
            error_message = f"API Error (Status {response.status_code}): {response.text}"
            st.error(error_message)
            return f"Error communicating with the AI service: {error_message}"
    except requests.exceptions.Timeout:
        return "Connection timed out. The API service might be busy or unavailable."
    except requests.exceptions.ConnectionError:
        return "Connection error. Please check your internet connection."
    except Exception as e:
        return f"Connection error: {str(e)}"
# ------------------------------
# MAIN STREAMLIT APP
# ------------------------------
def main():
    st.title("Plant Disease Diagnosis System")
    st.markdown("### Upload a leaf image for disease diagnosis and treatment recommendation")
    
    # Create sidebar for configurations
    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input("Model Path", r"C:\Users\USER\Desktop\Project2\inception_model_trained.h5")
        train_data_folder = st.text_input("Training Data Folder", r"D:\project\25000reduced_dataset_split_dataset\test")
        ontology_csv = st.text_input("Ontology CSV Path", r"C:\Users\USER\Desktop\Project2\plant_disease_ontology.csv")
        treatment_excel = st.text_input("Treatment Excel Path", r"C:\Users\USER\Desktop\plant_disease_treatments.xlsx")
        lstm_model_path = st.text_input("LSTM Model Path", r"C:\Users\USER\Desktop\Project2\lstm_model.h5")
        time_series_csv = st.text_input("Time Series CSV Path", r"C:\Users\USER\Downloads\synthetic_time_series.csv")
        
        st.header("LLM Assistant")
        api_key = st.text_input("OpenRouter API Key", "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", type="password")
        
        st.header("Actions")
        if st.button("Generate PDF Report"):
            if st.session_state.results:
                success = generate_pdf_report(st.session_state.results)
                if success:
                    st.success("PDF Report Generated Successfully!")
                    st.download_button(
                        label="Download Report",
                        data=open("Plant_Disease_Report.pdf", "rb").read(),
                        file_name="Plant_Disease_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("No results available. Please analyze an image first.")
    
    # Load the model at the start of app if paths are provided
    try:
        if os.path.exists(model_path):
            with st.spinner("Loading model..."):
                model = load_model(model_path)
                st.sidebar.success("Model loaded successfully!")
                
                # Get class names from training data if available
                if os.path.exists(train_data_folder):
                    class_names = get_class_names_from_directory(train_data_folder)
                    st.sidebar.success(f"Found {len(class_names)} plant disease classes.")
                else:
                    st.sidebar.warning("Training data folder not found. Using default class names.")
                    class_names = {0: "Unknown Class"}
        else:
            st.sidebar.error("Model path not found. Please provide a valid path.")
            model = None
            class_names = {0: "Unknown Class"}
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        model = None
        class_names = {0: "Unknown Class"}
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    # If file is uploaded
    if uploaded_file is not None and model is not None:
        # Read and display the image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, width=300)
        
        # Save the image temporarily for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_file.write(uploaded_file.getbuffer())
        image_path = temp_file.name
        
        # Process the image
        with st.spinner("Processing image..."):
            # Image preprocessing
            img_array = preprocess_image(image_path)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_names.get(predicted_class, "Unknown")
            confidence = predictions[0][predicted_class] * 100
            
            # Segmentation
            leaf_mask, segmented_img, original_img = segment_leaf(image_path)
            
            # Save original image for the report
            plt.imsave("temp_report_images/original.png", cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            
            # Display segmented image
            with col2:
                st.subheader("Segmented Leaf")
                st.image(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB), width=300)
                plt.imsave("temp_report_images/segmented.png", cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
            
            # Identification
            st.subheader("Identification Results")
            st.write(f"**Predicted Disease:** {predicted_label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # XAI Analysis
            st.subheader("XAI Analysis")
            col1, col2 = st.columns(2)
            
            # Grad-CAM
            heatmap = generate_gradcam_heatmap(model, img_array, predicted_class)
            plt.imsave("temp_report_images/heatmap.png", heatmap, cmap='jet')
            overlay_image = cv2.addWeighted(np.uint8(img_array[0]*255), 0.6, 
                                          cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET), 0.4, 0)
            
            with col1:
                st.write("**Grad-CAM Heatmap**")
                st.image(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), width=300)
            
            # LIME
            lime_overlay, lime_mask = predict_and_display_with_lime(image_path, model)
            lime_result = mark_boundaries(lime_overlay, lime_mask)
            plt.imsave("temp_report_images/lime_overlay.png", lime_result)
            
            with col2:
                st.write("**LIME Explanation**")
                st.image(lime_result, width=300)
            
            # Faithfulness Score
            gradcam_faith, lime_faith = compute_faithfulness(image_path, model)
            st.write(f"**Faithfulness Scores:** Grad-CAM: {gradcam_faith:.4f} | LIME: {lime_faith:.4f}")
            
            # Disease Severity Analysis
            st.subheader("Disease Severity Analysis")
            severity, coverage_ratio = calculate_severity(leaf_mask, heatmap)
            col1, col2 = st.columns(2)
            
            with col1:
                severity_chart = create_severity_donut_chart(severity, coverage_ratio)
                st.pyplot(severity_chart)
            
            with col2:
                st.write(f"**Severity:** {severity}")
                st.write(f"**Coverage:** {coverage_ratio:.1f}%")
                st.write("**Severity Levels:**")
                st.write("- **Healthy (0%):** No disease detected")
                st.write("- **Mild (0-20%):** Early stage, limited coverage")
                st.write("- **Moderate (20-50%):** Significant coverage, progressing")
                st.write("- **Severe (>50%):** Advanced disease, extensive coverage")
            
           # Extract plant name from predicted label (assuming format is "PlantName_Disease")
            plant_name = predicted_label.split('_')[0] if '_' in predicted_label else "Unknown"
            
            # Disease Management & Treatment
            st.subheader("Disease Management & Treatment")

            # Get treatment recommendation
            if os.path.exists(treatment_excel):
                treatment = get_treatment_recommendation(predicted_label, severity, treatment_excel)
                st.write(f"**Recommended Treatment:** {treatment}")
            else:
                st.warning(f"Treatment database not found at '{treatment_excel}'. Treatment recommendations unavailable.")
                treatment = "Treatment database not found"
            
            # Knowledge Graph Visualization
            st.subheader("Disease Ontology")
            ontology_fig = display_knowledge_graph(plant_name, ontology_csv)
            if ontology_fig:
                st.pyplot(ontology_fig)
            
            # Disease Spread Prediction
            st.subheader("Disease Spread Prediction")
            spread_prediction = predict_disease_spread(predicted_label, severity, lstm_model_path, time_series_csv)
            
            if isinstance(spread_prediction, tuple):
                prediction_values, prediction_fig = spread_prediction
                severity_growth, confidence = prediction_values
                st.write(f"**Predicted Spread Severity:** {severity_growth:.2f} (Scale 0-5)")
                st.write(f"**Prediction Confidence:** {confidence*100:.2f}%")
                st.pyplot(prediction_fig)
            else:
                st.write(f"**Spread Prediction:** {spread_prediction}")
            
            # Store results in session state for report generation
            st.session_state.results = {
                'predicted_class': predicted_label,
                'confidence': confidence,
                'severity': severity,
                'coverage_ratio': coverage_ratio,
                'treatment': treatment,
                'spread_prediction': spread_prediction,
                'gradcam_faith': gradcam_faith,
                'lime_faith': lime_faith,
                'plant': plant_name
            }
            
            # AI Chat Assistant
            st.subheader("AI Assistant")
            user_question = st.text_input("Ask about this plant disease:", "")
            if user_question and api_key:
                with st.spinner("Getting response from AI assistant..."):
                    try:
                        disease_info = {
                            'disease': predicted_label,
                            'severity': severity,
                            'plant': plant_name
                        }
                        assistant_response = chat_with_llm(api_key, user_question, disease_info)
                        st.markdown(assistant_response)
                    except Exception as e:
                        st.error(f"Error with AI assistant: {str(e)}")
                        st.info("You can continue using other features of the application.")
            # NEW MODULE: Text-to-Speech and Translation
            st.subheader("Text-to-Speech & Translation")

            # Only show this section if we have a treatment recommendation
            if treatment and treatment != "Treatment database not found":
                # Tamil translation
                with st.spinner("Translating to Tamil..."):
                    tamil_treatment = translate_text(treatment, target_language='ta')
                
                # Display both original and translated text
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**English:**")
                    st.write(treatment)
                    
                    # Generate English TTS
                    english_audio = text_to_speech(treatment, language='en', filename="english_treatment.mp3")
                    if english_audio:
                        st.audio(open(english_audio, "rb").read())
                
                with col2:
                    st.write("**Tamil:**")
                    st.write(tamil_treatment)
                    
                    # Generate Tamil TTS
                    tamil_audio = text_to_speech(tamil_treatment, language='ta', filename="tamil_treatment.mp3")
                    if tamil_audio:
                        st.audio(open(tamil_audio, "rb").read())
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    if english_audio:
                        with open(english_audio, "rb") as file:
                            st.download_button(
                                "Download English Audio", 
                                data=file.read(), 
                                file_name="treatment_english.mp3", 
                                mime="audio/mp3"
                            )
                with col2:
                    if tamil_audio:
                        with open(tamil_audio, "rb") as file:
                            st.download_button(
                                "Download Tamil Audio", 
                                data=file.read(), 
                                file_name="treatment_tamil.mp3", 
                                mime="audio/mp3"
                            )
            else:
                st.info("Treatment recommendation not available. Text-to-speech feature is disabled.")
            

        # With this code:
        try:
            # Close any potential file handles first
            if 'temp_file' in locals() and hasattr(temp_file, 'close'):
                temp_file.close()
            
            # Try to delete the file
            os.unlink(image_path)
        except Exception as e:
            st.warning(f"Note: Could not delete temporary file: {e}")
            # Not critical, so we can continue
if __name__ == "__main__":
    main()
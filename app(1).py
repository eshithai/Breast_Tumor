import streamlit as st
import numpy as np
import joblib

# Load the saved model
model = joblib.load('best_model.pkl')

# Set custom page configuration
st.set_page_config(
    page_title="Tumor Detection System",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Background CSS styles
st.markdown(
    """
    <style>
    body {
        background-image: url('https://example.com/your-background-image.jpg');  /* Replace with your image URL */
        background-size: cover;
        color: #2C3E50;  /* Dark Blue */
    }
    .stButton>button {
        background-color: #2ECC71;  /* Green */
        color: white;
        border: None;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #27AE60;  /* Darker Green */
    }
    .stNumberInput {
        background-color: #ECF0F1;  /* Light Grey */
        border-radius: 5px;
        border: 2px solid #2980B9;  /* Bright Blue */
        padding: 10px;
    }
    .stHeader, .stSubheader, .stTitle {
        color: #2980B9;  /* Bright Blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar with creative information about the app
st.sidebar.title("About the App 🩺")
st.sidebar.markdown(""" 
### Welcome to the *Impedance-Based Tumor Detection System*! 🎉
This app uses the power of *Machine Learning* and *bioelectrical impedance measurements* to assist in predicting tumor types such as *Carcinoma, Fibro-adenoma, Mastopathy, Glandular, Connective, and Adipose*.

#### How Does It Work? 🧠
By analyzing key tissue impedance features such as *Impedance at 0 Hz, High-Frequency Slope*, and many more, the model predicts the tumor type with precision.

#### Why Use This App? 🚀
- *Early detection* can be lifesaving.
- *Non-invasive data* is used, meaning you don't need to worry about complex, high-risk procedures.
- It's *fast* and provides *instant results* after inputting the necessary impedance values!

#### Please Note: 🔍
This app is not a substitute for professional medical diagnosis but a tool to assist in early detection and awareness. Always consult with a medical professional for final evaluations and treatment plans.
""")

# Title and description
st.title("🩺 Impedance-Based Tumor Detection System")
st.markdown(""" 
#### A simple and effective tool for early tumor diagnosis.
This system analyzes *breast tissue impedance* measurements to predict the type of tumor.
""")

# Add some spacing and a divider
st.markdown("---")

# Input section for tumor measurements
st.header("🔬 Input the Tumor Measurements:")
st.write("Please provide the following tissue impedance feature values for analysis.")

# Organize the inputs in a more spacious layout using three columns
col1, col2, col3 = st.columns(3)

with col1:
    I0 = st.number_input('I0 (Impedance at 0 Hz)', min_value=-1000.0, max_value=1000.0, value=0.09, help="Impedance at 0 Hz")
    PA500 = st.number_input('PA500 (Phase Angle at 500 Hz)', min_value=-1000.0, max_value=1000.0, value=-0.07, help="Phase Angle at 500 Hz")

with col2:
    HFS = st.number_input('HFS (High-Frequency Slope)', min_value=-1000.0, max_value=1000.0, value=-0.13, help="High-Frequency Slope")
    DA = st.number_input('DA (Delta Amplitude)', min_value=-1000.0, max_value=1000.0, value=-0.08, help="Delta Amplitude")

with col3:
    Area = st.number_input('Area', min_value=-1000.0, max_value=1000.0, value=0.12, help="Area under the impedance curve")
    ADA = st.number_input('A.DA (Amplitude Delta Area)', min_value=-1000.0, max_value=1000.0, value=0.08, help="Amplitude Delta Area")

# Add more fields under the same spacious layout
col4, col5, col6 = st.columns(3)

with col4:
    Max_IP = st.number_input('Max.IP (Max Impedance Peak)', min_value=-1000.0, max_value=1000.0, value=0.08, help="Maximum Impedance Peak")

with col5:
    DR = st.number_input('DR (Decay Rate)', min_value=-1000.0, max_value=1000.0, value=0.07, help="Decay Rate")

with col6:
    P = st.number_input('P (Periodicity)', min_value=-1000.0, max_value=1000.0, value=50.0, help="Periodicity of the signal")

# Add custom feature input for I0_log
I0_log = st.number_input('I0_log (Log of Impedance at 0 Hz)', min_value=-1000.0, max_value=1000.0, value=0.09, help="Log of Impedance at 0 Hz")

# Add some space before the prediction button
st.markdown("---")

# Collect all inputs into a numpy array
input_data = np.array([I0, PA500, HFS, DA, Area, ADA, Max_IP, DR, P, I0_log]).reshape(1, -1)

# Map model output to tumor classes
class_mapping = {0: "Carcinoma", 1: "Fibro-adenoma", 2: "Mastopathy", 3: "Glandular", 4: "Connective", 5: "Adipose"}

# Predict when the user clicks the button
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)  # Use input data directly for prediction
        confidence = model.predict_proba(input_data)  # Get confidence score
        
        predicted_class = class_mapping[prediction[0]]
        class_confidence = confidence[0][prediction[0]] * 100
        
        # Display the prediction result and confidence
        if predicted_class == "Carcinoma":
            st.error(f"🛑 The tumor is predicted to be: *{predicted_class}*")
            st.markdown("""
            <div style="background-color:#ECF0F1; padding: 10px; border-radius: 5px;">
            <strong>General Medical Prescription for Carcinoma:</strong><br>
            1. Consult an Oncologist for evaluation and treatment planning.<br>
            2. Undergo diagnostic tests: MRI, CT scans, or biopsy.<br>
            3. Possible treatments: Surgery, Radiation, Chemotherapy, Hormonal Therapy, and Targeted Therapy.<br>
            4. Regular follow-ups for monitoring and support.
            </div>
            """, unsafe_allow_html=True)
        elif predicted_class == "Fibro-adenoma":
            st.error(f"🛑 The tumor is predicted to be: *{predicted_class}*")
            st.markdown("""
            <div style="background-color:#ECF0F1; padding: 10px; border-radius: 5px;">
            <strong>General Medical Prescription for Fibro-adenoma:</strong><br>
            - Consult a physician for monitoring or possible removal.<br>
            - Surgery may be considered if the tumor grows or causes discomfort.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ The tumor is predicted to be: *{predicted_class}*")
        
        st.write(f"🔬 *Model Confidence*: {class_confidence:.2f}% sure the tumor is {predicted_class}.")
        
        # Display general medical prescriptions for other tumor types
        if predicted_class == "Mastopathy":
            st.write("""#### 📋 General Medical Prescription for Mastopathy:
            - Regular check-ups to monitor the condition.
            - Consult your physician for further recommendations.""")
        elif predicted_class == "Glandular":
            st.write("""#### 📋 General Medical Prescription for Glandular Tumors:
            - Consider regular monitoring and follow-ups with a specialist.""")
        elif predicted_class == "Connective":
            st.write("""#### 📋 General Medical Prescription for Connective Tumors:
            - Surgery may be necessary based on the size or discomfort caused.
            - Follow-up with a medical professional for further advice.""")
        elif predicted_class == "Adipose":
            st.write("""#### 📋 General Medical Prescription for Adipose Tumors:
            - Consider consultation with a specialist.
            - Removal may be an option if the tumor grows or causes discomfort.""")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Add a footer with disclaimer
st.markdown(""" 
--- 
*Disclaimer*: This tool is for educational purposes only and is not a substitute for professional medical advice. Please consult a healthcare professional for  for accurate diagnosis and treatment options.
""")

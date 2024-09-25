import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

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
    .prescription-box {
        background-color: #DFF9F0; /* Light green for boxes */
        border: 1px solid #2ECC71; /* Green border */
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar information
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

# Input section for tumor measurements
st.header("🔬 Input the Tumor Measurements:")
st.write("Please provide the following tissue impedance feature values for analysis.")

# Organize the inputs using columns
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

col4, col5, col6 = st.columns(3)

with col4:
    Max_IP = st.number_input('Max.IP (Max Impedance Peak)', min_value=-1000.0, max_value=1000.0, value=0.08, help="Maximum Impedance Peak")

with col5:
    DR = st.number_input('DR (Decay Rate)', min_value=-1000.0, max_value=1000.0, value=0.07, help="Decay Rate")

with col6:
    P = st.number_input('P (Periodicity)', min_value=-1000.0, max_value=1000.0, value=50.0, help="Periodicity of the signal")

I0_log = st.number_input('I0_log (Log of Impedance at 0 Hz)', min_value=-1000.0, max_value=1000.0, value=0.09, help="Log of Impedance at 0 Hz")

# Collect all inputs into a numpy array
input_data = np.array([I0, PA500, HFS, DA, Area, ADA, Max_IP, DR, P, I0_log]).reshape(1, -1)

# Map model output to tumor classes
class_mapping = {0: "Carcinoma", 1: "Fibro-adenoma", 2: "Mastopathy", 3: "Glandular", 4: "Connective", 5: "Adipose"}

# Define colors for each class
color_mapping = {
    "Carcinoma": "red",           # Cancerous
    "Fibro-adenoma": "green",     # Benign
    "Mastopathy": "blue",          # Benign
    "Glandular": "purple",         # Benign
    "Connective": "orange",        # Benign
    "Adipose": "cyan"              # Benign
}

# Predict and generate radar chart
if st.button("🔍 Predict"):
    try:
        prediction = model.predict(input_data)
        confidence = model.predict_proba(input_data)
        
        predicted_class = class_mapping[prediction[0]]
        class_confidence = confidence[0][prediction[0]] * 100

        st.write(f"🔬 *Model Confidence*: {class_confidence:.2f}% sure the tumor is {predicted_class}.")

        # Radar chart for visual representation of inputs
        feature_names = ['I0', 'PA500', 'HFS', 'DA', 'Area', 'ADA', 'Max_IP', 'DR', 'P', 'I0_log']
        fig = go.Figure()

        # Use the color based on the predicted class
        predicted_color = color_mapping[predicted_class]

        fig.add_trace(go.Scatterpolar(
            r=input_data.flatten(),
            theta=feature_names,
            fill='toself',
            fillcolor=predicted_color,  # Set fill color based on prediction
            name='Input Features'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[-1000, 1000])
            ),
            showlegend=False,
            title="Input Features Visualization"
        )

        st.plotly_chart(fig)

        # Display the predicted result
        if predicted_class == "Carcinoma":
            st.error(f"🛑 The tumor is predicted to be: *{predicted_class}* (Cancerous)")
        else:
            st.success(f"✅ The tumor is predicted to be: *{predicted_class}*")

    except Exception as e:
        st.error(f"🚨 An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2024 Tumor Detection System. All rights reserved.")

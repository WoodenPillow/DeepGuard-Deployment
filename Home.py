import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from fpdf import FPDF
from features import extract_features  # Your feature extraction module

# Initialize session state if needed.
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []

# -------------------------------
# Set base directory for portability:
# Set base directory relative to current file
base_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(base_dir, "Models")

# -------------------------------
# Page Configuration & Custom CSS
# -------------------------------
st.set_page_config(page_title="DeepGuard", page_icon="🛡️", layout="wide")
st.markdown("""
    <style>
        /* Overall dark theme styling */
        body {
            background-color: #111;
            color: #EEE;
        }
        /* Title and gradient header styling */
        .special-header { 
            background: linear-gradient(90deg, #2e6c80, #00ff00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        .tagline {
            font-size: 0.9rem;
            color: #ccc;
            margin-top: 0;
        }
        /* Metrics styling */
        .stMetric {
            background-color: #222;
            border-radius: 8px;
            padding: 0.5rem;
            font-size: 1.1rem;
        }
        /* Use default "normal" delta colors: 
           positive in green, negative in red */
        /* Layout improvements */
        .left-col {
            padding-right: 2rem;
        }
        .right-col {
            padding-left: 2rem;
        }
        /* Make upload history container scrollable */
        .scrollable-history {
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Title & Tagline with Enhanced Styling
# -------------------------------
st.markdown("""
    <style>
        /* Overall dark theme styling */
        body {
            background-color: #111;
            color: #EEE;
        }
        /* Main Title with Gradient Text */
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 0.5rem;  /* Reduced margin to move title up */
            background: linear-gradient(90deg, #2e6c80, #00ff00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        /* Styled Subtitle with White Outline */
        .subtitle { 
            font-size: 1.2rem;
            font-weight: 600;
            color: #000;
            background-color: #fff;
            padding: 0.3rem 0.8rem;
            border: 2px solid #fff;  /* White border */
            border-radius: 15px;
            display: block;         /* block-level to allow centering */
            width: fit-content;      /* Shrinks to the content's width */
            margin: 0.5rem auto;     /* Centers horizontally with auto margins */
            text-align: center;
        }
        /* Tagline styling */
        .tagline {
            font-size: 0.9rem;
            color: #ccc;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

def get_gradient_color(prob):
    """
    Interpolates a color between green and red based on the probability.
    - If prob is 0, returns green (rgb(0, 255, 0)).
    - If prob is 1, returns red (rgb(255, 0, 0)).
    - Values in between produce a mix.
    """
    r = int(prob * 255)
    g = int((1 - prob) * 255)
    b = 0
    return f"rgb({r}, {g}, {b})"

def render_confidence_bar(percentage, prob):
    """
    Returns an HTML snippet with a horizontal progress bar (made slimmer)
    and the numeric percentage displayed alongside.
    """
    color = get_gradient_color(prob)
    return f"""
    <div style="display: flex; align-items: center;">
        <div style="width: 50%; background-color: #444; border-radius: 5px; margin-right: 10px;">
            <div style="width: {percentage}%; background-color: {color}; height: 10px; border-radius: 5px;"></div>
        </div>
        <div style="font-size: 1.2rem; color: #EEE;">{percentage}%</div>
    </div>
    """
def generate_pdf_report(summary_df, model_df, filetype_df, size_df, history_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Detection Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Benign vs Malicious Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in summary_df.iterrows():
        pdf.cell(0, 8, f"{row['Category']}: {row['Count']} ({row['Percentage']})", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Model Used Breakdown", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in model_df.iterrows():
        pdf.cell(0, 8, f"{row['Model Used']}: {row['Count']} ({row['Percentage']})", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "File Type Breakdown", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in filetype_df.iterrows():
        pdf.cell(0, 8, f"{row['FileType']}: {row['Count']} ({row['Percentage']})", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "File Size Statistics", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in size_df.iterrows():
        pdf.cell(0, 8, f"{row['Metric']}: {row['Value (bytes)']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Detailed History", ln=True)
    pdf.set_font("Arial", "", 10)
    for idx, row in history_df.iterrows():
        text = f"{row['Filename']} | {row['Model Used']} | {row['Prediction']} | {row['Confidence (%)']}"
        pdf.cell(0, 6, text, ln=True)
    
    return pdf.output(dest="S").encode("latin1")

st.markdown("<div class='main-title'>DeepGuard: Deep Learning Ransomware Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>File Detection</div>", unsafe_allow_html=True)
st.markdown("""
<p class="tagline">
This system uses a combination of classical machine learning and a deep learning MLP model for malware detection.<br>
Select your preferred model, upload a file, and view the predicted result.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Layout: Two columns for controls and evaluation metrics.
# -------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("Select Model & Upload a File")
    
    model_options = ["Random Forest", "XGBoost", "MLP", "Ensemble"]
    selected_model = st.selectbox("Choose the model", model_options)
    
    uploaded_file = st.file_uploader("Upload a file")
    
    if st.button("Run Detection") and uploaded_file is not None:
        # Save the uploaded file temporarily.
        temp_folder = "temp"
        os.makedirs(temp_folder, exist_ok=True)
        temp_file_path = os.path.join(temp_folder, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract features.
        features = extract_features(temp_file_path)
        if features is None:
            st.error("Feature extraction failed. Please try another file.")
        else:
            outcome = ""
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if selected_model == "Random Forest":
                model_path = os.path.join(model_folder, "rf_model.pkl")
                model_rf = joblib.load(model_path)
                # Use predict_proba to get a probability rather than a hard label.
                rf_prob = model_rf.predict_proba(features)[:, 1][0]
                outcome = "MALICIOUS" if rf_prob > 0.5 else "BENIGN"
                pred_prob = rf_prob

            elif selected_model == "XGBoost":
                model_path = os.path.join(model_folder, "xgb_model.pkl")
                model_xgb = joblib.load(model_path)
                xgb_prob = model_xgb.predict_proba(features)[:, 1][0]
                outcome = "MALICIOUS" if xgb_prob > 0.5 else "BENIGN"
                pred_prob = xgb_prob

            elif selected_model == "MLP":
                model_path = os.path.join(model_folder, "mlp.pt")
                input_dim = features.shape[1]  # Expected to be 6
                # Define the MLP architecture (must match training)
                class MLP(nn.Module):
                    def __init__(self, input_dim):
                        super(MLP, self).__init__()
                        self.model = nn.Sequential(
                            nn.Linear(input_dim, 1024),
                            nn.ReLU(),
                            nn.BatchNorm1d(1024),
                            nn.Dropout(0.4),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.4),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.BatchNorm1d(256),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Dropout(0.3),
                            nn.Linear(128, 1)
                        )
                    
                    def forward(self, x):
                        return self.model(x)
                mlp_model = MLP(input_dim).to(device)
                mlp_model.load_state_dict(torch.load(model_path, map_location=device))
                mlp_model.eval()
                with torch.no_grad():
                    tensor_features = torch.tensor(features, dtype=torch.float32, device=device)
                    logits = mlp_model(tensor_features).cpu().numpy().squeeze()
                    pred_prob = 1 / (1 + np.exp(-logits))
                outcome = "MALICIOUS" if pred_prob > 0.5 else "BENIGN"
                outcome += f" (Prob: {pred_prob:.2f})"
                
            elif selected_model == "Ensemble":
                # Load predictions from RF, XGBoost, and MLP.
                rf_model = joblib.load(os.path.join(model_folder, "rf_model.pkl"))
                xgb_model = joblib.load(os.path.join(model_folder, "xgb_model.pkl"))
                mlp_path = os.path.join(model_folder, "mlp.pt")
                input_dim = features.shape[1]
                # Define MLP as before.
                class MLP(nn.Module):
                    def __init__(self, input_dim):
                        super(MLP, self).__init__()
                        self.model = nn.Sequential(
                            nn.Linear(input_dim, 1024),
                            nn.ReLU(),
                            nn.BatchNorm1d(1024),
                            nn.Dropout(0.4),
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.BatchNorm1d(512),
                            nn.Dropout(0.4),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.BatchNorm1d(256),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.BatchNorm1d(128),
                            nn.Dropout(0.3),
                            nn.Linear(128, 1)
                        )
                    
                    def forward(self, x):
                        return self.model(x)
                mlp_model = MLP(input_dim).to(device)
                mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
                mlp_model.eval()
                with torch.no_grad():
                    tensor_features = torch.tensor(features, dtype=torch.float32, device=device)
                    logits = mlp_model(tensor_features).cpu().numpy().squeeze()
                    mlp_prob = 1 / (1 + np.exp(-logits))
                rf_prob = rf_model.predict_proba(features)[:, 1][0]
                xgb_prob = xgb_model.predict_proba(features)[:, 1][0]
                ensemble_prob = (rf_prob + xgb_prob + mlp_prob) / 3.0
                outcome = "MALICIOUS" if ensemble_prob > 0.5 else "BENIGN"
                outcome += f" (Prob: {ensemble_prob:.2f})"
                pred_prob = ensemble_prob  # Use ensemble probability for confidence
            
            else:
                outcome = "Model not found."
                pred_prob = None  # Not applicable

            # Compute confidence percentage if pred_prob is defined.
            if pred_prob is not None:
                confidence_percentage = int(pred_prob * 100)
            else:
                confidence_percentage = "N/A"

            # Add result to upload history (as before)
            st.session_state.upload_history.append({
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize (bytes)": uploaded_file.size,
                "Model Used": selected_model,
                "Prediction": outcome,
                "Confidence (%)": f"{confidence_percentage}%" if isinstance(confidence_percentage, int) else confidence_percentage
            })

            st.success("Prediction complete!")

            # Display immediate result with a colored icon.
            if outcome.startswith("BENIGN"):
                st.markdown("<h2 style='color:#00ff00;'>🟢 Benign</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:#ff3333;'>🔴 Malicious</h2>", unsafe_allow_html=True)

            # Now, display the confidence bar with dynamic gradient.
            if isinstance(confidence_percentage, int):
                st.markdown(render_confidence_bar(confidence_percentage, pred_prob), unsafe_allow_html=True)
            else:
                st.write("Confidence Level: N/A")
            
with col_right:
    st.header("Evaluation Metrics")
    # Show hard-coded evaluation metrics relative to Random Forest (91% baseline)
    if selected_model == "Random Forest":
        st.metric("Accuracy", "91%", delta="0%", delta_color="normal")
        st.metric("Precision", "91%", delta="0%", delta_color="normal")
        st.metric("Recall", "91%", delta="0%", delta_color="normal")
        st.metric("F1-Score", "91%", delta="0%", delta_color="normal")
    elif selected_model == "XGBoost":
        st.metric("Accuracy", "86%", delta="-5%", delta_color="normal")
        st.metric("Precision", "86%", delta="-5%", delta_color="normal")
        st.metric("Recall", "86%", delta="-5%", delta_color="normal")
        st.metric("F1-Score", "86%", delta="-5%", delta_color="normal")
    elif selected_model == "MLP":
        st.metric("Accuracy", "85%", delta="-6%", delta_color="normal")
        st.metric("Precision", "85%", delta="-6%", delta_color="normal")
        st.metric("Recall", "85%", delta="-6%", delta_color="normal")
        st.metric("F1-Score", "85%", delta="-6%", delta_color="normal")
    elif selected_model == "Ensemble":
        st.metric("Accuracy", "90%", delta="-1%", delta_color="normal")
        st.metric("Precision", "90%", delta="-1%", delta_color="normal")
        st.metric("Recall", "90%", delta="-1%", delta_color="normal")
        st.metric("F1-Score", "90%", delta="-1%", delta_color="normal")
    else:
        st.write("No metrics to show.")

st.markdown("---")

# -------------------------------
# Upload History Section (scrollable)
# -------------------------------
st.subheader("Upload History & Detection Results")
if st.session_state.upload_history:
    history_df = pd.DataFrame(st.session_state.upload_history)
    st.dataframe(history_df, height=300)
else:
    st.write("No files have been uploaded yet.")

# -------------------------------
# Download Report Section
# -------------------------------
if st.session_state.upload_history:
    history_df = pd.DataFrame(st.session_state.upload_history)

    # Create summaries similar to your CSV version.
    benign_count = (history_df["Prediction"].str.startswith("BENIGN")).sum()
    malicious_count = (history_df["Prediction"].str.startswith("MALICIOUS")).sum()
    total_files = len(history_df)
    benign_pct = (benign_count / total_files * 100) if total_files > 0 else 0
    malicious_pct = (malicious_count / total_files * 100) if total_files > 0 else 0
    summary_df = pd.DataFrame({
        "Category": ["Benign", "Malicious"],
        "Count": [benign_count, malicious_count],
        "Percentage": [f"{benign_pct:.1f}%", f"{malicious_pct:.1f}%"]
    })
    
    # Create a summary for models.
    model_summary_df = history_df.groupby("Model Used").size().reset_index(name="Count")
    model_summary_df["Percentage"] = model_summary_df["Count"] / total_files * 100
    model_summary_df["Percentage"] = model_summary_df["Percentage"].map(lambda x: f"{x:.1f}%")
    model_summary_df.rename(columns={"Model Used": "Model Used"}, inplace=True)
    
    # Create a summary for file types.
    filetype_summary_df = history_df.groupby("FileType").size().reset_index(name="Count")
    filetype_summary_df["Percentage"] = filetype_summary_df["Count"] / total_files * 100
    filetype_summary_df["Percentage"] = filetype_summary_df["Percentage"].map(lambda x: f"{x:.1f}%")
    
    # Create a summary for file sizes.
    avg_size = history_df["FileSize (bytes)"].mean()
    min_size = history_df["FileSize (bytes)"].min()
    max_size = history_df["FileSize (bytes)"].max()
    size_summary_df = pd.DataFrame({
        "Metric": ["Average File Size", "Minimum File Size", "Maximum File Size"],
        "Value (bytes)": [f"{avg_size:.0f}", f"{min_size}", f"{max_size}"]
    })
    
    # Generate PDF report.
    pdf_report = generate_pdf_report(summary_df, model_summary_df, filetype_summary_df, size_summary_df, history_df)
    
    st.download_button(
        label="Download Report as PDF",
        data=pdf_report,
        file_name='detection_report.pdf',
        mime='application/pdf',
        type="primary"
    )

else:
    st.write("No detection history available to generate a report.")
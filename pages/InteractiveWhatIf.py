import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
import joblib

st.set_page_config(page_title="Interactive What-If", page_icon="🌱", layout="wide")

# Custom CSS for header styling
st.markdown("""
    <style>
        body { background-color: #111; color: #EEE; }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-top: 0.5rem;
            background: linear-gradient(90deg, #2e6c80, #00ff00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        .subtitle { 
            font-size: 1.2rem;
            font-weight: 600;
            color: #000;
            background-color: #fff;
            padding: 0.3rem 0.8rem;
            border: 2px solid #fff;
            border-radius: 15px;
            display: block;
            width: fit-content;
            margin: 0.5rem auto;
            text-align: center;
        }
        .tagline { 
            font-size: 0.9rem;
            color: #ccc;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>DeepGuard: Deep Learning Ransomware Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive What-If Prediction</div>", unsafe_allow_html=True)
st.markdown("""
<p class="tagline">
Adjust feature values to see how the ensemble model's prediction changes.
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Input Widgets for Features (using sliders)
# -------------------------------
st.subheader("Adjust Feature Values")
# Use sliders for these four features.
size_val    = st.slider("File Size (bytes)", min_value=1000, max_value=1000000, value=100000, step=1000)
exports_val = st.slider("Number of Exports",     min_value=0,    max_value=100,      value=0,       step=1)
imports_val = st.slider("Number of Imports",     min_value=0,    max_value=300,      value=0,       step=1)
strings_val = st.slider("Number of Strings",     min_value=0,    max_value=10000,    value=1000,    step=100)

# Set Year and Month as static values.
year_val = 2018
month_val = 10

# Combine inputs into a feature vector (shape: 1 x 6)
features = np.array([[size_val, exports_val, imports_val, strings_val, year_val, month_val]], dtype=np.float32)

# -------------------------------
# Define the MLP Model Architecture (for ensemble branch)
# -------------------------------
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

# -------------------------------
# Helper Functions for Confidence Bar
# -------------------------------
def get_gradient_color(prob):
    r = int(prob * 255)
    g = int((1 - prob) * 255)
    b = 0
    return f"rgb({r}, {g}, {b})"

def render_confidence_bar(percentage, prob):
    color = get_gradient_color(prob)
    return f"""
    <div style="display: flex; align-items: center;">
        <div style="width: 50%; background-color: #444; border-radius: 5px; margin-right: 10px;">
            <div style="width: {percentage}%; background-color: {color}; height: 10px; border-radius: 5px;"></div>
        </div>
        <div style="font-size: 1.2rem; color: #EEE;">{percentage}%</div>
    </div>
    """

# -------------------------------
# Run What-If Prediction using Ensemble Model
# -------------------------------
if st.button("Run What-If Prediction"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features.shape[1]  # equals 6 now.
    
    # For multipage apps, since this file is inside Deployment/pages, go up three directories to the project root.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_folder = os.path.join(base_dir, "Models")
    
    # Load Random Forest and XGBoost models via joblib.
    try:
        rf_model = joblib.load(os.path.join(model_folder, "rf_model.pkl"))
        xgb_model = joblib.load(os.path.join(model_folder, "xgb_model.pkl"))
    except Exception as e:
        st.error(f"Error loading RF/XGBoost models: {e}")
    
    # Load MLP model checkpoint.
    mlp_path = os.path.join(model_folder, "mlp.pt")
    mlp_model = MLP(input_dim).to(device)
    try:
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
    except Exception as e:
        st.error(f"Error loading MLP model: {e}")
    else:
        mlp_model.eval()
        with torch.no_grad():
            tensor_features = torch.tensor(features, dtype=torch.float32, device=device)
            logits = mlp_model(tensor_features).cpu().numpy().squeeze()
            mlp_prob = 1 / (1 + np.exp(-logits))
        
        # Get prediction probabilities from RF and XGBoost.
        try:
            rf_prob = rf_model.predict_proba(features)[:, 1][0]
            xgb_prob = xgb_model.predict_proba(features)[:, 1][0]
        except Exception as e:
            st.error(f"Error predicting with RF/XGBoost: {e}")
            rf_prob, xgb_prob = 0.0, 0.0
        
        # Ensemble probability: average of RF, XGBoost, and MLP predictions.
        ensemble_prob = (rf_prob + xgb_prob + mlp_prob) / 3.0
        outcome = "MALICIOUS" if ensemble_prob > 0.5 else "BENIGN"
        st.subheader(f"Prediction: {outcome} (Probability: {ensemble_prob:.2f})")
        
        # Compute confidence percentage.
        confidence_percentage = int(ensemble_prob * 100)
        st.markdown(render_confidence_bar(confidence_percentage, ensemble_prob), unsafe_allow_html=True)

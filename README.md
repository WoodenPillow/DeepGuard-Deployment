# 🛡️ DeepGuard: Deep Learning Ransomware Detection System

**DeepGuard** is a deep learning-based ransomware detection prototype featuring an interactive web interface for real-time threat simulation and analysis. It allows users to explore how system behavior influences model predictions by manipulating input features and observing results dynamically.

---

## 🧠 Project Overview

This project includes:

- A **pre-trained ML model** that classifies input feature vectors as either *benign* or *ransomware-related* activity
- A **Streamlit-based interface** for interactive predictions and What-If experiments
- Custom **feature preprocessing logic** to transform user input into model-compatible vectors
- Support for **dynamic feature adjustment**, enabling real-time simulation of system metrics (e.g., CPU usage, file entropy)

---

## 🔧 Core Features

- **Interactive What-If Panel**  
  Adjust key behavioral features manually and observe real-time model outputs

- **Preprocessing Pipeline**  
  Encapsulates data normalization and transformation before feeding into the model

- **Modular UI Design**  
  Built using Streamlit multipage layout for separation of functionality (Home, Simulation, etc.)

- **Explainability-Ready**  
  Code structure is compatible with tools like SHAP for future explainability integration

- **Lightweight & Self-Contained**  
  Designed for fast experimentation with minimal external dependencies

---

## 🚀 Getting Started

```bash
git clone https://github.com/woodenpillow/deepguard-deployment.git
cd deepguard-deployment
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Notes
- The model expects specific numeric input features simulating system activity. These can be adjusted via the UI sliders.

- Prediction logic is encapsulated in the main app and uses a joblib-serialized model.

- For extension, new models or real telemetry input sources can be plugged into the pipeline.

---

## ✅ Future Enhancements
- Real-time data integration via agent or API

- Add SHAP/LIME-based feature attribution

- Model re-training interface for new datasets

- Backend API for integrating with external monitoring systems

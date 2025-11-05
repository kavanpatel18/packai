"""
AI Packaging Optimizer - Streamlit App
--------------------------------------
✅ Auto-detects model, calibration, and dataset
✅ Handles tab or comma-separated CSVs safely
✅ Detects invalid datasets (like Python files)
✅ Allows image upload & predicts dimensions
✅ Recommends optimal box size
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ================== AUTO-DETECTION ==================
def find_file(keyword, folder="."):
    """Find a file by keyword in the current directory."""
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            return os.path.join(folder, f)
    return None

MODEL_PATH = "packaging_dim_model_finetuned.keras"
CALIBRATION_FILE = "calibration_factors_linear.npy"  # optional

DATASET_FILE = None
for f in os.listdir("."):
    if f.endswith(".csv") and "dataset" in f.lower():
        DATASET_FILE = f
        break

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="AI Packaging Optimizer", layout="centered")
st.title("📦 AI Packaging Optimizer")
st.caption("Estimate real-world package dimensions and suggest optimal box fit.")

# Sidebar info
st.sidebar.header("🧠 Configuration")
st.sidebar.write(f"**Model:** {MODEL_PATH or '❌ Not found'}")
st.sidebar.write(f"**Calibration:** {CALIBRATION_FILE or '❌ Not found'}")
st.sidebar.write(f"**Dataset:** {DATASET_FILE or '❌ Not found'}")

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model(path):
    if not path or not os.path.exists(path):
        st.error("❌ Model file not found!")
        st.stop()
    model = tf.keras.models.load_model(path)
    return model

model = load_model(MODEL_PATH)

# ================== LOAD CALIBRATION ==================
def load_calibration(file):
    if file and os.path.exists(file):
        factors = np.load(file)
        st.success("✅ Calibration factors loaded.")
        return factors
    else:
        st.warning("⚠️ No calibration file found. Predictions will be raw.")
        return None

calibration = load_calibration(CALIBRATION_FILE)

# ================== SAFE DATASET LOADER ==================
def safe_load_csv(path):
    """Safely load a CSV with auto delimiter handling."""
    if not path or not os.path.exists(path):
        st.info("📁 No dataset detected. Upload one below.")
        return None
    try:
        # Try tab first, then comma
        df = pd.read_csv(path, sep="\t", engine="python", on_bad_lines="skip")
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip")

        # Ensure dataset validity
        if df.shape[1] < 3:
            st.error("❌ This file doesn’t look like a valid dataset (too few columns).")
            return None

        df.columns = [c.strip().lower() for c in df.columns]
        st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df

    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        return None

df = safe_load_csv(DATASET_FILE)

# Allow manual dataset upload if needed
if df is None:
    uploaded_csv = st.file_uploader("📤 Upload a valid dataset CSV", type=["csv"])
    if uploaded_csv:
        df = safe_load_csv(uploaded_csv)

# ================== DATASET OVERVIEW ==================
if df is not None:
    st.subheader("📊 Dataset Overview")
    possible_cols = [
        ["product_length", "product_width", "product_height"],
        ["length", "width", "height"]
    ]
    found_cols = None
    for cols in possible_cols:
        if all(c in df.columns for c in cols):
            found_cols = cols
            break

    if found_cols:
        stats = df.describe()[found_cols]
        st.dataframe(stats)
        st.bar_chart(stats.loc[["mean", "max"]])
    else:
        st.warning(f"⚠️ No dimension columns found. Columns available: {list(df.columns)}")

# ================== IMAGE UPLOAD & PREDICTION ==================
st.subheader("📷 Upload Product Image")
uploaded_file = st.file_uploader("Drag and drop an image here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    with st.spinner("⏳ Predicting dimensions..."):
        preds = model.predict(x, verbose=0)[0]
        preds = np.array(preds)

        if calibration is not None:
            if calibration.shape == (3, 2):  # linear (scale, bias)
                preds = preds * calibration[:, 0] + calibration[:, 1]
            elif calibration.shape == (3,):
                preds = preds * calibration

    # Show predictions
    st.success("✅ Prediction complete!")
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 Length (in)", f"{preds[0]:.2f}")
    col2.metric("📏 Width (in)", f"{preds[1]:.2f}")
    col3.metric("📏 Height (in)", f"{preds[2]:.2f}")

    volume = preds[0] * preds[1] * preds[2]
    st.markdown(f"### 📦 Volume ≈ {volume:.1f} cubic inches")

    # Recommend box size
    standard_boxes = [
        {"name": "Small", "L": 8, "W": 6, "H": 4},
        {"name": "Medium", "L": 12, "W": 10, "H": 8},
        {"name": "Large", "L": 18, "W": 14, "H": 10},
        {"name": "XL", "L": 24, "W": 18, "H": 12},
        {"name": "XXL", "L": 30, "W": 24, "H": 18},
    ]
    fits = [b for b in standard_boxes if all(preds[i] <= b[d] for i, d in enumerate(["L", "W", "H"]))]
    if fits:
        best_box = min(fits, key=lambda b: b["L"] * b["W"] * b["H"])
        st.info(f"🎯 Recommended Box: **{best_box['name']}** ({best_box['L']}×{best_box['W']}×{best_box['H']} in)")
    else:
        st.warning("🚫 No standard box found that fits this product.")

    os.remove(img_path)

st.markdown("---")
st.caption("AI Packaging Optimizer © 2025 – Powered by TensorFlow & Streamlit")

"""
📦 AI Packaging Optimizer
Optimized & visually appealing student-style app
"""

import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from itertools import permutations

# ================== CONFIG ==================
# Removed as requested
# MIN_DIMS = np.array([1.0, 1.0, 0.5])
# MAX_DIMS = np.array([36.0, 36.0, 18.0])  # max limits for safe packing

# ================== AUTO-DETECTION ==================
def find_file(keyword, folder="."):
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            return os.path.join(folder, f)
    return None

MODEL_PATH = find_file("packaging_dim_model") or find_file("model")
CALIBRATION_FILE = find_file("calibration_factors")

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="AI Packaging Optimizer", layout="centered")
st.image("https://img.icons8.com/emoji/48/000000/package-emoji.png", width=60)
st.title("📦 AI Packaging Optimizer")
st.subheader("Estimate product dimensions and find the optimal box")

# Sidebar instructions
st.sidebar.header("How to use")
st.sidebar.write("""
1. Upload a product image (JPG/PNG)
2. AI predicts Length, Width, Height
3. See the recommended box (standard or custom)
""")

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model(path):
    if not path or not os.path.exists(path):
        st.error("❌ Model file not found!")
        st.stop()
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# ================== LOAD CALIBRATION ==================
def load_calibration(file):
    if file and os.path.exists(file):
        factors = np.load(file)
        st.success("✅ Calibration loaded")
        return factors
    st.warning("⚠️ No calibration file found. Predictions will be raw.")
    return None

calibration = load_calibration(CALIBRATION_FILE)

# ================== IMAGE UPLOAD ==================
uploaded_file = st.file_uploader("Upload product image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # ================== PREDICT DIMENSIONS ==================
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x, verbose=0)[0]
    preds = np.array(preds)

    if calibration is not None:
        if calibration.shape == (3, 2):
            preds = preds * calibration[:, 0] + calibration[:, 1]
        elif calibration.shape == (3,):
            preds = preds * calibration

    # Clamp dimensions (Removed as requested)
    # preds = np.clip(preds, MIN_DIMS, MAX_DIMS)

    # ================== DISPLAY PREDICTIONS ==================
    st.subheader("Predicted Dimensions (in inches)")
    col1, col2, col3 = st.columns(3)
    col1.metric("📏 Length", f"{preds[0]:.2f}")
    col2.metric("📏 Width", f"{preds[1]:.2f}")
    col3.metric("📏 Height", f"{preds[2]:.2f}")

    # ================== STANDARD BOXES ==================
    standard_boxes = [
        {"name": "Small", "L": 8, "W": 6, "H": 4},
        {"name": "Medium", "L": 12, "W": 10, "H": 8},
        {"name": "Large", "L": 16, "W": 12, "H": 10},
        {"name": "XL", "L": 20, "W": 16, "H": 12},
        {"name": "XXL", "L": 24, "W": 18, "H": 14},
        {"name": "XXXL", "L": 30, "W": 24, "H": 18},
    ]

    # ================== FIND FITTING BOXES ==================
    fits = []
    for box in standard_boxes:
        for perm in permutations(preds):
            if all(perm[i] <= box[d] for i, d in enumerate(["L", "W", "H"])):
                fits.append(box)
                break  # only need one orientation to fit

    if fits:
        best_box = min(fits, key=lambda b: b["L"] * b["W"] * b.get("H", 1)) # .get("H", 1) for safety
        st.success(
            f"🎯 Recommended Standard Box: **{best_box['name']}** ({best_box['L']}×{best_box['W']}×{best_box['H']} in)"
        )
    else:
        # Custom box with 5% margin (Removed MAX_DIMS limit)
        custom_box = {
            "L": preds[0] * 1.05,
            "W": preds[1] * 1.05,
            "H": preds[2] * 1.05,
        }
        st.warning(
            f"🚨 No standard box fits this product.\n"
            f"📦 Recommended Custom Box: {custom_box['L']:.1f}×{custom_box['W']:.1f}×{custom_box['H']:.1f} in"
        )

    os.remove(img_path)

st.markdown("---")
st.caption("AI Packaging Optimizer © 2025 – 😎")
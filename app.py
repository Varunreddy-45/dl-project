"""
Garbage Classification for Smart Waste Management - Streamlit Web App
Single self-contained file with drag-and-drop upload, webcam, and smart bin advice.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# 1. Config & Classes
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

SMART_BINS = {
    "cardboard": {"bin": "Blue Bin", "color": "#1E88E5", "desc": "Recyclable Dry Packaging. Flatten to save space."},
    "glass":     {"bin": "Green Bin", "color": "#2E7D32", "desc": "100% Endlessly Recyclable. Rinse clean."},
    "metal":     {"bin": "Yellow Bin", "color": "#F9A825", "desc": "Recyclable Metal / Cans. Empty liquid residue."},
    "paper":     {"bin": "Blue Bin", "color": "#1565C0", "desc": "Recyclable Paper. Keep dry and unsoiled."},
    "plastic":   {"bin": "Orange Bin", "color": "#E65100", "desc": "Recyclable Polymers (PET/HDPE). Rinse and crush."},
    "trash":     {"bin": "Grey Bin", "color": "#424242", "desc": "General Non-Recyclable Residuals. Dispose in landfill bin."}
}

# 2. Simple CNN Architecture
class GarbageCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


@st.cache_resource
def get_trained_model():
    """Loads or trains the model."""
    import main
    data_dir = main.create_dataset_if_missing()
    model, transform, device = main.train_and_plot(data_dir, epochs=4)
    return model, transform, device


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Smart Waste AI", page_icon="♻️", layout="centered")

st.title("♻️ Smart Waste Management System")
st.write("Deep Learning (CNN) Autonomous Garbage Classification & Smart Bin Segregation")

# Load model
with st.spinner("Initializing Deep Learning Model..."):
    model, transform, device = get_trained_model()

# Input options
st.subheader("1. Provide a Waste Item Image")
option = st.radio("Choose Input Mode:", ["📁 Upload an Image", "📷 Use Live Webcam", "🧪 Select Sample Image"], horizontal=True)

selected_image = None

if option == "📁 Upload an Image":
    uploaded_file = st.file_uploader("Upload garbage item", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)

elif option == "📷 Use Live Webcam":
    camera_pic = st.camera_input("Take a photo of garbage")
    if camera_pic:
        selected_image = Image.open(camera_pic)

elif option == "🧪 Select Sample Image":
    sample_items = {
        "Plastic Bottle": "test_samples/plastic.jpg",
        "Glass Bottle": "test_samples/glass.jpg",
        "Paper Sheet": "test_samples/paper.jpg",
        "Metal Can": "test_samples/metal.jpg",
        "Cardboard Box": "test_samples/cardboard.jpg",
        "General Trash": "test_samples/trash.jpg"
    }
    choice = st.selectbox("Select a sample:", list(sample_items.keys()))
    sample_path = sample_items[choice]
    if os.path.exists(sample_path):
        selected_image = Image.open(sample_path)

# Predict & Display
if selected_image:
    st.image(selected_image, caption="Current Waste Item", width=300)

    # Inference
    tensor = transform(selected_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
        conf, pred_idx = probs.max(0)

    pred_class = CLASSES[pred_idx.item()]
    bin_info = SMART_BINS[pred_class]
    confidence_pct = conf.item() * 100

    # Smart Bin Badge
    st.markdown(f"""
    <div style="background-color: {bin_info['color']}; padding: 18px; border-radius: 10px; color: white; margin-top: 15px;">
        <h2 style="margin: 0; color: white;">{bin_info['bin']} — {pred_class.upper()}</h2>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem;"><strong>Confidence:</strong> {confidence_pct:.2f}%</p>
        <p style="margin: 5px 0 0 0; font-size: 0.95rem;"><strong>Action:</strong> {bin_info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Class Confidence Breakdown
    st.subheader("📊 Class Confidence Percentages")
    for i, c in enumerate(CLASSES):
        prob_pct = probs[i].item() * 100
        col_name, col_bar, col_pct = st.columns([2, 5, 2])
        col_name.write(f"**{c.capitalize()}**")
        col_bar.progress(int(prob_pct))
        col_pct.write(f"{prob_pct:.2f}%")

# Show Accuracy Graph
if os.path.exists("accuracy_graph.png"):
    st.markdown("---")
    st.subheader("📈 Model Training Accuracy Graph")
    st.image("accuracy_graph.png", caption="Training vs Validation Accuracy across Epochs")

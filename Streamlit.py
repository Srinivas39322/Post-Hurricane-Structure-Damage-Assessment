#%%

import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# Page config
st.set_page_config(
    page_title="Hurricane Damage Classifier",
    layout="wide"
)

# Global CSS
st.markdown("""
<style>
    body, .stApp {
        background-color: #0e1117;
    }

    * {
        color: #FFFFFF !important;
    }

    .stButton>button {
        background-color: #1f77b4;
        color: white !important;
    }

    .stFileUploader {
        background-color: #1e1e1e;
        border: 1px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("POST-HURRICANE DAMAGE DETECTOR")
st.markdown("Upload one or more **aerial images** to check if structural damage is present.")

# Load model
@st.cache_resource
def load_model():
    try:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load("best_resnet50_model.pt", map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Upload multiple files
uploaded_files = st.file_uploader("📤 Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if model is None:
        st.error("❌ Model not loaded. Cannot predict.")
    else:
        with st.spinner("Running predictions..."):
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    img_tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        output = model(img_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        confidence, prediction = torch.max(probs, dim=1)

                    label = "🔴 Damaged" if prediction.item() == 0 else "🟢 Undamaged"
                    emoji = "⚠️" if prediction.item() == 0 else "✅"
                    conf_score = confidence.item()

                    # Log in session
                    st.session_state.history.append({
                        "Image": uploaded_file.name,
                        "Prediction": label.replace("🔴", "").replace("🟢", "").strip(),
                        "Confidence": f"{conf_score * 100:.2f}%"
                    })

                    # Display in columns (3 per row)
                    if idx % 3 == 0:
                        cols = st.columns(3)
                    col = cols[idx % 3]

                    with col:
                        st.image(image, caption=uploaded_file.name, width=250)
                        st.markdown(f"{emoji} **Prediction:** {label}")
                        st.metric("Confidence", f"{conf_score * 100:.2f}%")
                        st.progress(int(conf_score * 100))

                        if conf_score >= 0.9:
                            st.caption("✅ High confidence")
                        elif conf_score >= 0.6:
                            st.caption("⚠️ Moderate confidence")
                        else:
                            st.caption("❓ Low confidence")

                        # Optional transformed view
                        if st.checkbox(f"🔧 Show Transformed [{uploaded_file.name}]", key=f"trans_{idx}"):
                            transformed_image = to_pil_image(img_tensor.squeeze(0).clamp(0, 1))
                            st.image(transformed_image, caption="Model Input", width=250)

                        # Feedback
                        if st.button(f"👎 Report Issue [{uploaded_file.name}]", key=f"feedback_{idx}"):
                            st.warning("⚠️ Feedback noted for this image.")

                except Exception as e:
                    st.error(f"❌ Error with {uploaded_file.name}: {e}")

# Display full history
if st.session_state.history:
    st.markdown("### 🔁 Prediction History (This Session)")
    st.dataframe(st.session_state.history)

# Explanation
with st.expander("ℹ️ How does this work?"):
    st.write("""
        This tool uses a ResNet50 deep learning model trained on aerial images of hurricane-affected regions. 
        It predicts whether structures are damaged based on roof patterns and visual anomalies.

        The confidence score shows how certain the model is about each prediction.
    """)

# Placeholder for advanced model attention
with st.expander("🔍 View Model Attention (Coming Soon)"):
    st.markdown("This section will highlight areas the model focused on using techniques like Grad-CAM.")


#%%

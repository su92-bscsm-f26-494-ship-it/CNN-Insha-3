import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
from pathlib import Path

st.set_page_config(page_title="Image Brightness Classifier", page_icon="🖼️", layout="centered")

MODEL_PATH = Path(__file__).parent / "model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def extract_rgb_features(image):
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    return pd.DataFrame([{
        "mean_r": float(arr[:, :, 0].mean()),
        "mean_g": float(arr[:, :, 1].mean()),
        "mean_b": float(arr[:, :, 2].mean()),
    }])

st.title("🖼️ Image Brightness Classifier")
st.write("Upload an image to calculate its average RGB values and run the saved Random Forest model.")

st.warning(
    "Model limitation: the supplied trained model contains only the class 'Bright'. "
    "It cannot distinguish Bright from Dark. This is a brightness demonstration, "
    "not a skin-lesion or medical diagnostic tool."
)

if not MODEL_PATH.exists():
    st.error("model.joblib was not found. Add it to the same GitHub repository folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

classes = list(getattr(model, "classes_", []))
st.caption("Model type: Random Forest | Learned classes: " + (", ".join(map(str, classes)) or "not available"))

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        features = extract_rgb_features(image)
        st.subheader("Average RGB features")
        st.dataframe(features, hide_index=True, use_container_width=True)

        if len(classes) < 2:
            st.error(
                "Prediction is not reliable: this model was trained with only one class "
                f"({classes[0] if classes else 'unknown'}). Retrain it with correctly labelled "
                "examples from both classes before using it as a two-class classifier."
            )
        else:
            prediction = model.predict(features)[0]
            st.success(f"Model prediction: {prediction}")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                st.subheader("Class probabilities")
                st.dataframe(
                    pd.DataFrame({"Class": classes, "Probability": probabilities}),
                    hide_index=True,
                    use_container_width=True
                )
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")

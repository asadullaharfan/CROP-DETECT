
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 Crop Disease Detection System")
st.write("Identify possible plant diseases and explore disease information.")

# Load CSV database
DB_PATH = Path(__file__).parent / "crop_disease_database_english_urdu.csv"

@st.cache_data
def load_database():
    return pd.read_csv(DB_PATH, encoding="utf-8-sig")

try:
    df = load_database()
except Exception as e:
    st.error(f"Database load nahi hui: {e}")
    st.stop()

st.success(f"Database loaded: {len(df)} records")

# Image input
st.subheader("📷 Plant Image")

option = st.radio(
    "Choose image source:",
    ["Live Camera", "Upload Picture"],
    horizontal=True
)

image = None

if option == "Live Camera":
    camera_image = st.camera_input("Take a picture of a plant leaf")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
else:
    uploaded_image = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")

if image is not None:
    st.image(image, caption="Selected plant image", width=400)
    st.warning(
        "Image received. Automatic prediction requires a trained AI model."
    )

st.divider()

# Search disease database
st.subheader("🔎 Search Crop Disease Information")

search_text = st.text_input(
    "Enter crop or disease name",
    placeholder="Example: Wheat, Rice, Rust"
)

if search_text.strip():
    searchable = df.fillna("").astype(str)
    mask = searchable.apply(
        lambda col: col.str.contains(
            search_text.strip(), case=False, regex=False
        )
    ).any(axis=1)

    results = df[mask]

    if results.empty:
        st.warning("No matching records found. Try another name.")
    else:
        st.success(f"Found {len(results)} matching records")

        for _, row in results.head(20).iterrows():
            title = (
                str(row.iloc[0])
                if str(row.iloc[0]).strip()
                else "Disease information"
            )

            with st.expander(title):
                for column, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        st.markdown(f"**{column}:** {value}")
else:
    st.info("Search a crop or disease name to view its information.")

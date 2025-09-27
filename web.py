import streamlit as st
import pyttsx3
import threading
from PIL import Image
import numpy as np
import cv2
from tempfile import NamedTemporaryFile
import speech_recognition as sr
from tensorflow.keras.models import load_model
import os
import re

# ------------------ TTS ------------------ #
engine = pyttsx3.init()
def speak(text):
    threading.Thread(target=_speak, args=(text,)).start()

def _speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------ SPEECH INPUT ------------------ #
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak now")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results; check your connection.")
    return ""

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(page_title="Kisan Mitra AI", page_icon="🌾", layout="wide")

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
<style>
body {background-color: #f0f8ff;}
h1, h2, h3 {color: #2e7d32;}
.card {
    padding: 1rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    margin: 10px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------ #
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = "Home"
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'captured_file' not in st.session_state:
    st.session_state['captured_file'] = None

# ------------------ SIDEBAR ------------------ #
st.sidebar.image("logo.jpg", use_container_width=True)
st.sidebar.title("Kisan Mitra AI")
lang = st.sidebar.selectbox("🌐 Language", ["English", "తెలుగు (Telugu)", "हिंदी (Hindi)"])

app_mode = st.sidebar.radio(
    "Navigate",
    ["Home", "Crop Disease Recognition", "Crop/Disease Search", "About"],
    index=["Home","Crop Disease Recognition","Crop/Disease Search","About"].index(st.session_state['app_mode'])
)
st.session_state['app_mode'] = app_mode
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Upload leaf images, use webcam, or search crops for guidance.")

# ------------------ HELPER FUNCTIONS ------------------ #
def parse_crop_disease(text):
    crops = ["Corn","Wheat","Rice","Sugarcane","Potato"]
    diseases = ["Healthy","Disease1","Disease2"]
    crop_found = next((c for c in crops if re.search(c, text, re.I)), None)
    disease_found = next((d for d in diseases if re.search(d, text, re.I)), None)
    return crop_found or crops[0], disease_found or "Healthy"

# ------------------ LOAD AI MODEL ------------------ #
@st.cache_resource
def load_ai_model():
    model_path = "crop_disease_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.warning("AI model not found! Predictions will be random.")
        return None

model = load_ai_model()
class_names = ["Healthy","Disease1","Disease2"]

def predict_disease(image_path):
    if model:
        img = Image.open(image_path).resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        pred_class = class_names[np.argmax(preds)]
        return pred_class
    else:
        return np.random.choice(class_names)

# ------------------ HOME ------------------ #
if st.session_state['app_mode'] == "Home":
    st.image("logo.jpg", width=250)
    st.header("🌿 Welcome to Kisan Mitra AI!")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("📤 Upload Leaf"):
        st.session_state['app_mode'] = "Crop Disease Recognition"
        st.session_state['method'] = "upload"
        st.experimental_rerun()
    if col2.button("📷 Webcam Capture"):
        st.session_state['app_mode'] = "Crop Disease Recognition"
        st.session_state['method'] = "webcam"
        st.experimental_rerun()
    if col3.button("🔎 Search Crop/Disease"):
        st.session_state['app_mode'] = "Crop/Disease Search"
        st.experimental_rerun()
    
    st.subheader("📺 How to Use Kisan Mitra AI")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# ------------------ ABOUT ------------------ #
elif st.session_state['app_mode'] == "About":
    st.header("📚 About Kisan Mitra AI")
    st.markdown("---")
    st.write("Kisan Mitra AI detects crop diseases using AI models. Farmers can use voice, upload, or webcam input to identify diseases and get remedies.")

# ------------------ DISEASE RECOGNITION ------------------ #
elif st.session_state['app_mode']=="Crop Disease Recognition":
    st.header("🌾 Crop Disease Analysis 🔍")
    st.markdown("---")
    
    use_voice = st.checkbox("🎤 Use Voice Input")
    crop_list = ["Corn", "Wheat", "Rice", "Sugarcane", "Potato"]
    
    crop, disease = crop_list[0], "Healthy"
    
    if use_voice:
        voice_text = listen()
        crop, disease = parse_crop_disease(voice_text)
    
    if not use_voice:
        crop = st.selectbox("🌾 Select Crop", crop_list)
    
    method = st.session_state.get('method', 'upload')
    
    if method=="upload" and not use_voice:
        uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
    elif method=="webcam":
        if st.button("📷 Capture Leaf"):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                cap.release(); cv2.destroyAllWindows()
                tmp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(tmp_file.name, frame)
                st.session_state['captured_file'] = tmp_file.name
                st.image(frame, channels="BGR", caption="Captured Leaf")
    
    img_file = st.session_state.get('uploaded_file') or st.session_state.get('captured_file')
    
    if st.button("🚀 Start Analysis") or use_voice:
        st.info("🔍 Analyzing leaf patterns...")
        prediction = disease if use_voice else predict_disease(img_file) if img_file else "Healthy"
        
        st.subheader("📋 Diagnosis Report")
        if prediction == "Healthy":
            st.success(f"🎉 {crop} is healthy!")
            speak(f"{crop} is healthy!")
        else:
            st.error(f"⚠️ {prediction} detected in {crop}!")
            remedies_text = f"Recommended pesticide for {crop} with {prediction}. Tutorial online."
            speak(f"{crop} is affected by {prediction}. {remedies_text}")

        st.markdown(f"""
        <div class="card">
            <h3>Crop: {crop}</h3>
            <h3>Condition: {prediction}</h3>
        </div>
        """, unsafe_allow_html=True)

        if prediction != "Healthy":
            st.subheader("💊 Suggested Remedies")
            st.markdown(f"""
            - Recommended Pesticide: [Buy Online](https://www.amazon.in/s?k={crop}+{prediction}+pesticide)  
            - Tutorial: [YouTube](https://www.youtube.com/results?search_query={crop}+{prediction}+treatment)
            """)

# # ------------------ CROP/DISEASE SEARCH ------------------ #
elif st.session_state['app_mode']=="Crop/Disease Search":
    st.header("🔎 Crop/Disease Lookup")
    st.markdown("---")
    
    crops = ["Corn","Wheat","Rice","Sugarcane","Potato"]
    diseases = ["Healthy","Disease1","Disease2"]
    
    use_voice = st.checkbox("🎤 Use Voice Input")
    
    # Default selections
    crop_search, disease_search = crops[0], "Healthy"
    
    if use_voice:
        voice_text = listen()
        crop_search, disease_search = parse_crop_disease(voice_text)
    else:
        crop_search = st.selectbox("Select Crop", crops)
        disease_search = st.selectbox("Select Disease", diseases)
    
    # ------------------ CROP/DISEASE SEARCH ------------------ #
# ------------------ CROP/DISEASE SEARCH / VOICE ------------------ #
elif st.session_state['app_mode']=="Crop/Disease Search":
    st.header("🔎 Crop/Disease Lookup")
    st.markdown("---")
    
    crops = ["Corn","Wheat","Rice","Sugarcane","Potato"]
    diseases = ["Healthy","Disease1","Disease2"]
    
    use_voice = st.checkbox("🎤 Use Voice Input")
    
    crop_search, disease_search = crops[0], "Healthy"
    img_file = None
    
    if use_voice:
        voice_text = listen()
        crop_search, disease_search = parse_crop_disease(voice_text)
        st.info(f"Detected Crop: {crop_search}, Disease: {disease_search}")
        
        # Show default image for the spoken crop/disease
        image_path = f"images/{crop_search}_{disease_search}.jpg"
        if os.path.exists(image_path):
            img_file = Image.open(image_path)
            st.image(img_file, caption=f"{crop_search} - {disease_search}", use_column_width=True)
        else:
            st.warning(f"No default image found for {crop_search} - {disease_search}")
        
        # Automatically start analysis
        st.info("🔍 Starting Analysis...")
        if disease_search=="Healthy":
            st.success(f"{crop_search} is healthy! No treatment required.")
            speak(f"{crop_search} is healthy! No treatment required.")
        else:
            st.error(f"{disease_search} detected in {crop_search}!")
            remedies_text = f"Recommended pesticide for {crop_search} with {disease_search}. Tutorial online."
            st.markdown(f"""
            - Pesticide: [Buy Online](https://www.amazon.in/s?k={crop_search}+{disease_search}+pesticide)  
            - Tutorial: [YouTube](https://www.youtube.com/results?search_query={crop_search}+{disease_search}+treatment)
            """)
            speak(remedies_text)
            
    else:
        # Manual selection fallback
        crop_search = st.selectbox("Select Crop", crops)
        disease_search = st.selectbox("Select Disease", diseases)
        
        uploaded_file = st.file_uploader("📤 Upload Leaf Image (optional)", type=["jpg","png","jpeg"])
        if uploaded_file:
            img_file = uploaded_file
            st.image(img_file, caption=f"Uploaded Leaf for {crop_search} - {disease_search}", use_column_width=True)
        else:
            image_path = f"images/{crop_search}_{disease_search}.jpg"
            if os.path.exists(image_path):
                img_file = Image.open(image_path)
                st.image(img_file, caption=f"{crop_search} - {disease_search}", use_column_width=True)
            else:
                st.warning(f"No image found for {crop_search} - {disease_search}")
        
        if st.button("🔍 Search Info"):
            st.markdown(f"### 🌾 {crop_search} - {disease_search}")
            if disease_search=="Healthy":
                st.success(f"{crop_search} is healthy! No treatment required.")
                speak(f"{crop_search} is healthy! No treatment required.")
            else:
                st.error(f"{disease_search} detected in {crop_search}!")
                remedies_text=f"Recommended pesticide for {crop_search} with {disease_search}. Tutorial online."
                st.markdown(f"""
                - Pesticide: [Buy Online](https://www.amazon.in/s?k={crop_search}+{disease_search}+pesticide)  
                - Tutorial: [YouTube](https://www.youtube.com/results?search_query={crop_search}+{disease_search}+treatment)
                """)
                speak(remedies_text)

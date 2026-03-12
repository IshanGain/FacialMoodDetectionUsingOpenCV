import streamlit as st
import cv2
import cv2.face
import numpy as np
import zipfile
import time
import tempfile
import os
from PIL import Image
import io

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Facial Mood Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Syne', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* Dark background */
.stApp {
    background-color: #0d0d0d;
    color: #f0f0f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #00ff88, #00ccff, #ff00aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.main-header p {
    color: #666;
    font-size: 1rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* Emotion cards */
.emotion-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s;
}
.emotion-result {
    background: linear-gradient(135deg, #0a2a1a, #0a1a2a);
    border: 1px solid #00ff88;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
}
.emotion-name {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #00ff88;
}
.confidence-text {
    font-size: 0.9rem;
    color: #666;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: #1a1a1a;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00ccff;
}
.metric-label {
    font-size: 0.75rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00ccff);
    color: #000;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-size: 0.95rem;
    letter-spacing: 1px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,255,136,0.3);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1a1a;
    border: 2px dashed #333;
    border-radius: 16px;
    padding: 1rem;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 10px;
    color: #f0f0f0;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #00ff88, #00ccff);
    border-radius: 10px;
}

/* Divider */
hr { border-color: #222; }

/* Info boxes */
.stInfo {
    background: #0a1a2a;
    border: 1px solid #00ccff44;
    border-radius: 10px;
}

/* Image container */
.img-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #222;
}

/* Emotion emoji row */
.emoji-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 0.5rem;
    margin: 1rem 0;
}
.emoji-item {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 0.5rem;
    text-align: center;
    font-size: 0.7rem;
    color: #666;
}
.emoji-item.active {
    border-color: #00ff88;
    background: #0a2a1a;
    color: #00ff88;
}

/* Status badge */
.status-badge {
    display: inline-block;
    background: #0a2a1a;
    border: 1px solid #00ff88;
    color: #00ff88;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.75rem;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
EMOTION_EMOJIS = {"anger": "😠", "contempt": "😒", "disgust": "🤢",
                  "fear": "😨", "happy": "😊", "sadness": "😢", "surprise": "😲"}
EMOTION_COLORS = {"anger": "#ff4444", "contempt": "#ff8800", "disgust": "#88cc00",
                  "fear": "#aa44ff", "happy": "#ffcc00", "sadness": "#4499ff", "surprise": "#ff44aa"}
IMG_SIZE = (100, 100)
LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all three trained OpenCV face recognizers."""
    models = {}
    model_paths = {
        "LBPH":       "./model_lbph.yml",
        "EigenFace":  "./model_eigen.yml",
        "FisherFace": "./model_fisher.yml"
    }
    creators = {
        "LBPH":       cv2.face.LBPHFaceRecognizer_create,
        "EigenFace":  cv2.face.EigenFaceRecognizer_create,
        "FisherFace": cv2.face.FisherFaceRecognizer_create
    }
    for name, path in model_paths.items():
        if os.path.exists(path):
            rec = creators[name]()
            rec.read(path)
            models[name] = rec
    return models

@st.cache_resource
def load_cascade():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cascade

# ─── Core Functions ────────────────────────────────────────────────────────────
def detect_and_crop_face(gray_img, cascade):
    faces = cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None, []
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return gray_img[y:y+h, x:x+w], faces

def preprocess_face(face_roi):
    face = cv2.resize(face_roi, IMG_SIZE)
    face = cv2.equalizeHist(face)
    return np.array(face, dtype=np.uint8)

def predict_from_array(img_bgr, recognizer, cascade):
    """Run full pipeline on a BGR numpy image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_roi, faces = detect_and_crop_face(gray, cascade)
    if face_roi is None:
        face_roi = gray
    face = preprocess_face(face_roi)
    label, confidence = recognizer.predict(face)
    return EMOTIONS[label], confidence, faces

def draw_prediction_on_frame(frame, faces, mood, confidence):
    """Draw bounding box and label on frame."""
    display = frame.copy()
    color = tuple(int(EMOTION_COLORS[mood].lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # BGR
    for (x, y, w, h) in faces:
        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
        label = f"{EMOTION_EMOJIS[mood]} {mood.upper()} ({confidence:.1f})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(display, (x, y-35), (x+text_size[0]+10, y), color, -1)
        cv2.putText(display, label, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return display

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 FMD Settings")
    st.markdown("---")

    # Model selector
    st.markdown("### Model")
    models = load_models()
    if models:
        model_choice = st.selectbox(
            "Select Recognizer",
            list(models.keys()),
            index=0,
            help="LBPH achieves 98.96% accuracy on CK+48"
        )
        recognizer = models[model_choice]

        # Model accuracy info
        accuracies = {"LBPH": 98.96, "EigenFace": 96.37, "FisherFace": 94.82}
        acc = accuracies.get(model_choice, 0)
        st.markdown(f"""
        <div style='background:#0a2a1a;border:1px solid #00ff8844;border-radius:10px;padding:0.8rem;margin:0.5rem 0;'>
            <div style='color:#666;font-size:0.7rem;letter-spacing:2px;text-transform:uppercase;'>Test Accuracy</div>
            <div style='color:#00ff88;font-size:1.8rem;font-weight:800;'>{acc}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ No models found! Run the training notebook first.")
        st.info("Expected files:\n- model_lbph.yml\n- model_eigen.yml\n- model_fisher.yml")
        st.stop()

    st.markdown("---")

    # Emotion reference
    st.markdown("### Emotion Classes")
    for emotion in EMOTIONS:
        st.markdown(f"{EMOTION_EMOJIS[emotion]} **{emotion.capitalize()}**")

    st.markdown("---")
    st.markdown("""
    <div style='color:#444;font-size:0.75rem;text-align:center;'>
        CK+48 Dataset · OpenCV Only<br>
        981 images · 7 classes
    </div>
    """, unsafe_allow_html=True)

# ─── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>Facial Mood Detector</h1>
    <p>OpenCV · CK+48 · Real-time Inference</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "🎥 Webcam", "📊 Model Info"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    cascade = load_cascade()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Drop a face image here",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Best results with front-facing, well-lit photos"
        )

        if uploaded_file:
            # Read image
            file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### Prediction")

        if uploaded_file:
            with st.spinner("Analysing..."):
                mood, confidence, faces = predict_from_array(img_bgr, recognizer, cascade)
                display_bgr = draw_prediction_on_frame(img_bgr, faces, mood, confidence)
                display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

            # Show annotated image
            st.image(display_rgb, caption="Detected Face + Prediction", use_column_width=True)

            # Result card
            emoji = EMOTION_EMOJIS[mood]
            color = EMOTION_COLORS[mood]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d0d0d,#1a1a1a);
                        border:2px solid {color};border-radius:20px;
                        padding:1.5rem;text-align:center;margin-top:1rem;'>
                <div style='font-size:3rem;'>{emoji}</div>
                <div style='font-size:2rem;font-weight:800;color:{color};
                            letter-spacing:-1px;'>{mood.upper()}</div>
                <div style='color:#555;font-size:0.75rem;letter-spacing:2px;
                            text-transform:uppercase;font-family:monospace;
                            margin-top:0.5rem;'>
                    Confidence Score: {confidence:.2f}
                    {'(lower = better for LBPH)' if model_choice == 'LBPH' else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Face detection status
            if len(faces) > 0:
                st.success(f"✅ {len(faces)} face(s) detected by Haar Cascade")
            else:
                st.warning("⚠️ No face detected — used full image as fallback")

            # All emotions breakdown
            st.markdown("#### All Emotion Scores")
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            face_roi, _ = detect_and_crop_face(gray, cascade)
            if face_roi is None:
                face_roi = gray
            face_proc = preprocess_face(face_roi)

            scores = {}
            for name, model in models.items():
                if name == model_choice:
                    lbl, conf = model.predict(face_proc)
                    scores[EMOTIONS[lbl]] = conf

            # Show emotion bars
            for emotion in EMOTIONS:
                is_predicted = (emotion == mood)
                bar_color = EMOTION_COLORS[emotion] if is_predicted else "#333"
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;'>
                    <div style='width:80px;font-size:0.8rem;color:{"#fff" if is_predicted else "#666"};'>
                        {EMOTION_EMOJIS[emotion]} {emotion}
                    </div>
                    <div style='flex:1;background:#1a1a1a;border-radius:4px;height:8px;'>
                        <div style='width:{"100%" if is_predicted else "15%"};
                                    background:{bar_color};height:8px;
                                    border-radius:4px;transition:width 0.5s;'></div>
                    </div>
                    {"<div style='color:" + EMOTION_COLORS[emotion] + ";font-size:0.75rem;font-family:monospace;'>PREDICTED</div>" if is_predicted else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#111;border:2px dashed #222;border-radius:16px;
                        padding:3rem;text-align:center;color:#444;margin-top:2rem;'>
                <div style='font-size:3rem;'>🎭</div>
                <div style='margin-top:1rem;font-size:1rem;'>Upload an image to detect mood</div>
                <div style='font-size:0.8rem;margin-top:0.5rem;'>Supports JPG, PNG, BMP</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🎥 Live Webcam Detection")
    st.info("ℹ️ Streamlit's webcam captures single frames. For live video, use the notebook's `run_webcam_demo()` function.")

    cascade = load_cascade()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Webcam snapshot
        img_file_buffer = st.camera_input("Take a photo for mood detection")

        if img_file_buffer:
            # Read the captured image
            bytes_data = img_file_buffer.getvalue()
            file_bytes = np.frombuffer(bytes_data, dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with col2:
        st.markdown("### Result")
        if img_file_buffer:
            with st.spinner("Detecting mood..."):
                mood, confidence, faces = predict_from_array(img_bgr, recognizer, cascade)
                display_bgr = draw_prediction_on_frame(img_bgr, faces, mood, confidence)
                display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

            st.image(display_rgb, caption="Annotated Result", use_column_width=True)

            emoji = EMOTION_EMOJIS[mood]
            color = EMOTION_COLORS[mood]
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0d0d0d,#1a1a1a);
                        border:2px solid {color};border-radius:20px;
                        padding:1.5rem;text-align:center;margin-top:1rem;'>
                <div style='font-size:3rem;'>{emoji}</div>
                <div style='font-size:2rem;font-weight:800;color:{color};'>{mood.upper()}</div>
                <div style='color:#555;font-size:0.75rem;letter-spacing:2px;
                            text-transform:uppercase;font-family:monospace;margin-top:0.5rem;'>
                    Score: {confidence:.2f} · {len(faces)} face(s) detected
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#111;border:2px dashed #222;border-radius:16px;
                        padding:3rem;text-align:center;color:#444;margin-top:2rem;'>
                <div style='font-size:3rem;'>📷</div>
                <div style='margin-top:1rem;'>Click "Take Photo" to capture your expression</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Tips for Better Detection")
    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        st.markdown("""
        **For accurate results:**
        - Face the camera directly
        - Ensure good lighting (face a lamp/window)
        - Keep face centered in frame
        - Remove glasses if possible
        """)
    with tips_col2:
        st.markdown("""
        **Emotion expressions:**
        - 😠 Anger: furrowed brows, tight lips
        - 😊 Happy: broad smile
        - 😲 Surprise: raised brows, open mouth
        - 😢 Sadness: drooping features
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Model Performance")

    # Accuracy comparison
    col1, col2, col3 = st.columns(3)
    results = [
        ("LBPH", 98.96, 98.90, "#00ff88"),
        ("EigenFace", 96.37, 94.33, "#00ccff"),
        ("FisherFace", 94.82, 93.21, "#ff44aa"),
    ]
    for col, (name, acc, f1, color) in zip([col1, col2, col3], results):
        with col:
            st.markdown(f"""
            <div style='background:#1a1a1a;border:1px solid {color}44;
                        border-radius:16px;padding:1.5rem;text-align:center;'>
                <div style='color:#666;font-size:0.7rem;letter-spacing:2px;
                            text-transform:uppercase;'>{'⭐ BEST' if name=='LBPH' else name}</div>
                <div style='color:{color};font-size:2.5rem;font-weight:800;
                            margin:0.5rem 0;'>{acc}%</div>
                <div style='color:#444;font-size:0.8rem;'>Accuracy</div>
                <div style='color:{color};font-size:1.2rem;font-weight:700;
                            margin-top:0.5rem;'>{f1}%</div>
                <div style='color:#444;font-size:0.8rem;'>Macro F1</div>
                <div style='color:#333;font-size:0.7rem;margin-top:0.8rem;
                            font-family:monospace;'>{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Per-class results
    st.markdown("### LBPH Per-Class Performance")
    per_class = {
        "anger":    (0.964, 1.000, 0.982, 27),
        "contempt": (1.000, 1.000, 1.000, 10),
        "disgust":  (0.972, 1.000, 0.986, 35),
        "fear":     (1.000, 1.000, 1.000, 15),
        "happy":    (1.000, 0.976, 0.988, 41),
        "sadness":  (1.000, 0.938, 0.968, 16),
        "surprise": (1.000, 1.000, 1.000, 49),
    }
    for emotion, (prec, rec, f1, sup) in per_class.items():
        color = EMOTION_COLORS[emotion]
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:1rem;
                    padding:0.6rem 1rem;margin:0.3rem 0;
                    background:#111;border-radius:10px;border-left:3px solid {color};'>
            <div style='width:100px;font-weight:600;color:#ddd;'>
                {EMOTION_EMOJIS[emotion]} {emotion}
            </div>
            <div style='flex:1;'>
                <div style='background:#1a1a1a;border-radius:4px;height:6px;'>
                    <div style='width:{f1*100}%;background:{color};
                                height:6px;border-radius:4px;'></div>
                </div>
            </div>
            <div style='width:160px;display:flex;gap:1rem;
                        font-family:monospace;font-size:0.8rem;color:#666;'>
                <span>P:{prec:.3f}</span>
                <span>R:{rec:.3f}</span>
                <span style='color:{color};font-weight:700;'>F1:{f1:.3f}</span>
            </div>
            <div style='color:#444;font-size:0.75rem;font-family:monospace;
                        width:60px;text-align:right;'>n={sup}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Dataset info
    st.markdown("### Dataset: CK+48")
    dataset_info = {
        "anger": 135, "contempt": 54, "disgust": 177,
        "fear": 75, "happy": 207, "sadness": 84, "surprise": 249
    }
    cols = st.columns(7)
    for col, (emotion, count) in zip(cols, dataset_info.items()):
        with col:
            color = EMOTION_COLORS[emotion]
            st.markdown(f"""
            <div style='background:#111;border:1px solid {color}44;
                        border-radius:10px;padding:0.8rem;text-align:center;'>
                <div style='font-size:1.5rem;'>{EMOTION_EMOJIS[emotion]}</div>
                <div style='color:{color};font-weight:700;font-size:1.1rem;'>{count}</div>
                <div style='color:#444;font-size:0.65rem;margin-top:0.2rem;'>{emotion}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline
    st.markdown("### Pipeline")
    pipeline_steps = [
        ("01", "Load Image", "cv2.imread()", "#00ff88"),
        ("02", "Grayscale", "cv2.cvtColor()", "#00ccff"),
        ("03", "Haar Cascade", "detectMultiScale()", "#ff44aa"),
        ("04", "Resize", "cv2.resize() → 100×100", "#ffcc00"),
        ("05", "Equalise", "cv2.equalizeHist()", "#ff8800"),
        ("06", "Predict", "recognizer.predict()", "#00ff88"),
    ]
    step_cols = st.columns(6)
    for col, (num, title, api, color) in zip(step_cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div style='background:#111;border:1px solid {color}44;
                        border-radius:10px;padding:0.8rem;text-align:center;'>
                <div style='color:{color};font-size:1.2rem;font-weight:800;
                            font-family:monospace;'>{num}</div>
                <div style='color:#ddd;font-size:0.8rem;font-weight:600;
                            margin:0.3rem 0;'>{title}</div>
                <div style='color:#444;font-size:0.65rem;font-family:monospace;'>{api}</div>
            </div>
            """, unsafe_allow_html=True)
# 🎭 Facial Mood Detection Using OpenCV

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-98.96%25-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-CK%2B48-orange?style=for-the-badge)

**A facial expression recognition system built using only OpenCV — no deep learning frameworks.**

[🌐 Live Demo](https://facialmooddetectionusingopencv.onrender.com) · [📓 Notebook](fmd.ipynb) · [🖥️ Streamlit App](app.py) · [📦 Dataset](https://www.kaggle.com/datasets/shawon10/ckplus/data?select=CK%2B48)

</div>

---

## 📌 Overview

This project implements a **7-class facial expression classifier** using OpenCV's built-in face recognizers trained on the **CK+48 dataset**. Three classic algorithms — LBPH, EigenFace, and FisherFace — are compared, with LBPH achieving the highest accuracy of **98.96%**.

The project is deployed as a live **Streamlit web application** on Render, supporting both image upload and real-time webcam detection.

---

## 🎯 Emotion Classes

| Emotion | Emoji | Training Samples |
|---------|-------|-----------------|
| Anger | 😠 | 135 |
| Contempt | 😒 | 54 |
| Disgust | 🤢 | 177 |
| Fear | 😨 | 75 |
| Happy | 😊 | 207 |
| Sadness | 😢 | 84 |
| Surprise | 😲 | 249 |
| **Total** | | **981** |

---

## 📊 Results

### Model Accuracy Comparison

| Recognizer | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
|------------|--------------|-----------------|--------------|----------|
| **LBPH** ⭐ | **98.96%** | **99.09%** | **98.76%** | **98.90%** |
| EigenFace | 96.37% | 95.26% | 93.71% | 94.33% |
| FisherFace | 94.82% | 93.93% | 92.66% | 93.21% |

### LBPH Per-Class Performance (Best Model)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Anger | 0.964 | 1.000 | 0.982 | 27 |
| Contempt | 1.000 | 1.000 | 1.000 | 10 |
| Disgust | 0.972 | 1.000 | 0.986 | 35 |
| Fear | 1.000 | 1.000 | 1.000 | 15 |
| Happy | 1.000 | 0.976 | 0.988 | 41 |
| Sadness | 1.000 | 0.938 | 0.968 | 16 |
| Surprise | 1.000 | 1.000 | 1.000 | 49 |
| **Macro Avg** | **0.991** | **0.988** | **0.989** | **193** |

---

## 🏗️ Project Structure

```
FacialMoodDetectionUsingOpenCV/
│
├── fmd.ipynb                  # Main training notebook
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
│
├── model_lbph.yml             # Trained LBPH model (best)
├── model_eigen.yml            # Trained EigenFace model
├── model_fisher.yml           # Trained FisherFace model
│
├── accuracy_comparison.png    # Model accuracy bar chart
├── class_distribution.png    # Dataset distribution chart
├── confusion_matrices.png     # Confusion matrices (all 3 models)
├── per_class_f1.png           # Per-class F1 comparison chart
└── sample_faces.png           # Sample training faces
```

---

## ⚙️ Pipeline

```
Input Image
    │
    ▼
cv2.imread()          ← Load image
    │
    ▼
cv2.cvtColor()        ← Convert to grayscale
    │
    ▼
Haar Cascade          ← Detect face region (detectMultiScale)
    │
    ▼
cv2.resize()          ← Resize to 100×100 pixels
    │
    ▼
cv2.equalizeHist()    ← Histogram equalization
    │
    ▼
recognizer.predict()  ← LBPH / EigenFace / FisherFace
    │
    ▼
Emotion Label + Confidence Score
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/IshanGain/FacialMoodDetectionUsingOpenCV.git
cd FacialMoodDetectionUsingOpenCV
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the **CK+48** dataset ZIP (`FMDuOPCV.zip`) and place it in the project root.

---

## 📓 Training the Models

Open and run all cells in `fmd.ipynb`:

```bash
jupyter notebook fmd.ipynb
```

This will:
- Load the CK+48 dataset from ZIP
- Preprocess all images (grayscale → Haar crop → resize → equalize)
- Train LBPH, EigenFace, and FisherFace recognizers
- Evaluate and compare all 3 models
- Save trained models as `.yml` files
- Generate all output charts

**Training time:** ~30 seconds on standard hardware

---

## 🌐 Running the Streamlit App

```bash
streamlit run app.py
```

Or use the virtual environment directly:
```bash
.venv\Scripts\streamlit run app.py
```

The app will open at `http://localhost:8501`

### App Features
- **📸 Image Upload** — Upload any face photo for mood detection
- **🎥 Webcam** — Capture a photo from your webcam for live detection
- **📊 Model Info** — View accuracy comparison, per-class F1 scores, dataset stats, and pipeline diagram
- **🔄 Model Switcher** — Switch between LBPH, EigenFace, and FisherFace in the sidebar

---

## 🎥 Live Webcam Demo (Local Only)

For continuous real-time video detection, run this in the notebook:

```python
run_webcam_demo(recognizer=lbph, recognizer_name="LBPH", duration=60)
```

Press `q` in the popup window to stop.

---

## 🔍 Predict on a Single Image

```python
# Simple prediction
mood, conf = predict_mood(r"C:\path\to\image.jpg")
print(f"Predicted: {mood}, Confidence: {conf:.2f}")

# With visualization
mood, conf = visualise_prediction(r"C:\path\to\image.jpg")
```

---

## 📦 Dependencies

```
opencv-contrib-python==4.13.0
numpy
streamlit
pillow
```

---

## 🗂️ Dataset

**CK+ (Extended Cohn-Kanade) Dataset — 48×48 grayscale images**

- 981 total images across 7 emotion classes
- Lab-controlled conditions with posed expressions
- Train/Test split: 80/20 (788 train, 193 test)
- Stratified split to preserve class distribution

> ⚠️ The dataset ZIP is not included in this repository due to size. Please obtain it separately.

---

## 📈 Key Observations

- **LBPH outperforms** EigenFace and FisherFace by a significant margin on this dataset
- **Class imbalance** (54–249 samples per class) did not significantly affect LBPH performance due to its distance-based nature
- **Contempt, Fear, and Surprise** achieved perfect F1 scores despite having fewer training samples
- Real-world webcam performance is lower than lab accuracy due to differences in lighting and pose

---

## 🔮 Future Improvements

- Data augmentation to balance class distribution
- Deep learning comparison (CNN, ResNet)
- Multi-face detection support
- Real-time confidence threshold filtering

---

## 👨‍💻 Author

**Ishan Gain**  
GitHub: [@IshanGain](https://github.com/IshanGain)

---

## 📄 License

This project is for educational purposes as part of a capstone project.

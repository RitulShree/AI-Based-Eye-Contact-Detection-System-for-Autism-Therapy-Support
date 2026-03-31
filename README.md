# AI-Based-Eye-Contact-Detection-System-for-Autism-Therapy-Support
---

## 📌 Overview

This project is a **real-time computer vision system** that detects and quantifies **eye contact behavior** from video input. It is designed to support **Autism Spectrum Disorder (ASD) therapy** by providing **objective and measurable insights** into visual engagement.

⚠️ **Disclaimer:** This system is a **support tool only** and does NOT perform medical diagnosis.

---

## 🎯 Problem Statement

Children with ASD often face difficulty in maintaining eye contact, which is a key indicator of social interaction.

Traditional assessment methods are:

* Manual
* Subjective
* Inconsistent

This project automates the process using AI to generate **data-driven behavioral metrics**.

---

## ⚙️ System Pipeline

```
Video Input → Face Detection → Eye Landmark Extraction  
→ Gaze Estimation → Head Pose Estimation  
→ Eye Contact Classification → Behavioral Metrics
```

---

## 🧠 Key Features

### 👁️ Eye Contact Detection

* Combines **gaze direction + head orientation**
* Detects **true eye contact** only when both are aligned

---

### 👁️ Blink Analysis (EAR-Based)

* Blink count
* Blink rate (per minute)
* Eye closure duration
* Inter-Blink Interval (IBI)
* Blink variability

---

### 🎯 Gaze & Attention Metrics

* Eye movement tracking
* Gaze stability (variance-based)
* Fixation detection
* Fixation duration

---

### 📊 Session Metrics Output

* Eye contact time (seconds)
* Eye contact percentage (%)
* Average fixation duration
* Gaze stability score
* Longest no-blink interval

---

## 📸 Demo

![Demo Output](photos/img.png)

### 💾 Data Logging

* Session data saved to CSV
* Supports future machine learning training

---

## 🛠️ Tech Stack

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| Python              | Core development           |
| OpenCV              | Video capture & processing |
| MediaPipe Face Mesh | Facial landmark detection  |
| NumPy               | Mathematical computations  |
| Pandas              | Data handling              |
| Matplotlib          | Visualization              |

---

## 🧮 Core Concepts Used

* Eye Aspect Ratio (EAR) for blink detection
* Euclidean distance for eye movement tracking
* Sliding window analysis for gaze stability
* Variance & standard deviation for behavioral metrics
* Temporal smoothing to reduce noise

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```
pip install opencv-python mediapipe numpy pandas matplotlib
```

### 2️⃣ Run the project

```
python main.py
```

👉 Press `ESC` to exit manually
👉 Session auto-stops after predefined duration

---

## 📊 Sample Output

```
Session Duration: 60.00 seconds
Blink Rate: 18.2 per minute
Eye Contact: 42.5%
Fixations: 12
Avg Fixation Duration: 0.45 sec
```

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Performance affected by extreme head movement
* Depends on camera quality
* Rule-based thresholds may not generalize to all users
* Not clinically validated

---

## 🚀 Future Improvements

* Machine Learning-based classification
* Personalized calibration per user
* Real-time feedback system
* Dashboard for therapists
* Mobile/Web deployment

---

## 🤖 ML Extension (Planned)

### Available Features

* Gaze ratio
* Head direction
* Blink rate
* Fixation duration
* Movement patterns

### Models

* **Logistic Regression** → baseline, interpretable
* **Random Forest** → handles non-linearity, more robust

---

## 🎓 Research Scope

* Automated behavioral analysis
* Rule-based vs ML comparison
* Real-world robustness testing
* Suitable for academic publication

---

## 🙌 Acknowledgements

* MediaPipe by Google
* OpenCV community

---

## 📢 Final Note

This project transforms **raw visual behavior into structured data**, enabling more consistent and objective monitoring in therapy environments.

---

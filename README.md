# AI-Powered Helmet & License Plate Detection System

## 📌 Project Overview

This project is an AI-based traffic monitoring system that detects motorcycle riders without helmets and automatically recognizes their license plates. The system helps improve road safety and supports traffic authorities in identifying traffic rule violations.

Using computer vision and deep learning techniques, the system processes video frames to detect riders, identify helmet violations, and extract license plate information.

---

## 🎯 Objectives

* Detect riders without helmets using object detection models.
* Capture the vehicle license plate of violating riders.
* Use OCR to extract license plate numbers.
* Automate traffic rule violation monitoring.

---

## 🧠 Technologies Used

* Python
* OpenCV
* YOLOv8 (Object Detection)
* EasyOCR / Tesseract OCR
* NumPy
* Computer Vision
* Deep Learning

---

## ⚙️ System Workflow

1. Input traffic video or camera feed.
2. Detect motorcycles and riders using YOLOv8.
3. Check whether the rider is wearing a helmet.
4. If no helmet is detected, capture the vehicle region.
5. Detect the license plate in the captured frame.
6. Use OCR to extract the license plate number.
7. Display or store the detected violation details.

---

## 📂 Project Structure

AI-Powered-Helmet-License-Plate-Detection-System
│
├── detect_helmet.py
├── detect_license_plate.py
├── models/
├── utils/
├── sample_images/
├── requirements.txt
└── README.md

---

## 🚀 How to Run the Project

### 1. Clone the repository

git clone https://github.com/MudikeChandrakala/AI-Powered-Helmet-License-Plate-Detection-System.git

### 2. Navigate to the project folder

cd AI-Powered-Helmet-License-Plate-Detection-System

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run the detection script

python detect_helmet.py

---

## 📊 Applications

* Smart traffic monitoring systems
* Automated traffic violation detection
* Smart city surveillance
* Road safety enforcement

---

## 🔮 Future Improvements

* Real-time CCTV integration
* Automatic violation reporting system
* Database storage for violations
* Integration with traffic authority systems

---

## 👩‍💻 Author

Chandrakala Mudike
B.Tech – Information Technology

---

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pytesseract
import os
from PIL import Image

# --------------------------
# Tesseract OCR Setup
# --------------------------
def setup_tesseract():
    try:
        version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR detected: {version}")
        return True
    except (EnvironmentError, pytesseract.TesseractNotFoundError):
        if os.name == "nt":
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                st.success(f"Tesseract found at {tesseract_path}")
                return True
        st.warning("⚠ Tesseract OCR not found! Plate text won't be recognized.")
        st.info("Download from: https://github.com/tesseract-ocr/tesseract")
        return False

tesseract_available = setup_tesseract()

# --------------------------
# Load Models
# --------------------------
st.title("🛵 Helmet & Number Plate Detection System")

helmet_model = YOLO("yolov8n.pt")  # trained helmet/no-helmet model
plate_model = YOLO("yolov8n.pt")    # trained number plate model

# --------------------------
# Helper Functions
# --------------------------
def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def nms_boxes(boxes, scores, iou_thresh=0.4):
    """Apply Non-Maximum Suppression."""
    if len(boxes) == 0:
        return [], []

    indices = cv2.dnn.NMSBoxes(
        bboxes=[list(map(int, box)) for box in boxes],
        scores=[float(s) for s in scores],
        score_threshold=0.5,
        nms_threshold=iou_thresh
    )

    if len(indices) == 0:
        return [], []

    # flatten indices safely
    if isinstance(indices[0], (list, tuple, np.ndarray)):
        indices = [i[0] for i in indices]
    else:
        indices = list(indices)

    return [boxes[i] for i in indices], [scores[i] for i in indices]

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def associate_helmet_plate(helmet_boxes, plate_boxes):
    """Associate nearest number plate to each helmet box."""
    associations = []
    for hbox in helmet_boxes:
        hx, hy = get_center(hbox)
        closest_plate = None
        min_dist = 99999
        for pbox in plate_boxes:
            px, py = get_center(pbox)
            dist = np.sqrt((hx - px) ** 2 + (hy - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_plate = pbox
        associations.append((hbox, closest_plate))
    return associations

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Helmet detection ---
    helmet_results = helmet_model(frame_rgb)
    helmet_boxes, helmet_scores = [], []

    for res in helmet_results:
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            helmet_boxes.extend(boxes)
            helmet_scores.extend(scores)

    helmet_boxes, helmet_scores = nms_boxes(helmet_boxes, helmet_scores)

    # --- Plate detection ---
    plate_results = plate_model(frame_rgb)
    plate_boxes, plate_scores = [], []

    for res in plate_results:
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            plate_boxes.extend(boxes)
            plate_scores.extend(scores)

    plate_boxes, plate_scores = nms_boxes(plate_boxes, plate_scores)

    # --- Associate nearest plate to each helmet ---
    pairs = associate_helmet_plate(helmet_boxes, plate_boxes)

    # --- Draw boxes and labels ---
    for hbox, pbox in pairs:
        # Helmet box
        x1, y1, x2, y2 = map(int, hbox)
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "Helmet"
        color = (0, 255, 0)

        # Plate detection and OCR
        plate_text = "N/A"
        if tesseract_available and pbox is not None:
            px1, py1, px2, py2 = map(int, pbox)
            plate_crop = frame_rgb[py1:py2, px1:px2]
            if plate_crop.size > 0:
                plate_crop_proc = preprocess_plate(plate_crop)
                plate_text = pytesseract.image_to_string(
                    plate_crop_proc, config="--psm 8"
                ).strip()
                cv2.rectangle(frame_rgb, (px1, py1), (px2, py2), (255, 0, 0), 2)
                cv2.putText(
                    frame_rgb,
                    f"Plate: {plate_text}",
                    (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

        # Helmet label + plate text
        cv2.putText(
            frame_rgb,
            f"{label} | Plate: {plate_text}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    return frame_rgb

# --------------------------
# Streamlit Upload
# --------------------------
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_path = uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # --- Image processing ---
    if uploaded_file.type.startswith("image"):
        img = cv2.imread(file_path)
        output = process_frame(img)
        st.image(output, channels="RGB")
    else:
        # --- Video processing ---
        cap = cv2.VideoCapture(file_path)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            output = process_frame(frame)
            stframe.image(output, channels="RGB")
        cap.release()

    os.remove(file_path)

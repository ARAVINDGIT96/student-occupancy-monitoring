import cv2
import numpy as np
import streamlit as st
import time
from collections import deque

LOGO_TEXT = "STUDENT MONITORING & OCCUPANCY ANALYTICS"
st.set_page_config(layout="wide")
st.title(LOGO_TEXT)

source = st.radio("Choose video source", ["Webcam", "Upload Video"])
if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])
    if uploaded_file:
        source_choice = uploaded_file.name
        with open(source_choice, "wb") as f:
            f.write(uploaded_file.read())
    else:
        st.warning("Please upload a video file to continue.")
        st.stop()
else:
    source_choice = 0

cap = cv2.VideoCapture(source_choice)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    st.error("Error: Haar Cascade could not be loaded.")
    st.stop()

def define_zones(frame_width, frame_height):
    zone_width = frame_width // 3
    return [(i * zone_width, 0, zone_width, frame_height) for i in range(3)]

def detect_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

def count_people_in_zones(faces, zones):
    counts = [0] * len(zones)
    for (x, y, w, h) in faces:
        cx, cy = x + w // 2, y + h // 2
        for i, (zx, zy, zw, zh) in enumerate(zones):
            if zx <= cx < zx + zw and zy <= cy < zy + zh:
                counts[i] += 1
                break
    return counts

def draw_zones(frame, zones, counts):
    overlay = frame.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (x, y, w, h) in enumerate(zones):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), colors[i], 2)
        cv2.putText(overlay, f"Zone {i+1}: {counts[i]} people", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
    return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

def generate_heatmap(frame, heatmap_accum):
    heatmap = cv2.applyColorMap(cv2.convertScaleAbs(heatmap_accum, alpha=0.5), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

run = st.checkbox("▶ Start Monitoring")
show_heatmap = st.sidebar.checkbox("Show Heatmap Overlay", value=True)
col1, col2 = st.columns([1, 2])

history_length = 50
engagement_history = deque(maxlen=history_length)
OVER_CROWD_THRESHOLD = 2
LOW_ACTIVITY_THRESHOLD = 5
MAX_PEOPLE_PER_ZONE = 5

if run:
    video_placeholder = col2.empty()
    occupied_text = col1.empty()
    area_usage_text = col1.empty()
    crowd_status_text = col1.empty()
    alert_text = col1.empty()
    engagement_trend_text = col1.empty()
    heatmap_accum = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or cannot read frame.")
            break

        frame_height, frame_width = frame.shape[:2]
        zones = define_zones(frame_width, frame_height)
        faces = detect_people(frame)
        counts_per_zone = count_people_in_zones(faces, zones)
        total_occupied = sum(counts_per_zone)
        engagement_history.append(total_occupied)
        area_usages = [(count / MAX_PEOPLE_PER_ZONE) * 100 for count in counts_per_zone]

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame = draw_zones(frame, zones, counts_per_zone)

        if heatmap_accum is None:
            heatmap_accum = np.zeros((frame_height, frame_width), dtype=np.float32)

        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            if 0 <= cx < frame_width and 0 <= cy < frame_height:
                heatmap_accum[cy, cx] += 1

        heatmap_accum = cv2.GaussianBlur(heatmap_accum, (0, 0), sigmaX=15, sigmaY=15)
        heatmap_accum *= 0.95
        frame_with_heatmap = generate_heatmap(frame.copy(), heatmap_accum) if show_heatmap else frame.copy()

        crowd_status = "Normal"
        alert_msg = ""
        if total_occupied == 0:
            crowd_status = "Empty"
            alert_msg = "⚠ Alert: No activity detected!"
        elif total_occupied > OVER_CROWD_THRESHOLD:
            crowd_status = "Overcrowded"
            alert_msg = "⚠ Alert: Overcrowding detected!"
        elif total_occupied < LOW_ACTIVITY_THRESHOLD:
            crowd_status = "Low Crowded"
            alert_msg = "⚠ Alert: Low activity detected!"

        occupied_text.markdown(f"*Occupied Count:* {total_occupied}")
        area_usage_text.markdown(f"📈 *Zone Usage (%):* {', '.join(f'{u:.1f}%' for u in area_usages)}")
        crowd_status_text.markdown(f"📊 *Crowd Status:* {crowd_status}")
        alert_text.markdown(f"{alert_msg}" if alert_msg else "")
        engagement_trend_text.line_chart(list(engagement_history), height=150)
        video_placeholder.image(cv2.cvtColor(frame_with_heatmap, cv2.COLOR_BGR2RGB), channels="RGB")

        if source == "Upload Video":
            time.sleep(0.05)

    cap.release()
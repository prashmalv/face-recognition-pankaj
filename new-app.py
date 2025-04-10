import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile
import sqlite3
import pandas as pd
import datetime

# --- Setup ---
st.set_page_config(page_title="Missing Person Recognition", layout="wide")
st.title("🌟 Missing Person Face Recognition System")
st.write("Upload known images and a crowd image or video to find missing persons.")

# --- Load known faces ---
def load_known_faces(directory='known_faces'):
    known_encodings = []
    known_names = []
    for filename in os.listdir(directory):
        if filename.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(directory, filename)
            st.write(f"Loading known face: {image_path}")
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model='hog')
            st.write(f"Detected {len(face_locations)} face(s) in known image")
            if len(face_locations) == 0:
                st.warning(f"⚠️ No face detected in known image: {filename}")
            encoding = face_recognition.face_encodings(image, face_locations)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

# --- Logging to SQLite ---
def init_db():
    conn = sqlite3.connect('recognized_faces.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS recognitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    confidence REAL,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

def log_recognition(name, confidence, image_path):
    conn = sqlite3.connect('recognized_faces.db')
    c = conn.cursor()
    c.execute('INSERT INTO recognitions (name, confidence, image_path) VALUES (?, ?, ?)', (name, confidence, image_path))
    conn.commit()
    conn.close()

def display_log():
    conn = sqlite3.connect('recognized_faces.db')
    df = pd.read_sql_query('SELECT * FROM recognitions ORDER BY timestamp DESC', conn)
    conn.close()
    st.write(df)
    for _, row in df.iterrows():
        st.image(row['image_path'], caption=f"{row['name']} ({row['confidence']*100:.2f}%)", use_container_width=True)

# Initialize DB
init_db()

# Upload crowd image or video
crowd_file = st.file_uploader("Upload a crowd image or video", type=["jpg", "jpeg", "png", "mp4"])

# Load known faces from folder
if not os.path.exists("known_faces"):
    os.makedirs("known_faces")
st.info("Place known images in the 'known_faces' folder (with names as filenames)")

known_encodings, known_names = load_known_faces()

if known_encodings and crowd_file:
    if crowd_file.type.startswith("image"):
        image = face_recognition.load_image_file(crowd_file)
        face_locations = face_recognition.face_locations(image, model='hog')
        st.write(f"Detected {len(face_locations)} face(s) in uploaded image")
        face_encodings = face_recognition.face_encodings(image, face_locations)

        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        found = False

        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            confidence = 1 - distances[best_match_index]

            st.write("🔍 Distances from known faces:", distances)
            for idx, dist in enumerate(distances):
                conf = 1 - dist
                st.write(f"Compared with {known_names[idx]}: Confidence = {conf:.2f}")

            name = "Unknown"
            if confidence > 0.4:
                name = known_names[best_match_index]
                found = True
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                match_img_path = f"match_{name}_{timestamp}.jpg"
                match_img_crop = image[face_location[0]:face_location[2], face_location[3]:face_location[1]]
                cv2.imwrite(match_img_path, cv2.cvtColor(match_img_crop, cv2.COLOR_RGB2BGR))
                log_recognition(name, confidence, match_img_path)

            top, right, bottom, left = face_location
            label = f"{name} ({confidence*100:.1f}%)" if name != "Unknown" else f"No Match ({confidence*100:.1f}%)"
            draw.rectangle([left, top, right, bottom], outline="green", width=3)
            draw.text((left, top - 10), label, fill=(255, 0, 0))

        st.image(pil_img, caption="Processed Image", use_container_width=True)
        if found:
            st.success("✅ Person Found in Image!")
        else:
            if face_locations:
                st.warning("👤 Faces detected, but no match found.")
            else:
                st.warning("🚫 No faces detected in image.")

    elif crowd_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(crowd_file.read())

        video = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        found_in_video = False

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]

                name = "Unknown"
                if confidence > 0.4:
                    name = known_names[best_match_index]
                    found_in_video = True
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    match_img_path = f"match_{name}_{timestamp}.jpg"
                    match_img_crop = rgb_frame[face_location[0]:face_location[2], face_location[3]:face_location[1]]
                    cv2.imwrite(match_img_path, cv2.cvtColor(match_img_crop, cv2.COLOR_RGB2BGR))
                    log_recognition(name, confidence, match_img_path)

                top, right, bottom, left = face_location
                label = f"{name} ({confidence*100:.1f}%)" if name != "Unknown" else f"No Match ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

        video.release()
        os.unlink(tfile.name)

        if found_in_video:
            st.success("✅ Person Found in Video!")
        else:
            st.warning("❌ No match found in video.")

# Show recognition log
if st.button("Show Recognition Log"):
    display_log()

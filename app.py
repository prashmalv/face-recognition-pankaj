import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.title("🎯 Missing Person Face Recognition")
st.write("Upload a target image and a crowd video/photo to find the person.")

# Upload known image
known_img_file = st.file_uploader("Upload the image of the person to find", type=["jpg", "jpeg", "png"])
# Upload crowd video or image
crowd_file = st.file_uploader("Upload a crowd video or image", type=["jpg", "jpeg", "png", "mp4"])

if known_img_file and crowd_file:
    known_img = face_recognition.load_image_file(known_img_file)
    known_encodings = face_recognition.face_encodings(known_img)

    if not known_encodings:
        st.error("No face found in the target image.")
    else:
        known_encoding = known_encodings[0]

        # If image
        if crowd_file.type.startswith("image"):
            crowd_img = face_recognition.load_image_file(crowd_file)
            face_locations = face_recognition.face_locations(crowd_img)
            face_encodings = face_recognition.face_encodings(crowd_img, face_locations)

            pil_img = Image.fromarray(crowd_img)
            draw = ImageDraw.Draw(pil_img)

            found = False
            for face_encoding, face_location in zip(face_encodings, face_locations):
                match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                if match:
                    found = True
                    top, right, bottom, left = face_location
                    draw.rectangle([left, top, right, bottom], outline="green", width=4)

            st.image(pil_img, caption="Processed Image", use_column_width=True)
            if found:
                st.success("✅ Person Found in Image!")
            else:
                st.warning("❌ Person Not Found.")
        
        # If video
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
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
                    if match:
                        found_in_video = True
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                        cv2.putText(frame, "Person Found", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)

            video.release()
            os.unlink(tfile.name)

            if found_in_video:
                st.success("✅ Person Found in Video!")
            else:
                st.warning("❌ Person Not Found in Video.")

import face_recognition
import os
import pickle
from PIL import Image
import numpy as np

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "face_encodings.pkl"

known_encodings = []
known_names = []

for root, dirs, files in os.walk(KNOWN_FACES_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, file)
            name = os.path.basename(root)  # Folder name is used as person's name
            print(f"Encoding: {name} - {file}")

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)
            else:
                print(f"⚠️ No face found in {file}, skipping.")

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print(f"✅ Encoded {len(known_encodings)} faces and saved to '{ENCODINGS_FILE}'")

import face_recognition
import cv2
import pickle
import time

# Load known encodings
with open("face_encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Webcam setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Access Control", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Access Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

access_granted = False
last_access_time = 0
DISPLAY_DURATION = 3  # seconds

print("🔒 Face recognition access system started.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_names[best_match_index]
            access_granted = True
            last_access_time = time.time()

        top, right, bottom, left = [v * 4 for v in location]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show Access Status
    if access_granted and (time.time() - last_access_time) < DISPLAY_DURATION:
        cv2.putText(frame, "✅ Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    elif (time.time() - last_access_time) >= DISPLAY_DURATION:
        access_granted = False

    cv2.imshow("Face Access Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

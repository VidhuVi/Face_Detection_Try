from face_data import load_known_faces
from recognizer import recognize_faces_from_webcam

if __name__ == "__main__":
    print("[INFO] Loading known face data...")
    known_encodings, known_names = load_known_faces()

    print(f"[INFO] Loaded {len(known_names)} known face(s): {known_names}")
    recognize_faces_from_webcam(known_encodings, known_names)
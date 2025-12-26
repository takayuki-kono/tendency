import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

def check_pose(img_path):
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        return

    faces = app.get(img)
    if not faces:
        print("No face detected.")
        return

    print(f"Found {len(faces)} faces.")
    for i, face in enumerate(faces):
        print(f"--- Face {i} ---")
        # pose is usually [pitch, yaw, roll]
        # Pitch: Up/Down (Positive/Negative depends on implementation, usually + is up, - is down or vice versa)
        # Yaw: Left/Right
        # Roll: Tilt
        if face.pose is not None:
            pitch, yaw, roll = face.pose
            print(f"Pose: Pitch={pitch:.2f}, Yaw={yaw:.2f}, Roll={roll:.2f}")
        else:
            print("Pose is None")
            
        print(f"BBox: {face.bbox}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Default to the user provided example if available
        img_path = r"D:\tendency\train\aefh\harukaFukuhara\9732_390_0.jpg"
    
    print(f"Checking {img_path}...")
    check_pose(img_path)

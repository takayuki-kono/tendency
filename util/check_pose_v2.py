import cv2
import sys
import os
from insightface.app import FaceAnalysis

def check_pose(img_path):
    print(f"Checking {img_path}")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    
    img = cv2.imread(img_path)
    if img is None:
        with open("pose_result.txt", "w") as f:
            f.write(f"Error: Could not read {img_path}")
        return

    faces = app.get(img)
    with open("pose_result.txt", "w") as f:
        if not faces:
            f.write("No face detected.")
            return

        for i, face in enumerate(faces):
            if face.pose is not None:
                pitch, yaw, roll = face.pose
                f.write(f"Face {i}: Pitch={pitch}, Yaw={yaw}, Roll={roll}\n")
                print(f"Face {i}: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            else:
                f.write(f"Face {i}: Pose is None\n")

if __name__ == "__main__":
    img_path = r"D:\tendency\train\aefh\harukaFukuhara\9732_390_0.jpg"
    check_pose(img_path)

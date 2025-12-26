import cv2
from insightface.app import FaceAnalysis

img_path = r"D:\tendency\train\aefi\6650_052_0.jpg"

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

img = cv2.imread(img_path)
if img is None:
    print(f"Failed to read {img_path}")
else:
    faces = app.get(img)
    if not faces:
        print("No face detected")
    else:
        face = faces[0]
        if face.pose is not None:
            pitch, yaw, roll = face.pose
            print(f"Pose values:")
            print(f"  Pitch (上下): {pitch:.2f}")
            print(f"  Yaw (左右): {yaw:.2f}")
            print(f"  Roll (傾き): {roll:.2f}")
            print(f"\nCurrent filter:")
            print(f"  Pitch check: abs({pitch:.2f}) > 15 = {abs(pitch) > 15}")
            print(f"  Yaw check (not implemented): abs({yaw:.2f}) > ??")
        else:
            print("Pose is None")

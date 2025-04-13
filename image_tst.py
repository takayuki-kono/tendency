import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのセットアップ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 画像の読み込み（グレースケール）
image_path = "C:/tendency/train/category1/KoShibasaki_015.jpg"
image = cv2.imread(image_path)

if image is None:
    print("画像を読み込めませんでした")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # グレースケール変換
    img_height, img_width = gray.shape  # 高さと幅を取得
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 顔のランドマークを検出
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                print(idx)
                print(f"{landmark.x}, {landmark.y}")
                x, y = int(landmark.x * img_width), int(landmark.y * img_height)
                
                # 指定したランドマークのみ表示（例: 右目周り）
                if idx in [7, 33, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]: 
                    cv2.circle(gray, (x, y), 2, (255, 255, 255), -1)  # 白色で描画
                    cv2.putText(gray, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # 画像を表示
    cv2.imshow("Grayscale with Landmarks", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
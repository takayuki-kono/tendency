import cv2
import sys
import os
from insightface.app import FaceAnalysis

def draw_landmarks(image_path, output_path):
    # Initialize InsightFace
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Detect faces and landmarks
    faces = app.get(img)
    if not faces:
        print(f"No face detected in {image_path}")
        # Save the original image to the output path so the user sees something
        cv2.imwrite(output_path, img)
        return

    # Draw landmarks on the image
    # We'll use the first detected face
    face = faces[0]
    landmarks = face.landmark_2d_106

    # Create a copy of the image to draw on
    output_img = img.copy()

    # Draw each landmark and its index number
    for i in range(landmarks.shape[0]):
        p = tuple(landmarks[i].astype(int))
        # Draw the landmark point
        cv2.circle(output_img, p, 2, (0, 255, 0), -1) # Green circle
        # Draw the landmark index number
        cv2.putText(output_img, str(i), (p[0] + 2, p[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Save the image with landmarks
    cv2.imwrite(output_path, output_img)
    print(f"Image with landmarks saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_landmarks.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    # Define a fixed output path
    output_path = "landmarks_output.jpg"
    
    draw_landmarks(image_path, output_path)

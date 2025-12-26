
import os
import sys

# Add site-packages to path
sys.path.append('/mnt/d/tendency/.venv_new/lib/python3.12/site-packages')

print("--- Starting face_recognition debug script V2 ---")
print("--- This version also imports scikit-learn and scikit-image ---")

try:
    # Import the potentially conflicting libraries first
    from skimage.metrics import structural_similarity as ssim
    from sklearn.cluster import DBSCAN
    import face_recognition
    import cv2
    print("All libraries imported successfully.")
except ImportError as e:
    print(f"Failed to import a library: {e}")
    sys.exit(1)

# Use an image that was successfully processed earlier in the log
dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.isdigit()]
if not dirs:
    print("ERROR: Could not find the output directory (e.g., '0989').")
    sys.exit(1)

latest_dir = max(dirs, key=lambda d: os.path.getmtime(d))
# Let's find a valid image in the rotated directory
rotated_dir = os.path.join(latest_dir, "rotated")
rotated_images = [f for f in os.listdir(rotated_dir) if os.path.isfile(os.path.join(rotated_dir, f))]

if not rotated_images:
    print(f"ERROR: No images found in {rotated_dir}")
    sys.exit(1)

image_path = os.path.join(rotated_dir, rotated_images[0])

print(f"Attempting to load image: {image_path}")

if not os.path.exists(image_path):
    print(f"ERROR: Image not found at {image_path}")
else:
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: cv2.imread failed to load the image.")
        else:
            print("Image loaded successfully. Converting to RGB...")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            print("Calling face_recognition.face_locations with 'hog' model...")
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            print("--- SUCCESS! ---")
            print(f"face_recognition.face_locations ran without crashing.")
            print(f"Found {len(face_locations)} face(s) in the image.")

    except Exception as e:
        print(f"--- SCRIPT FAILED WITH AN EXCEPTION ---")
        import traceback
        traceback.print_exc()

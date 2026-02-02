import os
import cv2
import sys
import logging
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing InsightFace...")
    try:
        face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize InsightFace: {e}")
        return

    img_path = "image.jpg"
    if not os.path.exists(img_path):
        logger.error(f"Test image {img_path} not found.")
        return

    logger.info(f"Reading image {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        logger.error("Failed to read image.")
        return

    logger.info("Detecting faces...")
    try:
        faces = face_app.get(img)
        logger.info(f"Found {len(faces)} faces.")
        for i, face in enumerate(faces):
            logger.info(f"Face {i}: bbox={face.bbox}, score={face.det_score}")
    except Exception as e:
        logger.error(f"Detection failed: {e}")

if __name__ == "__main__":
    main()

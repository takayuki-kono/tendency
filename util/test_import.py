print("Start")
import os
print("Imported os")
import cv2
print("Imported cv2")
import numpy
print("Imported numpy")
from insightface.app import FaceAnalysis
print("Imported FaceAnalysis")
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
print("Initialized App")
app.prepare(ctx_id=0, det_size=(640, 640))
print("Prepared App")

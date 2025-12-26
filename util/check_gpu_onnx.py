import onnxruntime as ort
import sys

print(f"Python Executable: {sys.executable}")
print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

try:
    import tensorflow as tf
    print(f"TensorFlow Version (if installed): {tf.__version__}")
except ImportError:
    print("TensorFlow not installed in this environment.")

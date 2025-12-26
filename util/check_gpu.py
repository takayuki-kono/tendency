import tensorflow as tf
import sys

print("--- TensorFlow and GPU Check ---")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set successfully.")
        print("GPU(s) found and configured:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
    except RuntimeError as e:
        print(f"Error during GPU configuration: {e}")
else:
    print("No GPUs found by TensorFlow.")

print("---------------------------------")
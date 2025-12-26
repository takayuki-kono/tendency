import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
if not gpus:
    print("No GPUs found. Running on CPU.")
else:
    print(f"Found {len(gpus)} GPUs.")
    for gpu in gpus:
        print(f" - {gpu}")

import os
import sys
import dlib

print(f"--- Python Information ---")
print(f"Executable: {sys.executable}")
print(f"dlib version: {dlib.__version__}")
print(f"dlib file: {dlib.__file__}")

print(f"\n--- dlib CUDA Check ---")
print(f"dlib.DLIB_USE_CUDA: {dlib.DLIB_USE_CUDA}")
if dlib.DLIB_USE_CUDA:
    print(f"dlib.cuda.get_num_devices(): {dlib.cuda.get_num_devices()}")

print(f"\n--- System PATH Variable ---")
path_vars = os.environ.get('PATH', '').split(';')
print('\n'.join(path_vars))

print(f"\n--- Checking for CUDA/cuDNN in PATH ---")
cuda_found = any('cuda' in p.lower() for p in path_vars)
cudnn_found = any('cudnn' in p.lower() for p in path_vars)
print(f"CUDA path found in PATH: {cuda_found}")
print(f"cuDNN path found in PATH: {cudnn_found}")


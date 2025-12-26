
import onnxruntime
import numpy as np

print(f"ONNX Runtime version: {onnxruntime.__version__}")
print(f"NumPy version: {np.__version__}")

available_providers = onnxruntime.get_available_providers()
print(f"Available ONNX Runtime providers: {available_providers}")

if 'CUDAExecutionProvider' in available_providers:
    print("\nSUCCESS: GPU environment is configured correctly!")
    print("InsightFace should now be able to use your RTX 4060.")
else:
    print("\nERROR: CUDAExecutionProvider not found.")
    print("There might be an issue with the CUDA or cuDNN installation.")

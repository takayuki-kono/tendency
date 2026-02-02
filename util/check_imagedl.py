
import sys
import os

try:
    from imagedl import imagedl
    print("imagedl imported successfully")
    print(f"Location: {os.path.dirname(imagedl.__file__)}")
except ImportError:
    print("imagedl not found")

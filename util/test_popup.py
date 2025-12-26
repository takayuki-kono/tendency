import ctypes
import sys

print("Testing MessageBox...")
try:
    # 0x10 = MB_ICONERROR, 0x0 = MB_OK
    result = ctypes.windll.user32.MessageBoxW(0, "This is a test popup.\nIf you see this, MessageBox is working.", "Test Popup", 0x10)
    print(f"MessageBox returned: {result}")
except Exception as e:
    print(f"Error showing MessageBox: {e}")

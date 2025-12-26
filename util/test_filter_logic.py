
def check_filter(w, lx, rx, thresh_ratio=0.0625):
    center_x = w / 2.0
    diff_left_screen = (rx - center_x) * -1
    diff_right_screen = lx - center_x
    
    diff = abs(diff_left_screen - diff_right_screen)
    threshold = thresh_ratio * w
    
    print(f"W={w}, LX={lx}, RX={rx}")
    print(f"  Center={center_x}")
    print(f"  DiffLeft (rx) = {diff_left_screen}")
    print(f"  DiffRight (lx) = {diff_right_screen}")
    print(f"  AbsDiff = {diff}, Threshold = {threshold}")
    
    if diff >= threshold:
        print("  -> SKIPPED")
        return False
    else:
        print("  -> SAVED")
        return True

print("--- Test Case 1: Perfectly Centered ---")
# Width 100, Center 50. RX=10, LX=90.
# DL = (10-50)*-1 = 40. DR = 90-50 = 40. Diff=0.
check_filter(100, 90, 10)

print("\n--- Test Case 2: Slightly Off-Center (Within Threshold) ---")
# Width 100, Thresh=6.25.
# RX=12 (DL=38), LX=90 (DR=40). Diff=2. < 6.25.
check_filter(100, 90, 12)

print("\n--- Test Case 3: Significantly Off-Center (Right Shifted) ---")
# Face shifted right. RX=20 (DL=30), LX=98 (DR=48). Diff=18. > 6.25.
check_filter(100, 98, 20)

print("\n--- Test Case 4: Significantly Off-Center (Left Shifted) ---")
# Face shifted left. RX=2 (DL=48), LX=80 (DR=30). Diff=18. > 6.25.
check_filter(100, 80, 2)

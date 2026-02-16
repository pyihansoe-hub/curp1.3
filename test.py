import cv2

# Try different camera indices
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        success, frame = cap.read()
        if success:
            print(f"  Can read frames: Yes")
            print(f"  Frame size: {frame.shape if success else 'N/A'}")
        else:
            print(f"  Can read frames: No")
        cap.release()
    else:
        print(f"❌ No camera at index {i}")
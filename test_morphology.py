import cv2
import numpy as np

# Test morphology code
# Create a dummy eye_roi (grayscale image)
eye_roi = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

print("Original eye_roi shape:", eye_roi.shape)

# Apply blur
eye_roi = cv2.GaussianBlur(eye_roi, (7, 7), 0)
print("After blur")

# Threshold
_, threshold = cv2.threshold(eye_roi, 40, 255, cv2.THRESH_BINARY_INV)
print("After threshold")

# Apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
print("After morphology")

# Find contours
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Found contours:", len(contours))

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f"Pupil center: ({cx}, {cy})")
    else:
        print("No valid moments")
else:
    print("No contours found")

print("Morphology test completed successfully!")
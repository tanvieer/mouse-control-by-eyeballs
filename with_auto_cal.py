import cv2
import pyautogui
import numpy as np
import time

# Disable PyAutoGUI fail-safe for eye tracking
pyautogui.FAILSAFE = False

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Get screen size
screen_width, screen_height = pyautogui.size()

# Smoothing: list of recent mouse positions
recent_positions = []
smoothing_window = 15  # Increased for smoother movement

# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Calibration points
calibration_points = [(0, 0), (screen_width - 1, 0), (0, screen_height - 1), (screen_width - 1, screen_height - 1), (screen_width // 2, screen_height // 2)]
calibration_labels = ["top left", "top right", "bottom left", "bottom right", "center"]
calibration_data = []

print("Starting calibration. Look at each point as instructed.")

for i, (cx, cy) in enumerate(calibration_points):
    print(f"Look at {calibration_labels[i]} and press Enter when ready...")
    input()  # Wait for user to press enter

    print("Recording for 2 seconds...")
    start_time = time.time()
    ratios_x = []
    ratios_y = []

    while time.time() - start_time < 2:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h//2, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

            eye_centers = []
            for (ex, ey, ew, eh) in eyes:
                eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                eye_roi = cv2.GaussianBlur(eye_roi, (7, 7), 0)
                _, threshold = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M['m00'] != 0:
                        cx_eye = int(M['m10'] / M['m00'])
                        cy_eye = int(M['m01'] / M['m00'])
                        eye_centers.append((x + ex + cx_eye, y + ey + cy_eye))

            if eye_centers:
                avg_x = sum(c[0] for c in eye_centers) / len(eye_centers)
                avg_y = sum(c[1] for c in eye_centers) / len(eye_centers)

                eye_x_ratio = (avg_x - x) / w
                eye_y_ratio = (avg_y - y) / h

                eye_x_ratio = max(0, min(1, eye_x_ratio))
                eye_y_ratio = max(0, min(1, eye_y_ratio))

                ratios_x.append(eye_x_ratio)
                ratios_y.append(eye_y_ratio)

    if ratios_x and ratios_y:
        avg_ratio_x = sum(ratios_x) / len(ratios_x)
        avg_ratio_y = sum(ratios_y) / len(ratios_y)
        calibration_data.append((avg_ratio_x, avg_ratio_y, cx, cy))
        print(f"Recorded ratios: ({avg_ratio_x:.2f}, {avg_ratio_y:.2f}) for {calibration_labels[i]}")

print("Calibration complete.")

# Compute mapping ranges
if len(calibration_data) == 5:
    ratios_x = [d[0] for d in calibration_data]
    ratios_y = [d[1] for d in calibration_data]
    min_ratio_x = min(ratios_x)
    max_ratio_x = max(ratios_x)
    min_ratio_y = min(ratios_y)
    max_ratio_y = max(ratios_y)
else:
    # Fallback
    min_ratio_x, max_ratio_x = 0.3, 0.8
    min_ratio_y, max_ratio_y = 0.3, 0.5

# Hardcoded calibration values
min_ratio_x = 0.50
max_ratio_x = 0.56
min_ratio_y = 0.37
max_ratio_y = 0.40

print("Using hardcoded calibration values. Eye tracking mouse control started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Region of interest for eyes (upper half of face)
        roi_gray = gray[y:y + h//2, x:x + w]
        roi_color = frame[y:y + h//2, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi = cv2.GaussianBlur(eye_roi, (7, 7), 0)
            _, threshold = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    eye_centers.append((x + ex + cx, y + ey + cy))

        if eye_centers:
            # Average eye centers
            avg_x = sum(c[0] for c in eye_centers) / len(eye_centers)
            avg_y = sum(c[1] for c in eye_centers) / len(eye_centers)

            # Map relative to face bounding box for calibration
            eye_x_ratio = (avg_x - x) / w
            eye_y_ratio = (avg_y - y) / h

            # Clamp ratios to 0-1
            eye_x_ratio = max(0, min(1, eye_x_ratio))
            eye_y_ratio = max(0, min(1, eye_y_ratio))

            # Map to screen coordinates using calibration
            if max_ratio_x > min_ratio_x:
                mouse_x = int(((eye_x_ratio - min_ratio_x) / (max_ratio_x - min_ratio_x)) * screen_width)
            else:
                mouse_x = int(eye_x_ratio * screen_width)
            if max_ratio_y > min_ratio_y:
                mouse_y = int(((eye_y_ratio - min_ratio_y) / (max_ratio_y - min_ratio_y)) * screen_height)
            else:
                mouse_y = int(eye_y_ratio * screen_height)

            # Clamp
            mouse_x = max(0, min(screen_width - 1, mouse_x))
            mouse_y = max(0, min(screen_height - 1, mouse_y))

            # Add to recent positions for smoothing
            recent_positions.append((mouse_x, mouse_y))
            if len(recent_positions) > smoothing_window:
                recent_positions.pop(0)

            # Compute smoothed position
            if recent_positions:
                avg_x = sum(p[0] for p in recent_positions) / len(recent_positions)
                avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
                smoothed_x = int(avg_x)
                smoothed_y = int(avg_y)
            else:
                smoothed_x, smoothed_y = mouse_x, mouse_y

            # Print logs for calibration
            print(f"Eye centers: {eye_centers}, Avg: ({avg_x:.2f}, {avg_y:.2f}), Ratios: ({eye_x_ratio:.2f}, {eye_y_ratio:.2f}), Mouse: ({smoothed_x}, {smoothed_y})")

            # Move mouse with smooth animation
            pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.05)

        # Draw rectangle around face (optional)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
import cv2
import pyautogui
import numpy as np

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Eye tracking mouse control started. Press 'q' to quit.")

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

            # Map to screen coordinates
            mouse_x = int((avg_x / frame.shape[1]) * screen_width)
            mouse_y = int((avg_y / frame.shape[0]) * screen_height)

            # Move mouse
            pyautogui.moveTo(mouse_x, mouse_y)

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
# Eye Mouse Control

This project allows you to control your mouse cursor using eye tracking via your webcam.

## Requirements

- Python 3.x
- Webcam

## Installation

1. Create a virtual environment:
   ```
   python3 -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```
python eye_mouse_control.py
```

- The script will start with a calibration phase. Follow the on-screen instructions to look at 5 points on the screen (top left, top right, bottom left, bottom right, center).
- After calibration, your mouse cursor will move based on the detected eye position.
- Press 'q' to quit.

## How it works

- Uses OpenCV Haar cascades to detect faces and eyes.
- Within each detected eye region, applies image processing to find the pupil (darkest area).
- Calculates the center of the pupil(s) and averages them.
- Maps the pupil position relative to the face bounding box to screen coordinates (automatic calibration).
- Applies smoothing over multiple frames to reduce jittery movement.
- Moves the mouse cursor accordingly.

## Calibration

The script automatically calibrates by mapping eye positions relative to your detected face. To improve accuracy:
- Position your face centrally in the camera view.
- Ensure good lighting so eyes are clearly visible.
- Look around naturally; the mouse will move within the screen bounds based on your eye position within the face.

The script prints logs in the terminal showing eye center coordinates, ratios, and mouse positions for manual calibration checks.

## Troubleshooting

- Ensure your webcam is working and not used by other applications.
- Make sure you have good lighting for better detection.
- If the mouse movement is erratic, adjust the detection parameters in the code.
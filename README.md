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

- The script will start capturing video from your webcam.
- Your mouse cursor will move based on the detected eye position.
- Press 'q' to quit.

## How it works

- Uses OpenCV Haar cascades to detect faces and eyes.
- Calculates the center of the detected eye.
- Maps the eye position to screen coordinates.
- Moves the mouse cursor accordingly.

## Troubleshooting

- Ensure your webcam is working and not used by other applications.
- Make sure you have good lighting for better detection.
- If the mouse movement is erratic, adjust the detection parameters in the code.
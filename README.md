# Depth-Based Face Recognition

A real-time face recognition system using Intel RealSense depth cameras with liveness detection and PyQt5 GUI. This application combines 2D face recognition with 3D depth analysis to prevent spoofing attacks.

## Features

- **Real-time Face Recognition** – Detects and identifies faces using the `face_recognition` library
- **Depth-Based Liveness Detection** – Uses RealSense depth maps to detect spoofing attempts (2D photos vs. real faces)
- **Face Enrollment** – Enroll new faces into a persistent JSON database
- **PyQt5 GUI** – Full-screen video interface with real-time status display
- **Fast Mode** – Optimized for speed (424x240 @ 30fps) with optional detailed analysis mode
- **Intel RealSense D455** – Professional depth and RGB camera integration

## System Requirements

- **Hardware**: Intel RealSense D455 (or compatible RealSense camera)
- **Python**: 3.8 or higher
- **OS**: Windows / Linux / macOS

## Installation

1. **Clone/Download** this repository
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ensure RealSense SDK** is installed:
   - Download and install Intel RealSense SDK from: https://github.com/IntelRealSense/librealsense/releases
   - Or: `pip install pyrealsense2`

## Dependencies

- **pyrealsense2** ≥ 2.55.1 – Intel RealSense camera interface
- **opencv-python** ≥ 4.9.0 – Image processing
- **face_recognition** ≥ 1.3.0 – Face detection & embedding
- **dlib** ≥ 19.24 – Deep learning toolkit
- **numpy** 1.26.4 – Numerical computing
- **PyQt5** ≥ 5.15.11 – GUI framework

## Project Structure

```
├── camera.py              # RealSense D455 camera interface & frame capture
├── encoder.py             # Face detection and embedding extraction
├── database.py            # Face enrollment database (JSON storage)
├── liveness.py            # Depth-based liveness detection
├── gui_app.py             # PyQt5 GUI application (main entry point)
├── view_camera.py         # Debugging utility for camera streams
├── face_database.json     # Enrolled face embeddings (auto-generated)
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## Usage

### Run the Main GUI Application

```bash
python gui_app.py
```

This launches the full-screen PyQt5 GUI with:
- Real-time video feed from RealSense camera
- Live face detection and recognition
- Liveness detection status
- Face enrollment controls
- Recognized face display with match confidence

### Debug Camera Stream

To view raw camera output (color + depth):

```bash
python view_camera.py
```

## Module Overview

### `camera.py` – RealSense Interface
- `RealSenseCamera` class: Manages camera pipeline, frame capture, and depth data
- Supports configurable resolution (424x240 for fast mode, 640x480 for high-quality)
- Auto-recovery from camera errors
- Returns `FrameBundle` with color (BGR), depth (Z16 format), and depth scale

### `encoder.py` – Face Recognition
- `FaceEncoder` class: Detects faces and generates 128-D embeddings
- Configurable detection models: HOG (fast) or CNN (accurate)
- Support for small/large encoding models for speed/accuracy trade-offs
- Uses `face_recognition` library (dlib-based)

### `database.py` – Face Storage
- `FaceDatabase` class: Persists enrolled faces in JSON format
- Simple enrollment and matching with distance tolerance (default: 0.45)
- In-memory cache of all enrolled faces for fast lookups

### `liveness.py` – Anti-Spoofing
- `LivenessDetector` class: Analyzes depth maps to detect real vs. fake faces
- Checks depth range, variance, and sharpness
- Returns liveness decision with confidence score
- Prevents spoofing via printed photos or screen displays

### `gui_app.py` – Main Application
- `VideoWindow` class: Full-screen PyQt5 GUI
- Real-time processing pipeline with configurable frame skip
- Face enrollment UI with name input
- Status display (detected/recognized/liveness status)
- Keyboard controls for enrollment and exit

## Configuration

Edit **`gui_app.py`** to adjust:

```python
FAST_MODE = True              # True for 424x240 @ 30fps, False for 640x360
self.camera.width             # Frame width
self.camera.height            # Frame height
self.detect_every             # Process every N frames
```

Edit **`database.py`** to change:

```python
tolerance = 0.45              # Face match sensitivity (lower = stricter)
```

## Keyboard Controls (GUI)

- **E** – Enroll new face (enter name when prompted)
- **R** – Clear current recognition
- **Q** or **Esc** – Exit application

## Workflow

1. **Camera captures** color + depth frames (30 fps)
2. **Face detection** finds faces in RGB image
3. **Liveness check** analyzes depth data (reject if 2D photo)
4. **Face encoding** generates embedding for detected face
5. **Database matching** compares with enrolled faces
6. **Display result** shows recognized name or "Unknown"

## Known Limitations

- Requires RealSense D455 camera (depth camera essential for liveness detection)
- Liveness detection relies on adequate lighting and depth sensor performance
- Fast mode reduces accuracy for distant/small faces
- Face database grows as text (no size limit management)

## Troubleshooting

### Camera not detected
- Verify RealSense D455 is connected via USB 3.0
- Check: `python -c "import pyrealsense2; print('OK')"`
- Reinstall RealSense SDK if needed

### Low FPS
- Set `FAST_MODE = True` for 424x240 resolution
- Reduce `num_jitters` in `FaceEncoder` (currently 0 in fast mode)
- Close other applications using GPU

### Liveness detection failing
- Ensure adequate lighting on face
- Move closer to camera (better depth accuracy)
- Check depth sensor is not obstructed

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

## Contributors

[Your name/team]

## References

- Intel RealSense: https://github.com/IntelRealSense/librealsense
- face_recognition: https://github.com/ageitgey/face_recognition
- PyQt5: https://www.riverbankcomputing.com/software/pyqt/

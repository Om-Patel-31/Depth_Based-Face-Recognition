"""PyQt5 GUI for RealSense depth-based face recognition with optional fast mode."""
import sys
import time
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import face_recognition
from PyQt5 import QtCore, QtGui, QtWidgets

from camera import RealSenseCamera
from encoder import FaceEncoder
from database import FaceDatabase
from liveness import LivenessDetector

logger = logging.getLogger(__name__)


FAST_MODE = True  # set False to show landmarks/head pose (slower)


class VideoWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Depth Face ID")
        self.showFullScreen()

        # Core components
        # Balanced clarity and speed; drop to 424x240 for ~30fps when FAST_MODE
        if FAST_MODE:
            self.camera = RealSenseCamera(width=424, height=240, fps=30)
            self.detect_every = 2
        else:
            self.camera = RealSenseCamera(width=640, height=360, fps=30)
            self.detect_every = 3
        # Lightweight face encoder
        self.encoder = FaceEncoder(detection_model="hog", encoding_model="small", num_jitters=0)
        self.db = FaceDatabase()
        self.liveness = LivenessDetector()

        # UI elements
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #0e1116; border: 1px solid #1f252d;")
        self.video_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.status_label = QtWidgets.QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #cfd8e3; font-size: 13px;")

        self.db_label = QtWidgets.QLabel(f"Database: {len(self.db.names)} faces")
        self.db_label.setStyleSheet("color: #8ab4ff; font-size: 12px;")

        self.fps_label = QtWidgets.QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8ab4ff; font-size: 12px;")

        self.enroll_btn = QtWidgets.QPushButton("Enroll Live Face")
        self.enroll_btn.clicked.connect(self.enroll_face)
        self.enroll_btn.setCursor(QtCore.Qt.PointingHandCursor)

        self.clear_btn = QtWidgets.QPushButton("Clear Database")
        self.clear_btn.clicked.connect(self.clear_db)
        self.clear_btn.setCursor(QtCore.Qt.PointingHandCursor)

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.clicked.connect(self.close)
        self.quit_btn.setCursor(QtCore.Qt.PointingHandCursor)

        # Layout
        side_panel = QtWidgets.QVBoxLayout()
        side_panel.addWidget(self.status_label)
        side_panel.addSpacing(8)
        side_panel.addWidget(self.db_label)
        side_panel.addWidget(self.fps_label)
        side_panel.addSpacing(16)
        side_panel.addWidget(self.enroll_btn)
        side_panel.addWidget(self.clear_btn)
        side_panel.addWidget(self.quit_btn)
        side_panel.addStretch()

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.addWidget(self.video_label, stretch=6)
        layout.addLayout(side_panel, stretch=1)
        self.setCentralWidget(container)

        # Timer for video refresh
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS target

        # State
        self.last_live_embedding: Optional[Tuple[np.ndarray, str]] = None
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self._cached_locations = []
        self._cached_landmarks = []
        self._cam_mtx = None

    def set_status(self, text: str):
        self.status_label.setText(f"Status: {text}")

    def closeEvent(self, event):
        try:
            self.timer.stop()
            if self.camera:
                self.camera.release()
        finally:
            super().closeEvent(event)

    def clear_db(self):
        self.db.clear()
        self.db_label.setText(f"Database: {len(self.db.names)} faces")
        self.set_status("Database cleared")

    def enroll_face(self):
        if self.last_live_embedding is None:
            self.set_status("Enroll failed: no live face")
            return
        embedding, suggested_name = self.last_live_embedding
        name = suggested_name
        self.db.add(name, embedding)
        self.db_label.setText(f"Database: {len(self.db.names)} faces")
        self.set_status(f"Enrolled {name}")

    def update_frame(self):
        frame_bundle = self.camera.read()
        if frame_bundle is None:
            self.set_status("Waiting for camera...")
            return

        color = frame_bundle.color_bgr
        depth = frame_bundle.depth_z16
        h, w = color.shape[:2]
        if self._cam_mtx is None:
            fx = fy = 1.15 * max(w, h)
            cx, cy = w / 2.0, h / 2.0
            self._cam_mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Face detection (skipped on intermediate frames for speed)
        if self.frame_counter % self.detect_every == 0:
            face_locations = self.encoder.detect(color)
            frame_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            face_landmarks_list = [] if FAST_MODE else face_recognition.face_landmarks(frame_rgb)
            self._cached_locations = face_locations
            self._cached_landmarks = face_landmarks_list
        else:
            face_locations = self._cached_locations
            face_landmarks_list = self._cached_landmarks

        display = color.copy()

        self.last_live_embedding = None
        info_lines = []

        for idx, (top, right, bottom, left) in enumerate(face_locations):
            is_alive, live_conf, live_msg = self.liveness.is_alive(depth, (top, right, bottom, left), frame_bundle.depth_scale)
            box_color = (0, 255, 0) if is_alive else (0, 0, 255)
            cv2.rectangle(display, (left, top), (right, bottom), box_color, 2)

            # Head pose estimation if landmarks available and not in fast mode
            pose_text = None
            if not FAST_MODE and idx < len(face_landmarks_list):
                pose = self._estimate_head_pose(face_landmarks_list[idx], self._cam_mtx)
                if pose is not None:
                    yaw, pitch, roll = pose
                    pose_text = f"Yaw {yaw:.0f}° Pitch {pitch:.0f}° Roll {roll:.0f}°"
                    cv2.putText(display, pose_text, (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            embedding = self.encoder.encode(color, (top, right, bottom, left))
            label = "Unknown"
            match_dist = None

            if embedding is not None and is_alive:
                name, distance = self.db.match(embedding, tolerance=0.45)
                if name:
                    label = f"{name} ({distance:.2f})"
                    match_dist = distance
                else:
                    label = "Unknown (Live)"
                self.last_live_embedding = (embedding, f"Person_{len(self.db.names)+1}")

            cv2.putText(display, label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            status_text = f"LIVE {live_conf:.0%}" if is_alive else live_msg
            cv2.putText(display, status_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            extra = f" | dist={match_dist:.2f}" if match_dist else ""
            pose_info = f" | {pose_text}" if pose_text else ""
            info_lines.append(f"Face {idx+1}: {label} | {status_text}{extra}{pose_info}")

            # Landmarks overlay (skip in fast mode)
            if not FAST_MODE and len(face_locations) <= 2 and idx < len(face_landmarks_list):
                for pts in face_landmarks_list[idx].values():
                    for x, y in pts:
                        cv2.circle(display, (int(x), int(y)), 1, (0, 220, 255), -1)

        # Info panel overlay (compact)
        y0 = h - 70
        cv2.rectangle(display, (10, y0), (380, h - 10), (0, 0, 0), -1)
        for i, line in enumerate(info_lines[:2]):
            cv2.putText(display, line, (20, y0 + 25 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(display, f"DB: {len(self.db.names)}", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

        # FPS estimate
        self.frame_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            fps = self.frame_counter / (now - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_counter = 0
            self.last_fps_time = now

        # Convert to QImage
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        bytes_per_line = ch * w2
        qimg = QtGui.QImage(rgb.data, w2, h2, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        target_size = self.video_label.size()
        if not target_size.isEmpty():
            pix = pix.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)
        self.set_status("Live")

    def _estimate_head_pose(self, landmarks_dict, cam_mtx):
        # Map required facial points from face_recognition landmarks
        def pick(feature, idx):
            pts = landmarks_dict.get(feature)
            if pts and len(pts) > idx:
                return pts[idx]
            return None

        nose_tip = pick("nose_bridge", 3) or pick("nose_tip", 2)
        left_eye = pick("left_eye", 0)
        right_eye = pick("right_eye", 3)
        left_mouth = pick("top_lip", 0) or pick("bottom_lip", 0)
        right_mouth = pick("top_lip", 6) or pick("bottom_lip", 6)
        left_ear = pick("chin", 0)
        right_ear = pick("chin", 16)

        keypoints = [nose_tip, left_eye, right_eye, left_mouth, right_mouth, left_ear, right_ear]
        if any(p is None for p in keypoints):
            return None

        image_points = np.array(keypoints, dtype=np.float32)
        model_points = np.array([
            (0, 0, 0),
            (-30, -65, -30),
            (30, -65, -30),
            (-20, 20, -30),
            (20, 20, -30),
            (-45, 15, 0),
            (45, 15, 0),
        ], dtype=np.float32)
        dist_coeffs = np.zeros(4, dtype=np.float32)

        ok, rvec, _ = cv2.solvePnP(model_points, image_points, cam_mtx, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rmat, _ = cv2.Rodrigues(rvec)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        pitch = np.arcsin(-rmat[2, 0])
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        return np.degrees([yaw, pitch, roll])


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = VideoWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

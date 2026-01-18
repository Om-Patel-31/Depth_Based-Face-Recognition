"""Face detection and embedding extraction using dlib."""
import logging
from typing import Optional, List, Tuple
import numpy as np

try:
    import cv2
    import face_recognition
except ImportError:
    raise RuntimeError("Install face_recognition: pip install face_recognition")

logger = logging.getLogger(__name__)


class FaceEncoder:
    """Extract face embeddings from images."""
    
    def __init__(self, detection_model="hog", encoding_model="large", num_jitters: int = 3):
        """
        Args:
            detection_model: "hog" (fast) or "cnn" (more accurate, slower)
            encoding_model: "small" (fast) or "large" (more accurate)
            num_jitters: how many random jitters to apply for robust embeddings
        """
        self.detection_model = detection_model
        self.encoding_model = encoding_model
        self.num_jitters = num_jitters
        logger.info(
            f"FaceEncoder initialized (detect={detection_model}, encode={encoding_model}, jitters={num_jitters})"
        )
    
    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect face bounding boxes in frame.
        
        Returns:
            List of (top, right, bottom, left) boxes, or empty if no faces.
        """
        try:
            # Ensure uint8 format and convert BGR to RGB
            if frame_bgr.dtype != np.uint8:
                frame_bgr = (frame_bgr * 255).astype(np.uint8) if frame_bgr.max() <= 1 else frame_bgr.astype(np.uint8)
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(frame_rgb, model=self.detection_model)
            return face_locations
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def encode(self, frame_bgr: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Generate 128-D embedding for a detected face.
        
        Args:
            frame_bgr: Input image (H, W, 3)
            face_location: (top, right, bottom, left) bounding box
        
        Returns:
            128-D embedding vector or None if encoding fails.
        """
        try:
            # Encode expects RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(
                frame_rgb,
                [face_location],
                num_jitters=self.num_jitters,
                model=self.encoding_model,
            )
            if encoding:
                return encoding[0]
            return None
        except Exception as e:
            logger.error(f"Face encoding error: {e}")
            return None
    
    def detect_and_encode(self, frame_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """One-shot detect all faces and encode them.
        
        Returns:
            List of (embedding, bbox) tuples.
        """
        faces = []
        locations = self.detect(frame_bgr)
        
        for loc in locations:
            embedding = self.encode(frame_bgr, loc)
            if embedding is not None:
                faces.append((embedding, loc))
        
        return faces

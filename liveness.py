"""Liveness detection using depth analysis."""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Detect if face is real using depth map analysis."""
    
    def __init__(self):
        """Initialize liveness detector with strict thresholds."""
        pass
    
    def is_alive(self, depth_z16: np.ndarray, face_bbox: tuple, depth_scale: float) -> tuple:
        """Check if face is real (3D) or spoofed (2D photo).
        
        Args:
            depth_z16: Raw depth map from camera
            face_bbox: (top, right, bottom, left) bounding box
            depth_scale: Depth conversion factor
        
        Returns:
            (is_alive: bool, confidence: float, details: str)
        """
        top, right, bottom, left = face_bbox
        
        # Extract depth region
        face_depth = depth_z16[top:bottom, left:right]
        
        # Remove invalid depths (0 means no reading)
        valid_mask = face_depth > 0
        valid_depths = face_depth[valid_mask]
        
        if len(valid_depths) < 10:
            return False, 0.0, "No depth data"
        
        # Convert to meters
        depth_m = valid_depths * depth_scale
        
        # Get average distance (for adaptive thresholds)
        mean_depth = np.mean(depth_m)
        
        # Check 1: Depth range should be 2-5% of distance (adaptive)
        # Far away = need less absolute relief, Close up = more relief
        depth_range = depth_m.max() - depth_m.min()
        min_relief = max(0.02, mean_depth * 0.02)  # At least 2cm or 2% of distance
        max_relief = mean_depth * 0.15  # Up to 15% of distance
        
        if depth_range < min_relief:
            return False, 0.1, f"Insufficient relief ({depth_range*100:.1f}cm)"
        
        # Check 2: High variance required
        depth_var = np.var(depth_m)
        if depth_var < 0.0001:
            return False, 0.2, f"Uniform depth"
        
        # Check 3: Coefficient of variation (good at any distance)
        depth_std = np.std(depth_m)
        depth_cv = depth_std / mean_depth if mean_depth > 0 else 0
        
        if depth_cv < 0.05:  # Less strict than before
            return False, 0.3, f"Low variation (CV: {depth_cv:.3f})"
        
        # Check 4: Face contour - nose must be closer than edges
        h, w = face_depth.shape
        center_y, center_x = h // 2, w // 2
        
        nose_size = max(h // 4, 5)
        nose_region = face_depth[max(0, center_y - nose_size):min(h, center_y + nose_size),
                                  max(0, center_x - nose_size):min(w, center_x + nose_size)]
        
        # Sample edges
        edges = []
        if h > 10:
            edges.append(face_depth[0:5, :])  # Top
            edges.append(face_depth[-5:, :])  # Bottom
        if w > 10:
            edges.append(face_depth[:, 0:5])  # Left
            edges.append(face_depth[:, -5:])  # Right
        
        if not edges:
            return False, 0.0, "Face region too small"
        
        edge_depths = np.concatenate([e.flatten() for e in edges])
        edge_depths = edge_depths[edge_depths > 0]
        
        nose_valid = nose_region[nose_region > 0]
        
        if len(nose_valid) < 5 or len(edge_depths) < 5:
            return False, 0.0, "Insufficient samples"
        
        nose_mean = np.mean(nose_valid * depth_scale)
        edge_mean = np.mean(edge_depths * depth_scale)
        
        # Nose prominence should be at least 1% of distance (adaptive)
        min_prominence = max(0.01, mean_depth * 0.01)
        nose_prominence = edge_mean - nose_mean
        
        if nose_prominence < min_prominence:
            return False, 0.4, f"No contour ({nose_prominence*100:.1f}cm)"
        
        # All checks passed - real face!
        confidence = min(0.85 + depth_cv * 5, 1.0)
        return True, confidence, f"Real ({depth_range*100:.0f}cm relief, {nose_prominence*100:.0f}cm nose)"

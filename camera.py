"""Intel RealSense D455 camera interface."""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise RuntimeError("Install pyrealsense2: pip install pyrealsense2")

logger = logging.getLogger(__name__)

@dataclass
class FrameBundle:
    """Captured frame data."""
    color_bgr: np.ndarray  # (H, W, 3) BGR format
    depth_z16: np.ndarray  # (H, W) raw depth in Z16 format
    depth_scale: float      # meter/unit conversion factor


class RealSenseCamera:
    """RealSense D455 wrapper with error recovery."""
    
    def __init__(self, width=640, height=480, fps=30, timeout_ms=10000):
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.pipeline = None
        self.depth_scale = None
        self._connect()
    
    def _connect(self):
        """Initialize RealSense pipeline."""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            profile = self.pipeline.start(config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            
            logger.info(f"RealSense connected: {self.width}x{self.height} @ {self.fps}fps")
        except Exception as e:
            logger.error(f"RealSense connection failed: {e}")
            raise RuntimeError("Failed to connect to RealSense D455") from e
    
    def read(self) -> Optional[FrameBundle]:
        """Capture single synchronized frame pair."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                logger.warning("Frame dropped (incomplete pair)")
                return None
            
            color_bgr = np.asanyarray(color_frame.get_data())
            depth_z16 = np.asanyarray(depth_frame.get_data())
            
            return FrameBundle(
                color_bgr=color_bgr,
                depth_z16=depth_z16,
                depth_scale=self.depth_scale
            )
        except RuntimeError as e:
            logger.warning(f"Frame timeout ({self.timeout_ms}ms): {e}")
            return None
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            return None
    
    def release(self):
        """Cleanup."""
        if self.pipeline:
            self.pipeline.stop()
            logger.info("RealSense disconnected")

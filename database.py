"""Face database with JSON persistence."""
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    """Simple face enrollment storage and matching."""
    
    def __init__(self, db_path: str = "face_database.json"):
        self.db_path = Path(db_path)
        self.names = []
        self.embeddings = []
        self.load()
    
    def add(self, name: str, embedding: np.ndarray):
        """Enroll a new face."""
        self.names.append(name)
        self.embeddings.append(embedding.tolist())
        self.save()
        logger.info(f"Enrolled face: {name}")
    
    def match(self, embedding: np.ndarray, tolerance: float = 0.45) -> Tuple[Optional[str], Optional[float]]:
        """Match embedding against database.
        
        Returns:
            (name, distance) tuple, or (None, None) if no match found.
        """
        if not self.embeddings:
            return None, None
        
        # Compute distances to all enrolled faces
        distances = []
        for stored_embedding in self.embeddings:
            dist = np.linalg.norm(embedding - np.array(stored_embedding))
            distances.append(dist)
        
        min_distance = min(distances)
        best_idx = distances.index(min_distance)
        
        if min_distance < tolerance:
            return self.names[best_idx], min_distance
        
        return None, None
    
    def save(self):
        """Persist database to JSON."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump({
                    "names": self.names,
                    "embeddings": self.embeddings
                }, f)
            logger.info(f"Database saved: {len(self.names)} faces")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
    
    def load(self):
        """Load database from JSON."""
        if not self.db_path.exists():
            logger.info("No database file found, starting fresh")
            return
        
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.names = data.get("names", [])
                self.embeddings = data.get("embeddings", [])
            logger.info(f"Database loaded: {len(self.names)} faces")
        except Exception as e:
            logger.error(f"Database load failed: {e}")
    
    def clear(self):
        """Clear database."""
        self.names = []
        self.embeddings = []
        self.save()
        logger.info("Database cleared")

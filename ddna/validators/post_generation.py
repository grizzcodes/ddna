"""Post-generation validation."""

import logging
from typing import List, Dict
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class PostGenerationValidator:
    """Validates generated frames."""
    
    def validate(self, frames: List[Image.Image]) -> Dict:
        """Run post-generation checks."""
        
        results = {
            "total_frames": len(frames),
            "valid_frames": 0,
            "issues": []
        }
        
        for i, frame in enumerate(frames):
            if self._validate_frame(frame):
                results["valid_frames"] += 1
            else:
                results["issues"].append(f"Frame {i} validation failed")
                
        return results
    
    def _validate_frame(self, frame: Image.Image) -> bool:
        """Validate individual frame."""
        
        # Check resolution
        if frame.size[0] < 512 or frame.size[1] < 512:
            return False
            
        # Check if not blank
        arr = np.array(frame)
        if arr.std() < 1.0:  # Very low variation
            return False
            
        return True
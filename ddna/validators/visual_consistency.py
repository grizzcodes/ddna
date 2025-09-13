"""Visual consistency analysis."""

import numpy as np
from typing import List
from PIL import Image

class VisualConsistencyAnalyzer:
    def analyze(self, frames: List[Image.Image]) -> float:
        """Analyze consistency across frames."""
        if len(frames) < 2:
            return 1.0
        
        scores = []
        for i in range(1, len(frames)):
            score = self._compare_frames(frames[i-1], frames[i])
            scores.append(score)
        
        return np.mean(scores) if scores else 1.0
    
    def _compare_frames(self, frame1: Image.Image, frame2: Image.Image) -> float:
        """Compare two frames for consistency."""
        # Simple color histogram comparison
        hist1 = frame1.histogram()
        hist2 = frame2.histogram()
        
        # Calculate correlation
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return max(0, correlation)
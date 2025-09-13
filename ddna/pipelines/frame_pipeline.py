"""Single frame generation pipeline."""

import logging
from typing import Optional, Dict, Any
from PIL import Image

from ..core import SceneDNA
from ..generators import FrameGenerator

logger = logging.getLogger(__name__)

class FramePipeline:
    """Pipeline for generating single frames."""
    
    def __init__(self):
        self.generator = FrameGenerator()
        
    def generate_frame(self, 
                      dna: SceneDNA, 
                      frame_number: int = 0,
                      **kwargs) -> Image.Image:
        """Generate a single frame from DNA."""
        
        logger.info(f"Generating frame {frame_number} for {dna.scene_id}")
        
        frame = self.generator.generate(
            dna=dna,
            frame_number=frame_number,
            **kwargs
        )
        
        return frame
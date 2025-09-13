"""Composite generator for fallback generation."""

import logging
from typing import Dict, List, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class CompositeGenerator:
    """Fallback generator using composition techniques."""
    
    def __init__(self):
        self.generators = []
        
    def add_generator(self, generator, priority: int = 0):
        """Add a generator to the composite."""
        self.generators.append((priority, generator))
        self.generators.sort(key=lambda x: x[0], reverse=True)
        
    def generate(self, prompt: str, **kwargs) -> Optional[Image.Image]:
        """Try generators in priority order."""
        for priority, generator in self.generators:
            try:
                result = generator.generate(prompt, **kwargs)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Generator failed: {e}")
                continue
        
        # All failed, return placeholder
        return self._generate_placeholder(prompt)
    
    def _generate_placeholder(self, prompt: str) -> Image.Image:
        """Generate a placeholder image."""
        img = Image.new('RGB', (512, 512), 'gray')
        return img
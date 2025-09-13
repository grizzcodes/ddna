"""Pre-generation validation."""

import logging
from typing import Dict, List, Tuple

from ..core import SceneDNA

logger = logging.getLogger(__name__)

class PreGenerationValidator:
    """Validates scene before generation."""
    
    def validate(self, dna: SceneDNA) -> Tuple[bool, List[str]]:
        """Run pre-generation checks."""
        errors = []
        
        # Check required fields
        if not dna.scene_id:
            errors.append("Missing scene_id")
            
        if not dna.environment:
            errors.append("Missing environment")
            
        if not dna.style:
            errors.append("Missing style")
            
        # Check constraints compatibility
        if dna.style.genre == "sci_fi" and "horse" in dna.props:
            logger.warning("Anachronistic prop detected")
            
        return len(errors) == 0, errors
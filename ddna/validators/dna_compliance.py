"""DNA compliance validation."""

import logging
from typing import List, Dict
from ..core import SceneDNA

logger = logging.getLogger(__name__)

class DNAValidator:
    def validate(self, dna: SceneDNA) -> bool:
        errors = []
        
        if not dna.scene_id:
            errors.append("Missing scene_id")
        
        if not dna.environment:
            errors.append("Missing environment")
        elif not dna.environment.location:
            errors.append("Missing location")
        
        if not dna.style:
            errors.append("Missing style")
        
        if errors:
            logger.error(f"DNA validation failed: {errors}")
            return False
        
        return True
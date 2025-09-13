"""Module builder for pre-generating reusable components."""

import logging
from typing import Dict, Any
from PIL import Image
from ..core import SceneDNA

logger = logging.getLogger(__name__)

class ModuleBuilder:
    def build_modules(self, dna: SceneDNA) -> Dict[str, Any]:
        """Build reusable visual modules from DNA."""
        modules = {}
        
        # Build environment module
        modules['environment'] = self._build_environment(dna.environment)
        
        # Build character modules
        for char in dna.characters:
            modules[f'character_{char.id}'] = self._build_character(char)
        
        # Build prop modules
        for prop in dna.props:
            modules[f'prop_{prop}'] = self._build_prop(prop)
        
        logger.info(f"Built {len(modules)} modules")
        return modules
    
    def _build_environment(self, environment) -> Dict:
        return {
            'type': 'environment',
            'location': environment.location,
            'lighting': environment.lighting,
            'time': environment.time_of_day
        }
    
    def _build_character(self, character) -> Dict:
        return {
            'type': 'character',
            'id': character.id,
            'name': character.name,
            'appearance': character.appearance
        }
    
    def _build_prop(self, prop: str) -> Dict:
        return {
            'type': 'prop',
            'name': prop
        }
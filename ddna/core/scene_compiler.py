"""Scene compilation engine for assembling visual elements."""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .dna_parser import SceneDNA
from .constraint_manager import Constraint

logger = logging.getLogger(__name__)


@dataclass
class CompiledScene:
    """Compiled scene ready for generation."""
    base_prompt: str
    style_prompt: str
    negative_prompt: str
    constraints: Dict[str, Any]
    modules: Dict[str, Any]
    seed: Optional[int]
    metadata: Dict[str, Any]


class SceneCompiler:
    """Compiles Scene DNA into generation-ready format."""
    
    def __init__(self, style_presets: Optional[Dict] = None):
        """Initialize the scene compiler.
        
        Args:
            style_presets: Dictionary of style presets
        """
        self.style_presets = style_presets or {}
        self.composition_rules = self._load_composition_rules()
    
    def compile(self, 
                dna: SceneDNA, 
                modules: Optional[Dict] = None,
                constraints: Optional[List[Constraint]] = None) -> CompiledScene:
        """Compile Scene DNA into generation format.
        
        Args:
            dna: Scene DNA specification
            modules: Pre-built visual modules
            constraints: Active constraints
        
        Returns:
            CompiledScene ready for generation
        """
        logger.info(f"Compiling scene {dna.scene_id}")
        
        # Build base prompt from DNA
        base_prompt = self._build_base_prompt(dna)
        
        # Add style modifiers
        style_prompt = self._apply_style_modifiers(dna.style.genre)
        
        # Build negative prompt
        negative_prompt = self._build_negative_prompt(dna.style.genre)
        
        # Process constraints
        compiled_constraints = self._compile_constraints(constraints or [])
        
        # Process modules
        compiled_modules = self._compile_modules(modules or {})
        
        # Determine seed
        seed = dna.seed or self._generate_seed(dna.scene_id)
        
        return CompiledScene(
            base_prompt=base_prompt,
            style_prompt=style_prompt,
            negative_prompt=negative_prompt,
            constraints=compiled_constraints,
            modules=compiled_modules,
            seed=seed,
            metadata={
                "scene_id": dna.scene_id,
                "genre": dna.style.genre,
                "environment": dna.environment.location,
                "time_of_day": dna.environment.time_of_day,
            }
        )
    
    def _build_base_prompt(self, dna: SceneDNA) -> str:
        """Build base prompt from DNA elements."""
        prompt_parts = []
        
        # Environment description
        env = dna.environment
        env_desc = f"{env.location} scene"
        if env.time_of_day:
            env_desc += f" during {env.time_of_day}"
        if env.lighting:
            env_desc += f", {env.lighting}"
        if env.atmosphere:
            env_desc += f", {env.atmosphere} atmosphere"
        prompt_parts.append(env_desc)
        
        # Characters
        for char in dna.characters:
            char_desc = self._describe_character(char)
            if char_desc:
                prompt_parts.append(char_desc)
        
        # Props
        if dna.props:
            props_desc = f"visible props: {', '.join(dna.props)}"
            prompt_parts.append(props_desc)
        
        # Camera and composition
        if dna.style.camera_angle:
            prompt_parts.append(f"{dna.style.camera_angle} angle")
        if dna.style.shot_type:
            prompt_parts.append(dna.style.shot_type)
        
        return ", ".join(prompt_parts)
    
    def _describe_character(self, character: Any) -> str:
        """Generate character description for prompt."""
        desc_parts = [character.name]
        
        if hasattr(character, 'appearance') and character.appearance:
            desc_parts.append(character.appearance)
        
        if hasattr(character, 'position') and character.position:
            desc_parts.append(f"positioned {character.position}")
        
        if hasattr(character, 'action') and character.action:
            desc_parts.append(character.action)
        
        return ", ".join(desc_parts) if len(desc_parts) > 1 else desc_parts[0]
    
    def _apply_style_modifiers(self, genre: str) -> str:
        """Apply style-specific prompt modifiers."""
        if genre not in self.style_presets:
            return ""
        
        preset = self.style_presets.get(genre, {})
        modifiers = preset.get("prompt_modifiers", [])
        
        return ", ".join(modifiers)
    
    def _build_negative_prompt(self, genre: str) -> str:
        """Build negative prompt for style."""
        # Default negative prompts
        default_negative = [
            "low quality",
            "blurry",
            "distorted",
            "disfigured",
            "bad anatomy",
            "watermark",
            "text",
            "signature"
        ]
        
        # Add genre-specific negatives
        if genre in self.style_presets:
            preset = self.style_presets.get(genre, {})
            genre_negative = preset.get("negative_prompt", [])
            default_negative.extend(genre_negative)
        
        return ", ".join(default_negative)
    
    def _compile_constraints(self, constraints: List[Constraint]) -> Dict:
        """Compile constraints into generation format."""
        compiled = {
            "color_palette": [],
            "composition": {},
            "lighting": {},
            "character_positions": {},
        }
        
        for constraint in constraints:
            if constraint.type == "color":
                compiled["color_palette"].append(constraint.value)
            elif constraint.type == "composition":
                compiled["composition"].update(constraint.value)
            elif constraint.type == "lighting":
                compiled["lighting"].update(constraint.value)
            elif constraint.type == "character_position":
                compiled["character_positions"][constraint.target] = constraint.value
        
        return compiled
    
    def _compile_modules(self, modules: Dict) -> Dict:
        """Compile visual modules for use."""
        compiled = {}
        
        for module_name, module_data in modules.items():
            if isinstance(module_data, Image.Image):
                # Convert PIL Image to format needed for generation
                compiled[module_name] = {
                    "type": "image",
                    "data": module_data,
                    "size": module_data.size,
                }
            elif isinstance(module_data, np.ndarray):
                # Convert numpy array to PIL Image
                compiled[module_name] = {
                    "type": "array",
                    "data": Image.fromarray(module_data),
                    "shape": module_data.shape,
                }
            else:
                # Keep as-is for other types
                compiled[module_name] = module_data
        
        return compiled
    
    def _generate_seed(self, scene_id: str) -> int:
        """Generate deterministic seed from scene ID."""
        # Simple hash-based seed generation
        return abs(hash(scene_id)) % (2**32)
    
    def _load_composition_rules(self) -> Dict:
        """Load composition rules for scene assembly."""
        return {
            "rule_of_thirds": {
                "enabled": True,
                "grid_points": [(0.33, 0.33), (0.33, 0.67), (0.67, 0.33), (0.67, 0.67)]
            },
            "golden_ratio": {
                "enabled": False,
                "ratio": 1.618
            },
            "leading_lines": {
                "enabled": True,
                "strength": 0.7
            },
            "depth_layers": {
                "foreground": 0.2,
                "midground": 0.5,
                "background": 0.3
            }
        }
    
    def merge_with_previous(self, 
                           current: CompiledScene,
                           previous: CompiledScene,
                           blend_factor: float = 0.3) -> CompiledScene:
        """Merge current compilation with previous for continuity.
        
        Args:
            current: Current compiled scene
            previous: Previous compiled scene
            blend_factor: How much to blend (0=all current, 1=all previous)
        
        Returns:
            Merged CompiledScene
        """
        # Blend prompts
        merged_prompt = self._blend_prompts(
            current.base_prompt,
            previous.base_prompt,
            blend_factor
        )
        
        # Merge constraints
        merged_constraints = {**previous.constraints}
        for key, value in current.constraints.items():
            if key in merged_constraints:
                # Blend values where possible
                merged_constraints[key] = self._blend_values(
                    value, merged_constraints[key], blend_factor
                )
            else:
                merged_constraints[key] = value
        
        # Keep same seed for consistency
        seed = previous.seed
        
        return CompiledScene(
            base_prompt=merged_prompt,
            style_prompt=current.style_prompt,
            negative_prompt=current.negative_prompt,
            constraints=merged_constraints,
            modules=current.modules,
            seed=seed,
            metadata={**current.metadata, "blended": True}
        )
    
    def _blend_prompts(self, current: str, previous: str, factor: float) -> str:
        """Blend two prompts for continuity."""
        if factor == 0:
            return current
        elif factor == 1:
            return previous
        
        # Simple token-based blending
        current_tokens = set(current.split(", "))
        previous_tokens = set(previous.split(", "))
        
        # Keep common tokens and blend unique ones
        common = current_tokens & previous_tokens
        current_unique = current_tokens - common
        previous_unique = previous_tokens - common
        
        # Select tokens based on blend factor
        import random
        random.seed(42)  # Deterministic blending
        
        blended_tokens = list(common)
        
        for token in current_unique:
            if random.random() > factor:
                blended_tokens.append(token)
        
        for token in previous_unique:
            if random.random() < factor:
                blended_tokens.append(token)
        
        return ", ".join(blended_tokens)
    
    def _blend_values(self, current: Any, previous: Any, factor: float) -> Any:
        """Blend two values based on factor."""
        if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
            return current * (1 - factor) + previous * factor
        elif isinstance(current, list) and isinstance(previous, list):
            # Blend lists by selecting elements
            blended = []
            for i in range(max(len(current), len(previous))):
                if i < len(current) and i < len(previous):
                    # Choose based on factor
                    import random
                    random.seed(i)
                    if random.random() > factor:
                        blended.append(current[i])
                    else:
                        blended.append(previous[i])
                elif i < len(current):
                    blended.append(current[i])
                else:
                    blended.append(previous[i])
            return blended
        else:
            # For other types, choose based on factor
            import random
            random.seed(42)
            return previous if random.random() < factor else current
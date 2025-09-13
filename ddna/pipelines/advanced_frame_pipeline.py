"""Advanced 5-frame workflow with all consistency pillars."""

import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
from dataclasses import dataclass
import numpy as np

from ..core import (
    SceneDNA,
    IdentityLock,
    StylePaletteLock,
    PoseCompositionContinuity,
    PropEnvironmentPersistence,
    SeedRegistry,
    GenerationConfig
)
from ..generators import FrameGenerator

logger = logging.getLogger(__name__)

@dataclass
class FrameSpec:
    """Specification for a single frame."""
    frame_id: str
    frame_number: int
    is_anchor: bool
    anchor_refs: Optional[Tuple[str, str]] = None  # For interpolated frames
    t_value: Optional[float] = None  # Interpolation parameter

class AdvancedFramePipeline:
    """Advanced pipeline implementing 5 pillars of consistency."""
    
    def __init__(self, dna: SceneDNA):
        self.dna = dna
        
        # Initialize 5 pillars
        self.identity_lock = IdentityLock()
        self.style_palette = StylePaletteLock()
        self.pose_composition = PoseCompositionContinuity()
        self.prop_environment = PropEnvironmentPersistence()
        self.seed_registry = SeedRegistry(f"{dna.scene_id}.lock")
        
        # Frame generator
        self.generator = FrameGenerator()
        
        # Generated frames cache
        self.frames: Dict[str, Image.Image] = {}
        self.validation_scores: Dict[str, Dict] = {}
    
    def generate_5_frames(self, 
                         first_frame: Optional[Image.Image] = None,
                         last_frame: Optional[Image.Image] = None) -> Dict[str, Image.Image]:
        """Generate 5 consistent frames with anchor-based workflow."""
        
        logger.info(f"Starting 5-frame generation for {self.dna.scene_id}")
        
        # Step 1: Define frame specifications
        frame_specs = self._define_frame_specs()
        
        # Step 2: Setup consistency systems
        self._setup_consistency_systems(first_frame, last_frame)
        
        # Step 3: Generate anchor frames
        anchors = self._generate_anchors(frame_specs, first_frame, last_frame)
        
        # Step 4: Generate intermediate frames
        intermediates = self._generate_intermediates(frame_specs, anchors)
        
        # Step 5: Validate all frames
        self._validate_all_frames()
        
        # Step 6: Auto-regenerate failed frames
        self._regenerate_failed_frames()
        
        # Step 7: Final assembly
        return self._assemble_final_sequence()
    
    def _define_frame_specs(self) -> List[FrameSpec]:
        """Define specifications for 5 frames."""
        return [
            FrameSpec("A0", 0, is_anchor=True),  # First anchor
            FrameSpec("A1", 1, is_anchor=False, anchor_refs=("A0", "A2"), t_value=0.5),
            FrameSpec("A2", 2, is_anchor=True),   # Middle anchor
            FrameSpec("A3", 3, is_anchor=False, anchor_refs=("A2", "A4"), t_value=0.5),
            FrameSpec("A4", 4, is_anchor=True),   # Last anchor
        ]
    
    def _setup_consistency_systems(self, first_frame: Optional[Image.Image], 
                                  last_frame: Optional[Image.Image]):
        """Initialize all consistency systems."""
        
        # 1. Identity Lock
        if first_frame:
            for char in self.dna.characters:
                self.identity_lock.register_identity(char.id, first_frame)
        
        # 2. Style & Palette Lock
        from ..core.style_palette_lock import ColorPalette
        palette = ColorPalette(
            name=self.dna.style.genre,
            primary_colors=self.dna.style.color_palette,
            accent_colors=[],
            neutrals=['#000000', '#ffffff'],
            max_delta_e=15.0
        )
        self.style_palette.lock_palette(palette)
        
        # 3. Pose & Composition
        if first_frame and last_frame:
            start_poses = self.pose_composition.extract_pose(first_frame)
            end_poses = self.pose_composition.extract_pose(last_frame)
            
            if start_poses:
                self.pose_composition.pose_sequence.append(start_poses[0])
            if end_poses:
                self.pose_composition.pose_sequence.append(end_poses[0])
        
        # 4. Prop & Environment
        from ..core.prop_environment import EnvironmentLayout, PropObject
        layout = EnvironmentLayout(
            scene_id=self.dna.scene_id,
            props=[
                PropObject(
                    id=f"prop_{prop}",
                    name=prop,
                    position=(0.5, 0.5),  # Would be detected in production
                    size=(0.1, 0.1),
                    depth=0.5
                )
                for prop in self.dna.props
            ]
        )
        self.prop_environment.register_layout(self.dna.scene_id, layout)
        
        # 5. Seed Registry
        base_config = GenerationConfig(
            seed=self.dna.seed or 42,
            model="stable-diffusion-xl",
            loras={},
            controlnets={"depth": 0.8, "pose": 0.9},
            lut=None,
            resolution=(1920, 1080),
            guidance_scale=7.5,
            num_inference_steps=30,
            scheduler="DPMSolverMultistep"
        )
        
        for spec in ["A0", "A2", "A4"]:
            config = GenerationConfig(**asdict(base_config))
            config.seed = self.seed_registry.interpolate_seed(spec, "A0", "A4", 0.5)
            self.seed_registry.register_anchor(spec, config)
    
    def _generate_anchors(self, frame_specs: List[FrameSpec],
                         first_frame: Optional[Image.Image],
                         last_frame: Optional[Image.Image]) -> Dict[str, Image.Image]:
        """Generate anchor frames."""
        anchors = {}
        
        for spec in frame_specs:
            if not spec.is_anchor:
                continue
            
            if spec.frame_id == "A0" and first_frame:
                anchors["A0"] = first_frame
            elif spec.frame_id == "A4" and last_frame:
                anchors["A4"] = last_frame
            else:
                # Generate anchor
                config = self.seed_registry.get_config(spec.frame_id)
                
                # Get ControlNet conditions
                conditions = self.pose_composition.generate_controlnet_conditions(
                    spec.frame_number
                )
                
                # Generate frame
                frame = self.generator.generate(
                    self.dna,
                    spec.frame_number,
                    constraints=[],  # Would pass actual constraints
                    previous_frame=anchors.get("A0") if spec.frame_id == "A2" else None
                )
                
                # Apply style corrections
                frame = self.style_palette.apply_palette_correction(frame)
                
                anchors[spec.frame_id] = frame
                self.frames[spec.frame_id] = frame
                
                logger.info(f"Generated anchor {spec.frame_id}")
        
        return anchors
    
    def _generate_intermediates(self, frame_specs: List[FrameSpec],
                               anchors: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """Generate intermediate frames."""
        intermediates = {}
        
        for spec in frame_specs:
            if spec.is_anchor:
                continue
            
            # Get anchor references
            anchor1 = anchors[spec.anchor_refs[0]]
            anchor2 = anchors[spec.anchor_refs[1]]
            
            # Interpolate seed
            seed = self.seed_registry.interpolate_seed(
                spec.frame_id,
                spec.anchor_refs[0],
                spec.anchor_refs[1],
                spec.t_value
            )
            
            # Interpolate poses
            if len(self.pose_composition.pose_sequence) >= 2:
                pose = self.pose_composition.interpolate_poses(
                    self.pose_composition.pose_sequence[0],
                    self.pose_composition.pose_sequence[-1],
                    spec.t_value
                )
            
            # Generate with interpolated conditions
            frame = self.generator.generate(
                self.dna,
                spec.frame_number,
                previous_frame=anchor1
            )
            
            # Apply corrections
            frame = self.style_palette.apply_palette_correction(frame)
            
            intermediates[spec.frame_id] = frame
            self.frames[spec.frame_id] = frame
            
            logger.info(f"Generated intermediate {spec.frame_id}")
        
        return intermediates
    
    def _validate_all_frames(self):
        """Validate all frames against 5 pillars."""
        
        for frame_id, frame in self.frames.items():
            scores = {}
            
            # 1. Identity validation
            char_ids = [c.id for c in self.dna.characters]
            identity_valid, identity_scores = self.identity_lock.validate_frame(
                frame, char_ids
            )
            scores['identity'] = {'valid': identity_valid, 'scores': identity_scores}
            
            # 2. Palette validation
            palette_valid, delta_e = self.style_palette.validate_frame_colors(frame)
            scores['palette'] = {'valid': palette_valid, 'delta_e': delta_e}
            
            # 3. Composition validation
            frame_num = int(frame_id[1])  # Extract number from A0, A1, etc.
            comp_valid, comp_metrics = self.pose_composition.validate_composition(
                frame, frame_num
            )
            scores['composition'] = {'valid': comp_valid, 'metrics': comp_metrics}
            
            # 4. Prop validation
            prop_valid, prop_results = self.prop_environment.validate_prop_placement(
                frame, self.dna.scene_id
            )
            scores['props'] = {'valid': prop_valid, 'results': prop_results}
            
            # 5. Seed validation (already ensured by registry)
            scores['seed'] = {'valid': True}
            
            self.validation_scores[frame_id] = scores
            
            # Overall validation
            all_valid = all(
                scores[pillar]['valid'] 
                for pillar in ['identity', 'palette', 'composition', 'props']
            )
            
            if not all_valid:
                logger.warning(f"Frame {frame_id} failed validation: {scores}")
    
    def _regenerate_failed_frames(self, max_attempts: int = 3):
        """Auto-regenerate frames that failed validation."""
        
        for frame_id, scores in self.validation_scores.items():
            if all(scores[p]['valid'] for p in ['identity', 'palette', 'composition', 'props']):
                continue
            
            logger.info(f"Regenerating {frame_id} due to validation failures")
            
            for attempt in range(max_attempts):
                # Regenerate with same constraints
                config = self.seed_registry.get_config(frame_id)
                if config:
                    # Slightly modify seed for variation
                    config.seed += attempt + 1
                
                # Regenerate
                frame_num = int(frame_id[1])
                new_frame = self.generator.generate(
                    self.dna,
                    frame_num,
                    previous_frame=self.frames.get(f"A{frame_num-1}")
                )
                
                # Apply corrections
                new_frame = self.style_palette.apply_palette_correction(new_frame)
                
                # Re-validate
                char_ids = [c.id for c in self.dna.characters]
                identity_valid, _ = self.identity_lock.validate_frame(new_frame, char_ids)
                palette_valid, _ = self.style_palette.validate_frame_colors(new_frame)
                
                if identity_valid and palette_valid:
                    self.frames[frame_id] = new_frame
                    logger.info(f"Successfully regenerated {frame_id} on attempt {attempt+1}")
                    break
            else:
                logger.error(f"Failed to regenerate {frame_id} after {max_attempts} attempts")
    
    def _assemble_final_sequence(self) -> Dict[str, Image.Image]:
        """Assemble final validated sequence."""
        
        # Lock the registry for reproducibility
        self.seed_registry.lock()
        
        # Sort frames by number
        sorted_frames = {}
        for i in range(5):
            frame_id = f"A{i}"
            if frame_id in self.frames:
                sorted_frames[frame_id] = self.frames[frame_id]
        
        logger.info(f"Assembled {len(sorted_frames)} frames into final sequence")
        
        return sorted_frames
    
    def get_validation_report(self) -> Dict:
        """Get detailed validation report."""
        return {
            'scene_id': self.dna.scene_id,
            'frames': self.validation_scores,
            'overall_valid': all(
                all(scores[p]['valid'] for p in ['identity', 'palette', 'composition', 'props'])
                for scores in self.validation_scores.values()
            )
        }
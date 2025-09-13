"""Seed and constraint registry for reproducibility."""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Complete generation configuration."""
    seed: int
    model: str
    loras: Dict[str, float]
    controlnets: Dict[str, float]
    lut: Optional[str]
    resolution: tuple
    guidance_scale: float
    num_inference_steps: int
    scheduler: str
    
    def to_hash(self) -> str:
        """Generate deterministic hash of config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

class SeedRegistry:
    """Registry for seeds and generation configs."""
    
    def __init__(self, lock_file: str = "ddna.lock"):
        self.lock_file = Path(lock_file)
        self.registry: Dict[str, GenerationConfig] = {}
        self.frame_seeds: Dict[str, int] = {}
        self.locked: bool = False
        
        if self.lock_file.exists():
            self.load_lock_file()
    
    def register_anchor(self, frame_id: str, config: GenerationConfig):
        """Register anchor frame configuration."""
        config_hash = config.to_hash()
        self.registry[frame_id] = config
        self.frame_seeds[frame_id] = config.seed
        
        logger.info(f"Registered anchor {frame_id} with hash {config_hash}")
        
        if self.locked:
            self.save_lock_file()
    
    def get_config(self, frame_id: str) -> Optional[GenerationConfig]:
        """Get configuration for frame."""
        return self.registry.get(frame_id)
    
    def interpolate_seed(self, frame_id: str, anchor1: str, anchor2: str, t: float) -> int:
        """Interpolate seed between anchors."""
        if anchor1 in self.frame_seeds and anchor2 in self.frame_seeds:
            seed1 = self.frame_seeds[anchor1]
            seed2 = self.frame_seeds[anchor2]
            
            # Deterministic interpolation
            combined = f"{seed1}_{seed2}_{t:.3f}_{frame_id}"
            return int(hashlib.md5(combined.encode()).hexdigest(), 16) % (2**32)
        
        # Fallback to random seed
        return int(hashlib.md5(frame_id.encode()).hexdigest(), 16) % (2**32)
    
    def lock(self):
        """Lock the registry and save to file."""
        self.locked = True
        self.save_lock_file()
        logger.info(f"Registry locked with {len(self.registry)} configs")
    
    def unlock(self):
        """Unlock the registry."""
        self.locked = False
        logger.info("Registry unlocked")
    
    def save_lock_file(self):
        """Save registry to lock file."""
        lock_data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'locked': self.locked,
            'configs': {k: asdict(v) for k, v in self.registry.items()},
            'frame_seeds': self.frame_seeds
        }
        
        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        logger.info(f"Saved lock file: {self.lock_file}")
    
    def load_lock_file(self):
        """Load registry from lock file."""
        with open(self.lock_file, 'r') as f:
            lock_data = json.load(f)
        
        self.locked = lock_data.get('locked', False)
        self.frame_seeds = lock_data.get('frame_seeds', {})
        
        for frame_id, config_dict in lock_data.get('configs', {}).items():
            self.registry[frame_id] = GenerationConfig(**config_dict)
        
        logger.info(f"Loaded {len(self.registry)} configs from lock file")
    
    def verify_reproducibility(self, frame_id: str, generated_seed: int) -> bool:
        """Verify that generation used correct seed."""
        if frame_id not in self.frame_seeds:
            return True  # No constraint
        
        expected = self.frame_seeds[frame_id]
        if generated_seed != expected:
            logger.error(f"Seed mismatch for {frame_id}: expected {expected}, got {generated_seed}")
            return False
        
        return True
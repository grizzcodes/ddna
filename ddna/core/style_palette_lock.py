"""Style and color palette locking system."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageFilter
from dataclasses import dataclass
import logging
from skimage.color import deltaE_ciede2000, rgb2lab

logger = logging.getLogger(__name__)

@dataclass
class ColorPalette:
    """Color palette specification."""
    name: str
    primary_colors: List[str]
    accent_colors: List[str]
    neutrals: List[str]
    lut_path: Optional[str] = None
    max_delta_e: float = 10.0

class StylePaletteLock:
    """Enforces consistent style and color palette across frames."""
    
    def __init__(self):
        self.active_palette: Optional[ColorPalette] = None
        self.style_embeddings: Dict[str, np.ndarray] = {}
        self.lut_cache: Dict[str, np.ndarray] = {}
        
    def lock_palette(self, palette: ColorPalette):
        """Lock a color palette for the scene."""
        self.active_palette = palette
        logger.info(f"Locked palette: {palette.name}")
        
        if palette.lut_path:
            self._load_lut(palette.lut_path)
    
    def validate_frame_colors(self, frame: Image.Image) -> Tuple[bool, float]:
        """Validate frame against locked palette using Delta E."""
        if not self.active_palette:
            return True, 0.0
        
        # Extract frame colors
        frame_colors = self._extract_frame_colors(frame)
        
        # Convert to LAB for Delta E calculation
        palette_lab = self._hex_to_lab(self.active_palette.primary_colors)
        frame_lab = self._rgb_to_lab(frame_colors)
        
        # Calculate minimum Delta E for each frame color
        min_distances = []
        for fc in frame_lab:
            distances = [deltaE_ciede2000(fc, pc) for pc in palette_lab]
            min_distances.append(min(distances))
        
        avg_delta_e = np.mean(min_distances)
        
        passed = avg_delta_e <= self.active_palette.max_delta_e
        if not passed:
            logger.warning(f"Color validation failed: ΔE={avg_delta_e:.2f} > {self.active_palette.max_delta_e}")
        
        return passed, avg_delta_e
    
    def apply_palette_correction(self, frame: Image.Image) -> Image.Image:
        """Apply color correction to match palette."""
        if not self.active_palette:
            return frame
        
        # Apply LUT if available
        if self.active_palette.lut_path and self.active_palette.name in self.lut_cache:
            frame = self._apply_lut(frame, self.lut_cache[self.active_palette.name])
        
        # Additional color grading
        frame = self._grade_to_palette(frame)
        
        return frame
    
    def _extract_frame_colors(self, frame: Image.Image, n_colors: int = 10) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from frame."""
        # Resize for faster processing
        thumb = frame.resize((150, 150))
        pixels = np.array(thumb).reshape(-1, 3)
        
        # K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(c) for c in colors]
    
    def _hex_to_lab(self, hex_colors: List[str]) -> np.ndarray:
        """Convert hex colors to LAB color space."""
        rgb_colors = []
        for hex_color in hex_colors:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            rgb_colors.append(rgb)
        
        rgb_array = np.array(rgb_colors) / 255.0
        return rgb2lab(rgb_array.reshape(-1, 1, 3)).reshape(-1, 3)
    
    def _rgb_to_lab(self, rgb_colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """Convert RGB colors to LAB color space."""
        rgb_array = np.array(rgb_colors) / 255.0
        return rgb2lab(rgb_array.reshape(-1, 1, 3)).reshape(-1, 3)
    
    def _load_lut(self, lut_path: str):
        """Load a 3D LUT file for color grading."""
        # Placeholder - would load actual LUT (cube file) in production
        logger.info(f"Loading LUT from {lut_path}")
        self.lut_cache[self.active_palette.name] = np.random.randn(32, 32, 32, 3)
    
    def _apply_lut(self, frame: Image.Image, lut: np.ndarray) -> Image.Image:
        """Apply 3D LUT to frame."""
        # Placeholder - would apply actual LUT transformation
        return frame
    
    def _grade_to_palette(self, frame: Image.Image) -> Image.Image:
        """Grade frame colors toward palette."""
        if not self.active_palette:
            return frame
        
        # Simple color shift toward palette
        from PIL import ImageEnhance
        
        # Adjust saturation based on palette
        if 'noir' in self.active_palette.name.lower():
            enhancer = ImageEnhance.Color(frame)
            frame = enhancer.enhance(0.4)  # Desaturate for noir
        elif 'vibrant' in self.active_palette.name.lower():
            enhancer = ImageEnhance.Color(frame)
            frame = enhancer.enhance(1.3)  # Increase saturation
        
        return frame
    
    def register_style_embedding(self, style_name: str, embedding: np.ndarray):
        """Register style embedding from reference image or LoRA."""
        self.style_embeddings[style_name] = embedding
        logger.info(f"Registered style embedding: {style_name}")
    
    def compute_style_similarity(self, frame: Image.Image, style_name: str) -> float:
        """Compute style similarity between frame and target style."""
        if style_name not in self.style_embeddings:
            return 1.0
        
        # Extract style features from frame (placeholder)
        frame_embedding = np.random.randn(512)
        target_embedding = self.style_embeddings[style_name]
        
        # Cosine similarity
        similarity = np.dot(frame_embedding, target_embedding) / (
            np.linalg.norm(frame_embedding) * np.linalg.norm(target_embedding)
        )
        
        return float(similarity)
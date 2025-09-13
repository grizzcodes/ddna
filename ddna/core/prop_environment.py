"""Prop and environment persistence system."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from PIL import Image
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class PropObject:
    """Individual prop representation."""
    id: str
    name: str
    position: Tuple[float, float]  # Normalized coordinates (0-1)
    size: Tuple[float, float]  # Relative size
    depth: float  # Z-depth (0=foreground, 1=background)
    persistent: bool = True
    appearance: Optional[Dict] = None

@dataclass
class EnvironmentLayout:
    """Spatial layout of environment."""
    scene_id: str
    props: List[PropObject] = field(default_factory=list)
    regions: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    depth_layers: Dict[str, float] = field(default_factory=dict)
    
class PropEnvironmentPersistence:
    """Ensures props and environments maintain consistent placement."""
    
    def __init__(self):
        self.scene_graph: Dict[str, EnvironmentLayout] = {}
        self.prop_registry: Dict[str, PropObject] = {}
        self.spatial_index: Dict[str, Set[str]] = {}  # Region -> prop IDs
        self.object_detector = None  # Would be YOLO/Detectron2 in production
    
    def register_layout(self, scene_id: str, layout: EnvironmentLayout):
        """Register environment layout for scene."""
        self.scene_graph[scene_id] = layout
        
        # Update prop registry
        for prop in layout.props:
            self.prop_registry[prop.id] = prop
            
            # Update spatial index
            region = self._get_region(prop.position)
            if region not in self.spatial_index:
                self.spatial_index[region] = set()
            self.spatial_index[region].add(prop.id)
        
        logger.info(f"Registered layout for {scene_id} with {len(layout.props)} props")
    
    def validate_prop_placement(self, frame: Image.Image, scene_id: str) -> Tuple[bool, Dict]:
        """Validate prop placement in frame matches expected layout."""
        if scene_id not in self.scene_graph:
            logger.warning(f"No layout registered for {scene_id}")
            return True, {}
        
        layout = self.scene_graph[scene_id]
        detected_props = self._detect_props(frame)
        
        validation_results = {}
        all_valid = True
        
        for expected_prop in layout.props:
            if not expected_prop.persistent:
                continue
                
            # Check if prop is detected
            detected = self._find_matching_prop(expected_prop, detected_props)
            
            if detected:
                # Check position consistency
                position_error = self._calculate_position_error(
                    expected_prop.position,
                    detected['position']
                )
                
                validation_results[expected_prop.id] = {
                    'found': True,
                    'position_error': position_error,
                    'valid': position_error < 0.1  # 10% tolerance
                }
                
                if position_error >= 0.1:
                    all_valid = False
                    logger.warning(f"Prop {expected_prop.name} position error: {position_error:.3f}")
            else:
                validation_results[expected_prop.id] = {
                    'found': False,
                    'valid': False
                }
                all_valid = False
                logger.warning(f"Prop {expected_prop.name} not found in frame")
        
        return all_valid, validation_results
    
    def _detect_props(self, frame: Image.Image) -> List[Dict]:
        """Detect props in frame using object detection."""
        # Placeholder - would use YOLO/Detectron2 in production
        detected = []
        
        # Simulate detection
        for i in range(3):
            detected.append({
                'class': f'prop_{i}',
                'position': (np.random.random(), np.random.random()),
                'confidence': 0.8 + np.random.random() * 0.2
            })
        
        return detected
    
    def _find_matching_prop(self, expected: PropObject, detected: List[Dict]) -> Optional[Dict]:
        """Find matching prop in detected objects."""
        for det in detected:
            # Simple name matching - would use visual similarity in production
            if expected.name.lower() in det.get('class', '').lower():
                return det
        return None
    
    def _calculate_position_error(self, expected: Tuple[float, float], 
                                 actual: Tuple[float, float]) -> float:
        """Calculate normalized position error."""
        return np.sqrt((expected[0] - actual[0])**2 + (expected[1] - actual[1])**2)
    
    def _get_region(self, position: Tuple[float, float]) -> str:
        """Get region name for position."""
        x, y = position
        
        if x < 0.33:
            h_region = 'left'
        elif x < 0.67:
            h_region = 'center'
        else:
            h_region = 'right'
            
        if y < 0.33:
            v_region = 'top'
        elif y < 0.67:
            v_region = 'middle'
        else:
            v_region = 'bottom'
            
        return f"{v_region}_{h_region}"
    
    def generate_prop_mask(self, scene_id: str, frame_size: Tuple[int, int]) -> Image.Image:
        """Generate mask showing expected prop positions."""
        if scene_id not in self.scene_graph:
            return Image.new('L', frame_size, 0)
        
        layout = self.scene_graph[scene_id]
        mask = Image.new('L', frame_size, 0)
        
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        
        for prop in layout.props:
            if not prop.persistent:
                continue
                
            # Convert normalized coordinates to pixel coordinates
            x = int(prop.position[0] * frame_size[0])
            y = int(prop.position[1] * frame_size[1])
            w = int(prop.size[0] * frame_size[0])
            h = int(prop.size[1] * frame_size[1])
            
            # Draw prop region
            draw.ellipse([x-w//2, y-h//2, x+w//2, y+h//2], fill=255)
        
        return mask
    
    def export_scene_graph(self, scene_id: str) -> str:
        """Export scene graph to JSON."""
        if scene_id not in self.scene_graph:
            return "{}"
        
        layout = self.scene_graph[scene_id]
        
        export_data = {
            'scene_id': layout.scene_id,
            'props': [
                {
                    'id': prop.id,
                    'name': prop.name,
                    'position': prop.position,
                    'size': prop.size,
                    'depth': prop.depth,
                    'persistent': prop.persistent
                }
                for prop in layout.props
            ],
            'regions': layout.regions,
            'depth_layers': layout.depth_layers
        }
        
        return json.dumps(export_data, indent=2)
    
    def import_scene_graph(self, json_str: str):
        """Import scene graph from JSON."""
        data = json.loads(json_str)
        
        props = [
            PropObject(
                id=p['id'],
                name=p['name'],
                position=tuple(p['position']),
                size=tuple(p['size']),
                depth=p['depth'],
                persistent=p.get('persistent', True)
            )
            for p in data['props']
        ]
        
        layout = EnvironmentLayout(
            scene_id=data['scene_id'],
            props=props,
            regions=data.get('regions', {}),
            depth_layers=data.get('depth_layers', {})
        )
        
        self.register_layout(data['scene_id'], layout)
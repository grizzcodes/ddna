"""Scene DNA Parser - Extracts visual DNA from screenplays."""

import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
import fountain


class Character(BaseModel):
    """Character representation in DNA."""
    id: str
    name: str
    appearance: Optional[str] = None
    position: Optional[str] = None
    action: Optional[str] = None
    dialogue: Optional[str] = None


class Environment(BaseModel):
    """Environment representation in DNA."""
    location: str
    time_of_day: str
    lighting: Optional[str] = None
    weather: Optional[str] = None
    atmosphere: Optional[str] = None


class Style(BaseModel):
    """Visual style representation in DNA."""
    genre: str
    color_palette: List[str] = Field(default_factory=list)
    mood: Optional[str] = None
    camera_angle: Optional[str] = None
    shot_type: Optional[str] = None


class SceneDNA(BaseModel):
    """Complete DNA structure for a scene."""
    scene_id: str
    scene_number: Optional[int] = None
    environment: Environment
    characters: List[Character] = Field(default_factory=list)
    props: List[str] = Field(default_factory=list)
    style: Style
    action_lines: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_screenplay(cls, screenplay_path: str) -> "SceneDNA":
        """Create SceneDNA from a screenplay file."""
        parser = DNAParser()
        return parser.parse_screenplay(screenplay_path)

    def to_yaml(self) -> str:
        """Export DNA to YAML format."""
        return yaml.dump(self.model_dump(), default_flow_style=False)

    def to_prompt(self) -> str:
        """Convert DNA to generation prompt."""
        prompt_parts = []
        
        # Environment
        env = self.environment
        prompt_parts.append(
            f"{env.location} scene, {env.time_of_day}, {env.lighting or 'natural'} lighting"
        )
        
        # Characters
        for char in self.characters:
            char_desc = f"{char.name}"
            if char.appearance:
                char_desc += f", {char.appearance}"
            if char.position:
                char_desc += f", {char.position}"
            if char.action:
                char_desc += f", {char.action}"
            prompt_parts.append(char_desc)
        
        # Props
        if self.props:
            prompt_parts.append(f"Props: {', '.join(self.props)}")
        
        # Style
        style = self.style
        prompt_parts.append(
            f"{style.genre} style, {style.mood or 'dramatic'} mood"
        )
        
        if style.camera_angle:
            prompt_parts.append(f"Camera: {style.camera_angle}")
        
        return ", ".join(prompt_parts)


class DNAParser:
    """Parser for extracting DNA from screenplays."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize parser with optional config."""
        self.config = self._load_config(config_path)
        self.scene_counter = 0
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load parser configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default parser configuration."""
        return {
            "extract_props": True,
            "extract_lighting": True,
            "infer_mood": True,
            "default_genre": "drama",
            "default_palette": ["#2c3e50", "#34495e", "#7f8c8d"],
        }
    
    def parse_screenplay(self, screenplay_path: str) -> SceneDNA:
        """Parse screenplay and extract DNA."""
        screenplay = self._load_screenplay(screenplay_path)
        
        # Extract scene heading
        scene_heading = self._extract_scene_heading(screenplay)
        environment = self._parse_environment(scene_heading)
        
        # Extract characters and dialogue
        characters = self._extract_characters(screenplay)
        
        # Extract props from action lines
        action_lines = self._extract_action_lines(screenplay)
        props = self._extract_props(action_lines)
        
        # Determine style
        style = self._determine_style(screenplay)
        
        # Generate scene ID
        self.scene_counter += 1
        scene_id = self._generate_scene_id(environment, self.scene_counter)
        
        return SceneDNA(
            scene_id=scene_id,
            scene_number=self.scene_counter,
            environment=environment,
            characters=characters,
            props=props,
            style=style,
            action_lines=action_lines,
        )
    
    def _load_screenplay(self, screenplay_path: str) -> Dict:
        """Load and parse screenplay file."""
        path = Path(screenplay_path)
        
        if path.suffix == '.fountain':
            with open(path, 'r') as f:
                content = f.read()
            return fountain.Fountain(content)
        else:
            # Simple text parsing fallback
            with open(path, 'r') as f:
                lines = f.readlines()
            return {"lines": lines}
    
    def _extract_scene_heading(self, screenplay: Dict) -> str:
        """Extract scene heading from screenplay."""
        if isinstance(screenplay, fountain.Fountain):
            # Extract from Fountain format
            for element in screenplay.elements:
                if element.element_type == 'Scene Heading':
                    return element.element_text
        else:
            # Simple extraction from text
            for line in screenplay.get("lines", []):
                if line.strip().startswith(('INT.', 'EXT.', 'INT ', 'EXT ')):
                    return line.strip()
        
        return "INT. LOCATION - DAY"
    
    def _parse_environment(self, scene_heading: str) -> Environment:
        """Parse environment from scene heading."""
        # Parse INT/EXT
        is_interior = 'INT' in scene_heading.upper()
        
        # Parse location
        location_match = re.search(r'(?:INT|EXT)\.?\s+([^-]+)', scene_heading, re.I)
        location = location_match.group(1).strip() if location_match else "UNKNOWN"
        
        # Parse time of day
        time_patterns = {
            'DAY': ['DAY', 'MORNING', 'NOON', 'AFTERNOON'],
            'NIGHT': ['NIGHT', 'EVENING', 'DUSK'],
            'DAWN': ['DAWN', 'SUNRISE'],
            'SUNSET': ['SUNSET', 'GOLDEN HOUR'],
        }
        
        time_of_day = 'DAY'
        heading_upper = scene_heading.upper()
        for time, patterns in time_patterns.items():
            if any(p in heading_upper for p in patterns):
                time_of_day = time
                break
        
        # Infer lighting
        lighting = self._infer_lighting(is_interior, time_of_day)
        
        return Environment(
            location=location.lower().replace('_', ' '),
            time_of_day=time_of_day.lower(),
            lighting=lighting,
        )
    
    def _infer_lighting(self, is_interior: bool, time_of_day: str) -> str:
        """Infer lighting from environment."""
        lighting_map = {
            ('INT', 'DAY'): 'soft natural light from windows',
            ('INT', 'NIGHT'): 'warm interior lighting',
            ('EXT', 'DAY'): 'bright natural sunlight',
            ('EXT', 'NIGHT'): 'moonlight and street lamps',
            ('INT', 'DAWN'): 'dim morning light',
            ('EXT', 'DAWN'): 'soft golden hour lighting',
            ('INT', 'SUNSET'): 'warm golden light through windows',
            ('EXT', 'SUNSET'): 'dramatic golden hour lighting',
        }
        
        location_type = 'INT' if is_interior else 'EXT'
        key = (location_type, time_of_day.upper())
        
        return lighting_map.get(key, 'natural lighting')
    
    def _extract_characters(self, screenplay: Dict) -> List[Character]:
        """Extract characters from screenplay."""
        characters = []
        
        if isinstance(screenplay, fountain.Fountain):
            for element in screenplay.elements:
                if element.element_type == 'Character':
                    char_name = element.element_text.strip()
                    # Remove any parentheticals
                    char_name = re.sub(r'\([^)]+\)', '', char_name).strip()
                    
                    characters.append(Character(
                        id=char_name.upper().replace(' ', '_'),
                        name=char_name,
                    ))
        
        return characters
    
    def _extract_action_lines(self, screenplay: Dict) -> List[str]:
        """Extract action lines from screenplay."""
        action_lines = []
        
        if isinstance(screenplay, fountain.Fountain):
            for element in screenplay.elements:
                if element.element_type == 'Action':
                    action_lines.append(element.element_text.strip())
        else:
            # Simple extraction from text
            for line in screenplay.get("lines", []):
                line = line.strip()
                if line and not line.startswith(('INT.', 'EXT.', '(', '[')):
                    if not line.isupper():  # Not a character name
                        action_lines.append(line)
        
        return action_lines
    
    def _extract_props(self, action_lines: List[str]) -> List[str]:
        """Extract props from action lines."""
        props = set()
        
        # Common props to look for
        prop_patterns = [
            r'\b(table|chair|desk|door|window|phone|computer|laptop|gun|knife|' +
            r'car|book|glass|bottle|cigarette|newspaper|briefcase|suitcase|' +
            r'camera|television|tv|radio|lamp|light|candle|mirror|picture|' +
            r'painting|couch|sofa|bed|clock|watch)s?\b',
        ]
        
        for line in action_lines:
            line_lower = line.lower()
            for pattern in prop_patterns:
                matches = re.findall(pattern, line_lower)
                props.update(matches)
        
        return list(props)
    
    def _determine_style(self, screenplay: Dict) -> Style:
        """Determine visual style from screenplay content."""
        # This would be more sophisticated in production
        # Could analyze genre from content, dialogue tone, etc.
        
        return Style(
            genre=self.config["default_genre"],
            color_palette=self.config["default_palette"],
            mood="dramatic",
        )
    
    def _generate_scene_id(self, environment: Environment, scene_num: int) -> str:
        """Generate unique scene ID."""
        location = environment.location.upper().replace(' ', '_')
        time = environment.time_of_day.upper()
        return f"{location}_{time}_{scene_num:03d}"
"""Hierarchical constraint management system."""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from .dna_parser import SceneDNA

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints."""
    COLOR = "color"
    LIGHTING = "lighting"
    COMPOSITION = "composition"
    CHARACTER_POSITION = "character_position"
    PROP_PLACEMENT = "prop_placement"
    CAMERA = "camera"
    STYLE = "style"
    SEED = "seed"


class ConstraintPriority(Enum):
    """Priority levels for constraints."""
    CRITICAL = 1  # Must not change
    HIGH = 2      # Should remain very consistent
    MEDIUM = 3    # Some variation allowed
    LOW = 4       # Flexible


@dataclass
class Constraint:
    """Individual constraint definition."""
    id: str
    type: ConstraintType
    priority: ConstraintPriority
    target: Optional[str]  # What this constraint applies to
    value: Any
    locked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstraintManager:
    """Manages hierarchical constraints for scene consistency."""
    
    def __init__(self, level: str = "strict"):
        """Initialize constraint manager.
        
        Args:
            level: Consistency level (strict, moderate, loose)
        """
        self.level = level
        self.constraints: Dict[str, Constraint] = {}
        self.constraint_hierarchy = self._define_hierarchy()
        self.locked_elements: Set[str] = set()
    
    def _define_hierarchy(self) -> Dict[str, ConstraintPriority]:
        """Define constraint hierarchy based on consistency level."""
        if self.level == "strict":
            return {
                "seed": ConstraintPriority.CRITICAL,
                "color": ConstraintPriority.CRITICAL,
                "lighting": ConstraintPriority.HIGH,
                "character_position": ConstraintPriority.HIGH,
                "prop_placement": ConstraintPriority.HIGH,
                "composition": ConstraintPriority.MEDIUM,
                "camera": ConstraintPriority.MEDIUM,
                "style": ConstraintPriority.CRITICAL,
            }
        elif self.level == "moderate":
            return {
                "seed": ConstraintPriority.HIGH,
                "color": ConstraintPriority.HIGH,
                "lighting": ConstraintPriority.MEDIUM,
                "character_position": ConstraintPriority.MEDIUM,
                "prop_placement": ConstraintPriority.LOW,
                "composition": ConstraintPriority.LOW,
                "camera": ConstraintPriority.MEDIUM,
                "style": ConstraintPriority.HIGH,
            }
        else:  # loose
            return {
                "seed": ConstraintPriority.MEDIUM,
                "color": ConstraintPriority.MEDIUM,
                "lighting": ConstraintPriority.LOW,
                "character_position": ConstraintPriority.LOW,
                "prop_placement": ConstraintPriority.LOW,
                "composition": ConstraintPriority.LOW,
                "camera": ConstraintPriority.LOW,
                "style": ConstraintPriority.MEDIUM,
            }
    
    def create_constraints(self, dna: SceneDNA) -> List[Constraint]:
        """Create constraints from Scene DNA.
        
        Args:
            dna: Scene DNA specification
        
        Returns:
            List of constraints
        """
        constraints = []
        
        # Seed constraint
        if dna.seed:
            constraints.append(self._create_constraint(
                "seed",
                ConstraintType.SEED,
                dna.seed,
                locked=True
            ))
        
        # Color palette constraint
        if dna.style.color_palette:
            constraints.append(self._create_constraint(
                "color_palette",
                ConstraintType.COLOR,
                dna.style.color_palette
            ))
        
        # Lighting constraint
        if dna.environment.lighting:
            constraints.append(self._create_constraint(
                "lighting",
                ConstraintType.LIGHTING,
                dna.environment.lighting
            ))
        
        # Character position constraints
        for char in dna.characters:
            if hasattr(char, 'position') and char.position:
                constraints.append(self._create_constraint(
                    f"char_pos_{char.id}",
                    ConstraintType.CHARACTER_POSITION,
                    char.position,
                    target=char.id
                ))
        
        # Camera constraints
        if dna.style.camera_angle:
            constraints.append(self._create_constraint(
                "camera_angle",
                ConstraintType.CAMERA,
                dna.style.camera_angle
            ))
        
        # Style constraint
        constraints.append(self._create_constraint(
            "style",
            ConstraintType.STYLE,
            dna.style.genre
        ))
        
        # Store constraints
        for constraint in constraints:
            self.constraints[constraint.id] = constraint
        
        return constraints
    
    def _create_constraint(self,
                          constraint_id: str,
                          constraint_type: ConstraintType,
                          value: Any,
                          target: Optional[str] = None,
                          locked: bool = False) -> Constraint:
        """Create a single constraint."""
        priority = self.constraint_hierarchy.get(
            constraint_type.value,
            ConstraintPriority.MEDIUM
        )
        
        return Constraint(
            id=constraint_id,
            type=constraint_type,
            priority=priority,
            target=target,
            value=value,
            locked=locked
        )
    
    def update_constraints(self,
                          current_constraints: List[Constraint],
                          generated_frame: Any) -> List[Constraint]:
        """Update constraints based on generated frame.
        
        Args:
            current_constraints: Current active constraints
            generated_frame: The generated frame
        
        Returns:
            Updated constraints
        """
        updated = current_constraints.copy()
        
        # Analyze frame and lock successful elements
        # This would involve image analysis in production
        # For now, we'll just lock high-priority constraints
        
        for constraint in updated:
            if constraint.priority == ConstraintPriority.CRITICAL:
                constraint.locked = True
                self.locked_elements.add(constraint.id)
        
        return updated
    
    def validate_constraints(self, constraints: List[Constraint]) -> bool:
        """Validate that constraints are compatible.
        
        Args:
            constraints: List of constraints to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Check for conflicts
        constraint_values = {}
        
        for constraint in constraints:
            key = (constraint.type, constraint.target)
            if key in constraint_values:
                # Check if values conflict
                if constraint_values[key] != constraint.value:
                    logger.warning(
                        f"Constraint conflict: {constraint.type} for {constraint.target}"
                    )
                    return False
            constraint_values[key] = constraint.value
        
        return True
    
    def merge_constraints(self,
                         primary: List[Constraint],
                         secondary: List[Constraint]) -> List[Constraint]:
        """Merge two sets of constraints.
        
        Args:
            primary: Primary constraints (higher priority)
            secondary: Secondary constraints
        
        Returns:
            Merged constraint list
        """
        merged = {}
        
        # Add primary constraints
        for constraint in primary:
            key = (constraint.type, constraint.target)
            merged[key] = constraint
        
        # Add secondary constraints if not conflicting
        for constraint in secondary:
            key = (constraint.type, constraint.target)
            if key not in merged:
                merged[key] = constraint
            elif merged[key].priority.value > constraint.priority.value:
                # Replace with higher priority constraint
                merged[key] = constraint
        
        return list(merged.values())
    
    def lock_element(self, element_id: str):
        """Lock an element to prevent changes.
        
        Args:
            element_id: ID of element to lock
        """
        self.locked_elements.add(element_id)
        if element_id in self.constraints:
            self.constraints[element_id].locked = True
    
    def unlock_element(self, element_id: str):
        """Unlock an element to allow changes.
        
        Args:
            element_id: ID of element to unlock
        """
        self.locked_elements.discard(element_id)
        if element_id in self.constraints:
            self.constraints[element_id].locked = False
    
    def get_locked_elements(self) -> Set[str]:
        """Get all locked elements.
        
        Returns:
            Set of locked element IDs
        """
        return self.locked_elements.copy()
    
    def export_constraints(self) -> str:
        """Export constraints to JSON.
        
        Returns:
            JSON string of constraints
        """
        export_data = {
            "level": self.level,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "priority": c.priority.value,
                    "target": c.target,
                    "value": c.value,
                    "locked": c.locked,
                    "metadata": c.metadata
                }
                for c in self.constraints.values()
            ],
            "locked_elements": list(self.locked_elements)
        }
        return json.dumps(export_data, indent=2)
    
    def import_constraints(self, json_str: str):
        """Import constraints from JSON.
        
        Args:
            json_str: JSON string of constraints
        """
        data = json.loads(json_str)
        
        self.level = data.get("level", "strict")
        self.constraints = {}
        self.locked_elements = set(data.get("locked_elements", []))
        
        for c_data in data.get("constraints", []):
            constraint = Constraint(
                id=c_data["id"],
                type=ConstraintType(c_data["type"]),
                priority=ConstraintPriority(c_data["priority"]),
                target=c_data.get("target"),
                value=c_data["value"],
                locked=c_data.get("locked", False),
                metadata=c_data.get("metadata", {})
            )
            self.constraints[constraint.id] = constraint
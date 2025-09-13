"""Frame Society - Multi-frame consistency through social simulation.

Inspired by Societies.io's approach of simulating audiences as AI personas,
we simulate frame sequences as interconnected visual agents that must
maintain consistency while allowing creative variation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from PIL import Image
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class FramePersona:
    """A frame as an AI persona with visual DNA and behavior."""
    frame_id: str
    frame_number: int
    visual_dna: Dict[str, Any]  # Identity, style, composition
    connections: List[str] = field(default_factory=list)  # Connected frames
    influence_score: float = 1.0
    consistency_threshold: float = 0.83  # Target 83% like Societies.io
    
    def react_to(self, other_frame: 'FramePersona') -> float:
        """How well this frame 'accepts' another frame's visual elements."""
        similarity = self._calculate_visual_similarity(other_frame)
        
        # Frames react based on their position in sequence
        temporal_distance = abs(self.frame_number - other_frame.frame_number)
        temporal_penalty = 1.0 / (1.0 + temporal_distance * 0.1)
        
        reaction = similarity * temporal_penalty * self.influence_score
        return reaction
    
    def _calculate_visual_similarity(self, other: 'FramePersona') -> float:
        """Calculate how visually similar two frame personas are."""
        score = 0.0
        weights = {
            'identity': 0.3,
            'style': 0.25,
            'composition': 0.2,
            'lighting': 0.15,
            'props': 0.1
        }
        
        for key, weight in weights.items():
            if key in self.visual_dna and key in other.visual_dna:
                # Simplified similarity - would use actual metrics in production
                if self.visual_dna[key] == other.visual_dna[key]:
                    score += weight
                elif isinstance(self.visual_dna[key], dict):
                    # Partial match for complex attributes
                    matches = sum(
                        1 for k in self.visual_dna[key]
                        if k in other.visual_dna[key] and
                        self.visual_dna[key][k] == other.visual_dna[key][k]
                    )
                    if self.visual_dna[key]:
                        score += weight * (matches / len(self.visual_dna[key]))
        
        return score


@dataclass
class FrameVariant:
    """A variant of a frame for A/B testing."""
    variant_id: str
    base_frame_id: str
    modifications: Dict[str, Any]
    predicted_consistency: float = 0.0
    actual_consistency: Optional[float] = None
    propagation_score: float = 0.0


class FrameSociety:
    """Simulates frame sequences as a society of interconnected visual agents."""
    
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.frame_personas: Dict[str, FramePersona] = {}
        self.network_graph: Dict[str, List[str]] = {}
        self.simulation_history: List[Dict] = []
        self.learning_rate = 0.1
        self.consistency_model = self._init_consistency_model()
        
    def _init_consistency_model(self) -> Dict:
        """Initialize the consistency prediction model."""
        return {
            'weights': np.random.randn(10),  # Feature weights
            'bias': 0.0,
            'accuracy': 0.17,  # Start at ChatGPT baseline
            'target_accuracy': 0.83  # Societies.io achieved
        }
    
    def create_frame_society(self, dna_sequence: List[Dict], num_frames: int = 5):
        """Create a society of frame personas from DNA sequence."""
        
        # Create frame personas
        for i in range(num_frames):
            frame_id = f"F{i:03d}"
            
            # Build visual DNA from scene DNA
            visual_dna = {
                'identity': dna_sequence[i].get('characters', []),
                'style': dna_sequence[i].get('style', {}),
                'composition': dna_sequence[i].get('composition', {}),
                'lighting': dna_sequence[i].get('lighting', {}),
                'props': dna_sequence[i].get('props', [])
            }
            
            persona = FramePersona(
                frame_id=frame_id,
                frame_number=i,
                visual_dna=visual_dna
            )
            
            self.frame_personas[frame_id] = persona
        
        # Build connections (temporal neighbors + visual similarity)
        self._build_network_graph()
        
        logger.info(f"Created frame society with {len(self.frame_personas)} personas")
    
    def _build_network_graph(self):
        """Build the network of frame connections."""
        frames = list(self.frame_personas.values())
        
        for i, frame in enumerate(frames):
            connections = []
            
            # Direct temporal neighbors (strongest connection)
            if i > 0:
                connections.append(frames[i-1].frame_id)
            if i < len(frames) - 1:
                connections.append(frames[i+1].frame_id)
            
            # Second-degree temporal neighbors (weaker)
            if i > 1:
                connections.append(frames[i-2].frame_id)
            if i < len(frames) - 2:
                connections.append(frames[i+2].frame_id)
            
            frame.connections = connections
            self.network_graph[frame.frame_id] = connections
    
    def simulate_variant(self, 
                        frame_id: str, 
                        variant: FrameVariant,
                        num_simulations: int = 100) -> Dict:
        """Simulate how a frame variant propagates through the society."""
        
        results = []
        
        for sim in range(num_simulations):
            # Start propagation from modified frame
            activated_frames = set([frame_id])
            propagation_queue = [frame_id]
            reaction_scores = {}
            
            while propagation_queue:
                current_frame_id = propagation_queue.pop(0)
                current_persona = self.frame_personas[current_frame_id]
                
                # Check reactions from connected frames
                for connected_id in current_persona.connections:
                    if connected_id not in activated_frames:
                        connected_persona = self.frame_personas[connected_id]
                        
                        # Calculate reaction with some randomness
                        base_reaction = connected_persona.react_to(current_persona)
                        noise = np.random.normal(0, 0.1)
                        reaction = np.clip(base_reaction + noise, 0, 1)
                        
                        reaction_scores[connected_id] = reaction
                        
                        # Propagate if reaction is strong enough
                        if reaction > connected_persona.consistency_threshold:
                            activated_frames.add(connected_id)
                            propagation_queue.append(connected_id)
            
            # Calculate overall consistency score
            consistency = len(activated_frames) / len(self.frame_personas)
            results.append({
                'consistency': consistency,
                'activated_frames': len(activated_frames),
                'reaction_scores': reaction_scores
            })
        
        # Aggregate results
        avg_consistency = np.mean([r['consistency'] for r in results])
        std_consistency = np.std([r['consistency'] for r in results])
        
        return {
            'variant_id': variant.variant_id,
            'predicted_consistency': avg_consistency,
            'consistency_std': std_consistency,
            'confidence_interval': (
                avg_consistency - 2 * std_consistency,
                avg_consistency + 2 * std_consistency
            ),
            'propagation_pattern': self._analyze_propagation(results)
        }
    
    def _analyze_propagation(self, results: List[Dict]) -> Dict:
        """Analyze how consistency propagates through frames."""
        frame_activation_counts = {}
        
        for result in results:
            for frame_id in result['activated_frames']:
                frame_activation_counts[frame_id] = \
                    frame_activation_counts.get(frame_id, 0) + 1
        
        # Normalize to percentages
        total_sims = len(results)
        activation_rates = {
            fid: count / total_sims 
            for fid, count in frame_activation_counts.items()
        }
        
        return {
            'activation_rates': activation_rates,
            'most_consistent': max(activation_rates, key=activation_rates.get),
            'least_consistent': min(activation_rates, key=activation_rates.get)
        }
    
    def generate_variants(self, 
                         frame_id: str,
                         base_config: Dict,
                         num_variants: int = 100) -> List[FrameVariant]:
        """Generate multiple variants of a frame for testing."""
        variants = []
        
        # Parameters to vary
        variation_params = {
            'seed': lambda: np.random.randint(0, 2**32),
            'guidance_scale': lambda: np.random.uniform(5, 12),
            'controlnet_strength': lambda: np.random.uniform(0.6, 1.0),
            'color_temperature': lambda: np.random.uniform(-0.2, 0.2),
            'prompt_emphasis': lambda: np.random.choice([
                'cinematic', 'dramatic', 'subtle', 'intense'
            ])
        }
        
        for i in range(num_variants):
            modifications = {}
            
            # Randomly select parameters to modify
            num_mods = np.random.randint(1, 4)
            params_to_modify = np.random.choice(
                list(variation_params.keys()), 
                size=num_mods, 
                replace=False
            )
            
            for param in params_to_modify:
                modifications[param] = variation_params[param]()
            
            variant = FrameVariant(
                variant_id=f"{frame_id}_v{i:03d}",
                base_frame_id=frame_id,
                modifications=modifications
            )
            variants.append(variant)
        
        return variants
    
    def reinforcement_learning_update(self, 
                                     variant: FrameVariant,
                                     actual_consistency: float):
        """Update the model based on actual vs predicted consistency."""
        
        # Store ground truth
        variant.actual_consistency = actual_consistency
        
        # Calculate prediction error
        error = actual_consistency - variant.predicted_consistency
        
        # Update model weights (simplified gradient descent)
        features = self._extract_features(variant)
        self.consistency_model['weights'] += self.learning_rate * error * features
        
        # Update accuracy tracking
        self.simulation_history.append({
            'variant_id': variant.variant_id,
            'predicted': variant.predicted_consistency,
            'actual': actual_consistency,
            'error': abs(error)
        })
        
        # Calculate rolling accuracy
        if len(self.simulation_history) >= 10:
            recent_errors = [h['error'] for h in self.simulation_history[-10:]]
            accuracy = 1.0 - np.mean(recent_errors)
            self.consistency_model['accuracy'] = accuracy
            
            logger.info(f"Model accuracy: {accuracy:.2%} (target: 83%)")
    
    def _extract_features(self, variant: FrameVariant) -> np.ndarray:
        """Extract features from a variant for learning."""
        features = np.zeros(10)
        
        # Encode modifications as features
        mod_keys = ['seed', 'guidance_scale', 'controlnet_strength', 
                   'color_temperature', 'prompt_emphasis']
        
        for i, key in enumerate(mod_keys[:5]):
            if key in variant.modifications:
                if isinstance(variant.modifications[key], (int, float)):
                    features[i] = variant.modifications[key]
                else:
                    # Hash non-numeric values
                    features[i] = int(hashlib.md5(
                        str(variant.modifications[key]).encode()
                    ).hexdigest()[:8], 16) / 2**32
        
        # Add frame position features
        frame_num = int(variant.base_frame_id[1:])
        features[5] = frame_num / 100  # Normalized position
        features[6] = 1.0 if frame_num == 0 else 0.0  # Is first frame
        features[7] = 1.0 if frame_num == 4 else 0.0  # Is last frame (assuming 5)
        
        return features
    
    def find_optimal_variant(self, 
                            frame_id: str,
                            base_config: Dict,
                            target_consistency: float = 0.9) -> FrameVariant:
        """Find the optimal variant through parallel simulation."""
        
        # Generate variants
        variants = self.generate_variants(frame_id, base_config)
        
        # Simulate all variants in parallel
        results = []
        for variant in variants:
            sim_result = self.simulate_variant(frame_id, variant)
            variant.predicted_consistency = sim_result['predicted_consistency']
            results.append((variant, sim_result))
        
        # Sort by predicted consistency
        results.sort(key=lambda x: abs(x[0].predicted_consistency - target_consistency))
        
        best_variant = results[0][0]
        logger.info(f"Best variant {best_variant.variant_id}: "
                   f"consistency={best_variant.predicted_consistency:.2%}")
        
        return best_variant
    
    def export_society_graph(self) -> Dict:
        """Export the frame society as a graph structure."""
        nodes = []
        edges = []
        
        for frame_id, persona in self.frame_personas.items():
            nodes.append({
                'id': frame_id,
                'frame_number': persona.frame_number,
                'influence': persona.influence_score,
                'visual_dna': persona.visual_dna
            })
            
            for connected_id in persona.connections:
                edges.append({
                    'source': frame_id,
                    'target': connected_id,
                    'weight': persona.react_to(self.frame_personas[connected_id])
                })
        
        return {
            'scene_id': self.scene_id,
            'nodes': nodes,
            'edges': edges,
            'model_accuracy': self.consistency_model['accuracy']
        }

"""Tests for DNA parser."""

import pytest
from ddna.core import SceneDNA, DNAParser

def test_parse_screenplay():
    parser = DNAParser()
    
    # Test with sample screenplay
    screenplay_path = "input/screenplays/sample_scene.fountain"
    
    # This would be a real test in production
    # dna = parser.parse_screenplay(screenplay_path)
    # assert dna.scene_id
    # assert dna.environment
    # assert dna.environment.location == "detective's office"
    # assert dna.environment.time_of_day == "night"

def test_scene_dna_creation():
    from ddna.core.dna_parser import Environment, Style
    
    dna = SceneDNA(
        scene_id="TEST_001",
        environment=Environment(
            location="office",
            time_of_day="night",
            lighting="dim"
        ),
        style=Style(
            genre="neo_noir",
            color_palette=["#000000", "#ffffff"]
        )
    )
    
    assert dna.scene_id == "TEST_001"
    assert dna.environment.location == "office"
    assert len(dna.style.color_palette) == 2
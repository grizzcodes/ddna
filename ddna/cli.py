"""Command-line interface for DDNA."""

import click
import logging
from pathlib import Path
from .core import SceneDNA
from .pipelines import ScenePipeline

logging.basicConfig(level=logging.INFO)

@click.group()
def cli():
    """DDNA - Screenplay DNA System"""
    pass

@cli.command()
@click.argument('screenplay', type=click.Path(exists=True))
@click.option('--style', default='cinematic', help='Visual style')
@click.option('--frames', default=10, help='Number of frames')
@click.option('--output', default='output/scenes', help='Output directory')
def generate(screenplay, style, frames, output):
    """Generate scene from screenplay."""
    click.echo(f"Processing {screenplay}...")
    
    # Parse DNA
    dna = SceneDNA.from_screenplay(screenplay)
    click.echo(f"Extracted DNA for scene: {dna.scene_id}")
    
    # Generate scene
    pipeline = ScenePipeline(style=style, output_dir=output)
    result = pipeline.generate(dna, frames=frames)
    
    if result.success:
        click.echo(f"✓ Generated {result.frames_generated} frames")
        click.echo(f"✓ Output: {result.output_path}")
        click.echo(f"✓ Consistency: {result.consistency_score:.2f}")
    else:
        click.echo(f"✗ Generation failed: {result.errors}")

@cli.command()
@click.argument('dna_file', type=click.Path(exists=True))
def validate(dna_file):
    """Validate DNA file."""
    import json
    
    with open(dna_file, 'r') as f:
        dna_data = json.load(f)
    
    dna = SceneDNA(**dna_data)
    from .validators import DNAValidator
    
    validator = DNAValidator()
    if validator.validate(dna):
        click.echo("✓ DNA is valid")
    else:
        click.echo("✗ DNA validation failed")

def main():
    cli()

if __name__ == '__main__':
    main()
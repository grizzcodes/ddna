#!/usr/bin/env python3
"""Demo script showing 5 Pillars of Consistency in action."""

import sys
sys.path.append('..')

from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ddna.core import SceneDNA
from ddna.pipelines.advanced_frame_pipeline import AdvancedFramePipeline
from ddna.core.style_palette_lock import ColorPalette
from ddna.core.dna_parser import Environment, Style, Character

def create_demo_frames():
    """Create demo with the 5 pillars."""
    
    print("\n🎬 DDNA 5 Pillars Demo\n" + "="*50)
    
    # 1. Create DNA directly (simpler for demo)
    print("\n📝 Step 1: Creating Scene DNA...")
    
    dna = SceneDNA(
        scene_id="INT_OFFICE_NIGHT_001",
        environment=Environment(
            location="detective office",
            time_of_day="night",
            lighting="dim fluorescent"
        ),
        characters=[
            Character(id="JONES", name="Detective Jones"),
            Character(id="SARAH", name="Sarah Chen")
        ],
        props=["desk", "lamp", "whiskey glass", "case files"],
        style=Style(
            genre="neo_noir",
            color_palette=["#1a1a2e", "#e94560", "#0f0f0f"]
        )
    )
    
    print(f"  ✓ Scene ID: {dna.scene_id}")
    print(f"  ✓ Characters: {len(dna.characters)}")
    print(f"  ✓ Props: {len(dna.props)}")
    
    # 2. Initialize advanced pipeline
    print("\n🔧 Step 2: Initializing 5 Pillars...")
    pipeline = AdvancedFramePipeline(dna)
    
    print("  ✓ Identity Lock initialized")
    print("  ✓ Style & Palette Lock initialized")
    print("  ✓ Pose & Composition Continuity initialized")
    print("  ✓ Prop & Environment Persistence initialized")
    print("  ✓ Seed Registry initialized")
    
    # 3. Generate reference frames (placeholders for demo)
    print("\n🎨 Step 3: Creating reference frames...")
    first_frame = create_reference_frame(dna, "FIRST")
    last_frame = create_reference_frame(dna, "LAST")
    print("  ✓ First frame created")
    print("  ✓ Last frame created")
    
    # 4. Generate 5 frames with consistency
    print("\n🚀 Step 4: Generating 5 consistent frames...")
    frames = pipeline.generate_5_frames(first_frame, last_frame)
    
    for frame_id in sorted(frames.keys()):
        print(f"  ✓ {frame_id} generated")
    
    # 5. Get validation report
    print("\n📊 Step 5: Validation Report")
    report = pipeline.get_validation_report()
    
    for frame_id, scores in report['frames'].items():
        print(f"\n  Frame {frame_id}:")
        for pillar, results in scores.items():
            status = "✅" if results['valid'] else "❌"
            print(f"    {status} {pillar.capitalize()}: Valid={results['valid']}")
    
    # 6. Save results
    print("\n💾 Step 6: Saving results...")
    output_dir = Path("output/demo_5_pillars")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_id, frame in frames.items():
        frame_path = output_dir / f"{frame_id}.png"
        frame.save(frame_path)
        print(f"  ✓ Saved {frame_path}")
    
    # Save report
    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report saved to {report_path}")
    
    # Create comparison grid
    print("\n🎯 Creating comparison grid...")
    grid = create_comparison_grid(frames)
    grid_path = output_dir / "5_frame_grid.png"
    grid.save(grid_path)
    print(f"  ✓ Grid saved to {grid_path}")
    
    print("\n" + "="*50)
    print("✨ Demo complete! Check output/demo_5_pillars/")
    
    return frames, report

def create_reference_frame(dna, position="FIRST"):
    """Create a reference frame for demo."""
    img = Image.new('RGB', (1920, 1080))
    
    # Create noir-style gradient
    pixels = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    if position == "FIRST":
        # Dark blue gradient
        for i in range(1080):
            pixels[i, :] = [
                int(10 + (i/1080) * 20),
                int(10 + (i/1080) * 20),
                int(20 + (i/1080) * 30)
            ]
    else:
        # Slightly different for last frame
        for i in range(1080):
            pixels[i, :] = [
                int(15 + (i/1080) * 25),
                int(10 + (i/1080) * 20),
                int(25 + (i/1080) * 35)
            ]
    
    img = Image.fromarray(pixels)
    
    # Add visual elements
    draw = ImageDraw.Draw(img)
    
    # Add "desk" rectangle
    draw.rectangle([500, 600, 1420, 900], fill=(40, 30, 30), outline=(60, 50, 50))
    
    # Add "lamp" circle
    draw.ellipse([300, 200, 400, 300], fill=(200, 180, 100))
    
    # Add character placeholders
    if position == "FIRST":
        draw.rectangle([800, 300, 900, 600], fill=(80, 70, 70))  # Jones
    else:
        draw.rectangle([850, 300, 950, 600], fill=(80, 70, 70))  # Jones moved
        draw.rectangle([1200, 350, 1280, 650], fill=(100, 60, 60))  # Sarah
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 50), f"{position} FRAME", fill='white', font=font)
    draw.text((50, 980), f"Scene: {dna.scene_id}", fill='gray')
    
    return img

def create_comparison_grid(frames):
    """Create a grid showing all 5 frames."""
    # Create grid image
    grid_width = 1920
    grid_height = 1080
    grid = Image.new('RGB', (grid_width, grid_height), 'black')
    
    # Calculate positions for 5 frames in a row
    frame_width = grid_width // 5
    
    # Paste frames
    for i, frame_id in enumerate(sorted(frames.keys())):
        if frame_id in frames:
            # Resize frame to fit
            small_frame = frames[frame_id].resize((frame_width, grid_height), Image.Resampling.LANCZOS)
            
            # Paste to grid
            grid.paste(small_frame, (i * frame_width, 0))
            
            # Add label
            draw = ImageDraw.Draw(grid)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Label with frame ID
            label = f"{frame_id}"
            if i in [0, 2, 4]:  # Anchors
                label += "\n(ANCHOR)"
            else:
                label += "\n(INTERP)"
            
            draw.text((i * frame_width + 10, 10), label, fill='white', font=font)
    
    return grid

if __name__ == "__main__":
    create_demo_frames()
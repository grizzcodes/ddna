# 5 Pillars of Consistency - DDNA Implementation

## Overview

The DDNA system implements 5 core pillars to ensure frame-to-frame consistency in generated scenes:

## 1. Identity Lock 🔐

**File:** `ddna/core/identity_lock.py`

- **Face/Body Embeddings:** Uses ArcFace/CLIP encoders to create identity vectors
- **Validation:** Cosine similarity threshold of 0.92
- **Re-roll on Failure:** Automatically regenerates frames that fail identity checks
- **Character Registry:** Maintains persistent character representations

```python
identity_lock = IdentityLock(similarity_threshold=0.92)
identity_lock.register_identity("DETECTIVE_JONES", reference_image)
valid, scores = identity_lock.validate_frame(generated_frame, ["DETECTIVE_JONES"])
```

## 2. Style & Palette Lock 🎨

**File:** `ddna/core/style_palette_lock.py`

- **Color Distance (ΔE):** CIEDE2000 color difference validation
- **LUT Support:** 3D LUT application for consistent grading
- **Palette Enforcement:** Automatic color correction to match target palette
- **Style Embeddings:** Consistent style application via LoRA/IP-Adapter

```python
palette = ColorPalette(
    name="neo_noir",
    primary_colors=["#1a1a2e", "#e94560"],
    max_delta_e=10.0
)
style_lock.lock_palette(palette)
valid, delta_e = style_lock.validate_frame_colors(frame)
```

## 3. Pose & Composition Continuity 🎬

**File:** `ddna/core/pose_composition.py`

- **ControlNet Integration:** Depth, pose, lineart, normal maps
- **Camera Splines:** Smooth camera movement interpolation
- **Pose Interpolation:** Frame-to-frame pose consistency
- **Composition Validation:** Ensures perspective continuity

```python
pose_continuity = PoseCompositionContinuity()
camera_spline = pose_continuity.generate_camera_spline(start, end, num_frames=5)
conditions = pose_continuity.generate_controlnet_conditions(frame_number)
```

## 4. Prop & Environment Persistence 📦

**File:** `ddna/core/prop_environment.py`

- **Scene Graph:** Spatial relationship preservation
- **Object Detection:** Validates prop placement in generated frames
- **Spatial Indexing:** Ensures consistent positioning (stage left stays left)
- **Depth Layers:** Maintains foreground/background relationships

```python
layout = EnvironmentLayout(
    scene_id="INT_OFFICE_NIGHT",
    props=[PropObject(id="desk", position=(0.3, 0.5), persistent=True)]
)
prop_persistence.register_layout(scene_id, layout)
valid, results = prop_persistence.validate_prop_placement(frame, scene_id)
```

## 5. Seed & Constraint Registry 🔧

**File:** `ddna/core/seed_registry.py`

- **Deterministic Seeds:** Config hash-based seed generation
- **Lock File (`ddna.lock`):** Freezes all generation parameters
- **Reproducibility:** Exact regeneration capability
- **Config Management:** Tracks LoRAs, ControlNets, LUTs, seeds

```python
registry = SeedRegistry("scene_001.lock")
config = GenerationConfig(
    seed=42,
    model="sdxl",
    controlnets={"depth": 0.8, "pose": 0.9}
)
registry.register_anchor("A0", config)
registry.lock()  # Saves to ddna.lock
```

## Workflow: 5-Frame Generation

**File:** `ddna/pipelines/advanced_frame_pipeline.py`

### Frame Structure
```
A0 (Anchor) → A1 (Interpolated) → A2 (Anchor) → A3 (Interpolated) → A4 (Anchor)
```

### Generation Process

1. **Define Anchors:** A0 (start), A2 (middle), A4 (end)
2. **Generate Anchors:** With full constraint enforcement
3. **Interpolate A1:** Between A0-A2 with pose/seed interpolation
4. **Interpolate A3:** Between A2-A4 with continuity constraints
5. **Validate All:** Check against 5 pillars
6. **Auto-Regenerate:** Failed frames with adjusted parameters
7. **Lock & Save:** Final configuration to `ddna.lock`

### Usage Example

```python
from ddna.pipelines import AdvancedFramePipeline
from ddna.core import SceneDNA

# Parse screenplay
dna = SceneDNA.from_screenplay("screenplay.fountain")

# Initialize advanced pipeline
pipeline = AdvancedFramePipeline(dna)

# Generate 5 consistent frames
frames = pipeline.generate_5_frames(
    first_frame=optional_reference,
    last_frame=optional_reference
)

# Get validation report
report = pipeline.get_validation_report()
print(f"Consistency Score: {report['overall_valid']}")
```

## Validation Metrics

| Pillar | Metric | Threshold | Action on Failure |
|--------|---------|-----------|-------------------|
| Identity | Cosine Similarity | > 0.92 | Re-roll frame |
| Palette | Delta E (CIEDE2000) | < 10.0 | Apply color correction |
| Composition | Position Error | < 0.1 | Adjust ControlNet |
| Props | Detection Confidence | > 0.8 | Regenerate with mask |
| Seed | Hash Match | Exact | Use locked seed |

## Configuration Files

### `ddna.lock` Structure
```json
{
  "version": "1.0",
  "configs": {
    "A0": {
      "seed": 42,
      "model": "stable-diffusion-xl",
      "controlnets": {"depth": 0.8, "pose": 0.9},
      "loras": {},
      "guidance_scale": 7.5
    }
  },
  "frame_seeds": {"A0": 42, "A1": 12345, ...}
}
```

## Next Steps

1. **Model Integration:** Connect actual AI models (SD, ControlNet)
2. **Advanced Encoders:** Implement real ArcFace/CLIP encoders
3. **Object Detection:** Add YOLO/Detectron2 for prop validation
4. **LUT Pipeline:** Integrate professional color grading LUTs
5. **UI Dashboard:** Build validation visualization interface
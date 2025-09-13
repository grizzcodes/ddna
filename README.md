# Screenplay DNA System (DDNA)

## Visual Consistency Through Scene DNA

A revolutionary approach to generating visually consistent scenes from screenplays using "Scene DNA" - a hierarchical constraint system that ensures frame-to-frame consistency while maintaining creative flexibility.

## 🎬 Overview

The Screenplay DNA System transforms written screenplays into visually consistent image sequences by:
- Extracting "DNA" from screenplay scenes (characters, locations, props, lighting, etc.)
- Pre-building reusable visual modules for efficiency
- Maintaining strict visual consistency through constraint locking
- Generating frames that maintain continuity across entire scenes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/grizzcodes/ddna.git
cd ddna

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and settings
```

### Basic Usage

```python
from ddna import SceneDNA, ScenePipeline

# Parse screenplay and extract DNA
dna = SceneDNA.from_screenplay('input/screenplays/sample_scene.fountain')

# Configure generation pipeline
pipeline = ScenePipeline(
    style='neo_noir',
    resolution=(1920, 1080),
    consistency_level='strict'
)

# Generate scene
result = pipeline.generate(dna)
print(f"Scene generated: {result.output_path}")
```

## 📁 Project Structure

```
ddna/
├── config/          # Configuration files and templates
├── core/            # Core DNA parsing and compilation engines
├── generators/      # Frame and module generation systems
├── validators/      # Consistency validation tools
├── modules/         # Pre-generated visual modules library
├── pipelines/       # End-to-end processing pipelines
├── utils/           # Utility functions and helpers
├── templates/       # Prompt and DNA templates
├── input/           # Input screenplays and references
├── output/          # Generated scenes and reports
└── models/          # AI model weights and checkpoints
```

## 🧬 Scene DNA Specification

Scene DNA is a structured representation of all visual elements in a scene:

```json
{
  "scene_id": "INT_OFFICE_NIGHT_001",
  "environment": {
    "location": "office",
    "time_of_day": "night",
    "lighting": "dim fluorescent"
  },
  "characters": [
    {
      "id": "DETECTIVE_JONES",
      "appearance": "...",
      "position": "desk"
    }
  ],
  "props": ["desk", "computer", "coffee_mug"],
  "style": {
    "genre": "neo_noir",
    "color_palette": ["#1a1a2e", "#16213e", "#e94560"]
  }
}
```

## 🔧 Features

- **DNA Extraction**: Automatically parse screenplays to extract visual elements
- **Module Pre-building**: Generate reusable visual components for efficiency
- **Constraint Locking**: Maintain consistency through hierarchical constraints
- **Multi-Model Support**: Works with Stable Diffusion, DALL-E, Midjourney
- **Validation Pipeline**: Pre and post-generation consistency checks
- **Batch Processing**: Handle multiple scenes efficiently
- **Style Transfer**: Apply consistent visual styles across scenes

## 📊 Workflow

1. **Parse** screenplay → Extract Scene DNA
2. **Pre-build** reusable modules (environments, characters, props)
3. **Generate** frames with locked constraints
4. **Validate** consistency across frames
5. **Report** generation metrics and consistency scores

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Stable Diffusion team for the base models
- Fountain screenplay format creators
- The open-source AI art community

## 📞 Contact

For questions and support, please open an issue on GitHub.

---

**Note**: This project is in active development. Features and APIs may change.
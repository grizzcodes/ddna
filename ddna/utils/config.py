"""Configuration management for DDNA."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    def __init__(self, config_path: Optional[str] = None):
        load_dotenv()
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "config"
        self.config = self._load_config(config_path)
        self._setup_paths()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        config = {
            "model_provider": os.getenv("MODEL_PROVIDER", "local"),
            "device": os.getenv("DEVICE", "cuda"),
            "resolution": os.getenv("DEFAULT_RESOLUTION", "1920x1080"),
            "consistency_level": os.getenv("CONSISTENCY_LEVEL", "strict"),
            "seed_locking": os.getenv("SEED_LOCKING", "true").lower() == "true",
            "cache_enabled": os.getenv("CACHE_ENABLED", "true").lower() == "true",
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith(('.yaml', '.yml')):
                    config.update(yaml.safe_load(f))
                elif config_path.endswith('.json'):
                    config.update(json.load(f))
        
        return config
    
    def _setup_paths(self):
        self.models_path = Path(os.getenv("MODELS_PATH", "./models"))
        self.cache_path = Path(os.getenv("CACHE_PATH", "./output/cache"))
        self.output_path = Path(os.getenv("OUTPUT_PATH", "./output/scenes"))
        
        for path in [self.models_path, self.cache_path, self.output_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
"""Seed management for consistent generation."""

import hashlib
from typing import Optional

class SeedManager:
    def __init__(self, base_seed: Optional[int] = None):
        self.base_seed = base_seed or 42
        self.seeds = {}
    
    def get_seed(self, identifier: str) -> int:
        if identifier not in self.seeds:
            hash_obj = hashlib.md5(f"{self.base_seed}_{identifier}".encode())
            self.seeds[identifier] = int(hash_obj.hexdigest(), 16) % (2**32)
        return self.seeds[identifier]
    
    def lock_seed(self, identifier: str, seed: int):
        self.seeds[identifier] = seed
    
    def reset(self):
        self.seeds = {}
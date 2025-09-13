"""Batch processing for multiple scenes."""

import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core import SceneDNA
from .scene_pipeline import ScenePipeline, PipelineResult

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process multiple scenes in batch."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.pipeline = ScenePipeline()
        
    def process_batch(self, 
                     dna_list: List[SceneDNA], 
                     frames_per_scene: int = 10) -> List[PipelineResult]:
        """Process multiple scenes in parallel."""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.pipeline.generate, 
                    dna, 
                    frames_per_scene
                ): dna 
                for dna in dna_list
            }
            
            # Collect results
            for future in as_completed(futures):
                dna = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {dna.scene_id}")
                except Exception as e:
                    logger.error(f"Failed {dna.scene_id}: {e}")
                    
        return results
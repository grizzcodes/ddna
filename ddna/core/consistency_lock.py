"""Consistency locking mechanism."""

from typing import Set, Dict, Any
import json

class ConsistencyLock:
    def __init__(self):
        self.locked_elements: Set[str] = set()
        self.locked_values: Dict[str, Any] = {}
    
    def lock(self, element_id: str, value: Any = None):
        self.locked_elements.add(element_id)
        if value is not None:
            self.locked_values[element_id] = value
    
    def unlock(self, element_id: str):
        self.locked_elements.discard(element_id)
        self.locked_values.pop(element_id, None)
    
    def is_locked(self, element_id: str) -> bool:
        return element_id in self.locked_elements
    
    def get_locked_value(self, element_id: str) -> Any:
        return self.locked_values.get(element_id)
    
    def export(self) -> str:
        return json.dumps({
            'locked_elements': list(self.locked_elements),
            'locked_values': self.locked_values
        })
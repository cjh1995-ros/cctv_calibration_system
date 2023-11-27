from typing import List, Dict, Any
import numpy as np




class BaseVisualizer:
    def __init__(self):
        pass
    
    def visualize(self, image: np.ndarray, data: Any) -> None:
        pass
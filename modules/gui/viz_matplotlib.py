from modules.gui import BaseVisualizer
from typing import Any, List
import matplotlib.pyplot as plt
import numpy as np




class MatplotVisualizer(BaseVisualizer):
    def __init__(self):
        pass
    
    def to_vis_data(self, data: Any) -> Any:
        pass
    
    def vis_images(self, images: List[Any], cameras: List[Any], data: Any) -> None:
        assert len(images) > 0
        assert len(images) == len(cameras)
        assert len(data) > 0
        
        for i, (image, camera) in enumerate(zip(images, cameras)):
            pts = camera.project(data) # (N, 2)
            plt.imshow(image)
            plt.scatter(pts[:, 0], pts[:, 1], s=1)
            plt.title(f"Image {i+1}")

        plt.show()
    
    def vis_3d(self, images: List[Any], cameras: List[Any], data: Any) -> None:
        pass
    
    def vis_satellite(self, satellite_image: Any, cameras: List[Any], data: Any) -> None:
        pass
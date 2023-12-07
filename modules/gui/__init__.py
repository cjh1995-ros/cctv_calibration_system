from typing import List, Dict, Any




class BaseVisualizer:
    def __init__(self):
        pass
    
    def to_vis_data(self, data: Any) -> Any:
        """_summary_
        Args:
            data (Any): It's basically point cloud. But there are so many types.
            So through this function, we will convert them into numpy array.
        Returns:
            Any: numpy array
        """
        pass
    
    def vis_images(self, images: List[Any], cameras: List[Any], data: Any) -> None:
        """_summary_
        Visualize images. The cameras and images should be same length.
        Visualize projected points on images.
        Args:
            images (Any): This is numpy array.
            cameras (List[Any]): This is the list of camera objects. Explain about projection model
            data (Any): This is numpy array. Point cloud or Feature etc.
        """
        pass
    
    def vis_3d(self, images: List[Any], cameras: List[Any], data: Any) -> None:
        """_summary_
        Visualize location of cameras and points in 3D space.
        Args:
            images (Any): This is numpy array.
            cameras (List[Any]): This is the list of camera objects. Explain about projection model
            data (Any): This is numpy array. Point cloud or Feature etc.
        """
        pass
    
    def vis_satellite(self, satellite_image: Any, cameras: List[Any], data: Any) -> None:
        """_summary_
        Visualize location of cameras and points in satellite image.
        Args:
            images (Any): This is numpy array.
            cameras (List[Any]): This is the list of camera objects. Explain about projection model
            data (Any): This is numpy array. Point cloud or Feature etc.
        """
        pass
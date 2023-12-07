from modules.gui import BaseVisualizer
from typing import Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R



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
        """Visualize camera positions, orientations (as rotation matrices), and 3D points."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot 3D points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b')

        # Plot cameras
        self._draw_cameras(ax, cameras)
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
    
    def vis_satellite(self, satellite_image: Any, cameras: List[Any], data: Any) -> None:
        fig, ax = plt.subplots()

        ax.scatter(data[:, 0], data[:, 1], s=1)
        
        self._draw_cameras(ax, cameras, length=0.1, dimension=2)
        
        # ax.imshow(satellite_image)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        plt.show()

    
    def _draw_cameras(self, ax, cameras: List[Any], length: float = 1.0 ,dimension: int = 3):
        """
        Plot camera position in ax. cameras is a list of Camera.
        dimension is either 2 or 3.
        """
        for camera in cameras:
            self._draw_camera(ax, camera, length, dimension)

    def _draw_camera(self, ax, camera: Any, length, dimension):
        """
        Plot camera position in ax. camera is a Camera.
        dimension is either 2 or 3.
        """
        assert dimension == 2 or dimension == 3

        Rmat = R.from_rotvec(camera.R).as_matrix()
        t = camera.t
        
        # transform to world coordinate
        Rmat = Rmat.T # inverse rotation
        t = -Rmat @ t # inverse translation
        
        x, y, z = t
        
        if dimension == 2:
            # Z direction is the direction of the camera in world coord
            direction = Rmat[:, 2]
            u, v, w = direction * length
            ax.quiver(x, y, u, v, color='b')
        
        if dimension == 3:
            for i in range(3):
                # Extract the direction for each axis
                direction = Rmat[:, i]
                u, v, w = direction * length
                ax.quiver(x, y, z, u, v, w, color=['b', 'g', 'r'][i])

from modules.gui import BaseVisualizer
from typing import Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R



class MatplotVisualizer(BaseVisualizer):
    def __init__(self, is_inv: bool = False):
        self.is_inv = is_inv
    
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
    
        self._normalizing_axis(ax)
    
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

    def compare_points(self, pts1, pts2, pts3):
        """Compare pts1 and pts2. Add vector from pts1 to pts2"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot pts1
        ax.scatter(pts1[:, 0], pts1[:, 1], pts1[:, 2], color='r')
        
        # Plot pts2
        ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2], color='b')

        # Plot pts3
        ax.scatter(pts3[:, 0], pts3[:, 1], pts3[:, 2], color='g')
    
        self._normalizing_axis(ax)
    
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
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
        if self.is_inv:
            Rmat = Rmat.T # inverse rotation
            t = -Rmat @ t # inverse translation
        
        x, y, z = t
        
        if dimension == 2:
            # Z direction is the direction of the camera in world coord
            direction = Rmat[:, 2]
            u, v, w = direction * length
            ax.quiver(x, y, u, v, color='b')
            ax.text(x, y, f'{camera.id}', color='black')
        
        if dimension == 3:
            for i in range(3):
                # Extract the direction for each axis
                direction = Rmat[:, i]
                u, v, w = direction * length
                ax.quiver(x, y, z, u, v, w, color=['b', 'g', 'r'][i])

            ax.text(x, y, z, f'{camera.id}', color='black')


    def _normalizing_axis(self, ax):
        """Set 3D plot axes to equal scale.
        Make axes of 3D plot have equal scale so that spheres appear as spheres and cubes as cubes.
        """
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
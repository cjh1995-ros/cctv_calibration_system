import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from modules.cameras.generator import Generator
from scipy.spatial.transform import Rotation as R


def plot_camera(ax, position, rotation_matrix, length=1.0):
    """Plot a representation of a camera with 3 orientation axes."""
    x, y, z = position
    
    # rotation_matrix = R.from_rotvec(rvec).as_matrix()
    
    for i in range(3):
        # Extract the direction for each axis
        direction = rotation_matrix[:, i]
        u, v, w = direction * length
        ax.quiver(x, y, z, u, v, w, color=['b', 'g', 'r'][i])

def visualize_cameras_and_points(camera_positions, camera_orientations, points_3d):
    """Visualize camera positions, orientations (as rotation matrices), and 3D points."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='b')

    # Plot cameras
    for position, orientation in zip(camera_positions, camera_orientations):
        plot_camera(ax, position, orientation)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Example data
camera_positions = np.array([[3, 0, 3], [0, 5, 5]])  # Replace with your camera positions
# Replace with your camera rotation matrices (3x3)
camera_orientations = [np.array([[1, 0, 0],
                                [0, 0, 1],
                                [0, -1, 0]]),
                       np.array([[0, 0, 1], 
                                [1, 0, 0], 
                                [0, -1, 0]])]
points_3d = np.random.rand(10, 3) * 10  # Replace with your 3D points

visualize_cameras_and_points(camera_positions, camera_orientations, points_3d)

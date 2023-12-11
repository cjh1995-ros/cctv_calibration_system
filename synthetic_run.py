from modules.markers.point import Feature2D, Feature3D
from modules.generator.generator import CameraGenerator, MarkerGenerator
from modules.gui.viz_matplotlib import MatplotVisualizer
from copy import deepcopy
import numpy as np
from modules.optimizer import Optimizer








def z_from_theta(theta: float, xy: np.ndarray):
    ...

 
if __name__ == '__main__':
    cameras = CameraGenerator().generate_default()
    markers, hinz = MarkerGenerator().generate_default()

    markers = np.array(markers).reshape(-1, 3)

    # Visualize the initial state
    viz = MatplotVisualizer(is_inv=True)
    viz.vis_3d(None, cameras, data=markers)
    viz.vis_satellite(None, cameras, data=markers)
    
    # Add noise to the data
    noise = np.random.normal(0, 0.01, markers.shape)
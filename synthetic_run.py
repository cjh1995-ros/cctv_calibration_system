from modules.generator.generator import CameraGenerator
from modules.gui.viz_matplotlib import MatplotVisualizer
import numpy as np




mv = MatplotVisualizer()


default_cameras = CameraGenerator().generate_default()
points_3d = np.array([0., 4.0, 0.])+ np.random.rand(10, 3) * 3  # Replace with your 3D points

mv.vis_3d(None, default_cameras, points_3d)

# get true position of pixels
pts = np.zeros()
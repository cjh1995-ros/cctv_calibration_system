from modules.cameras.projectors import BasicConvertor
from modules.board import Chessboard
from modules.frame import read_mono_frames, is_good_frame

from scipy.optimize import least_squares

from argparse import ArgumentParser
import autograd.numpy as np
from autograd import jacobian
# import numpy as np
import json


def object_function(params, frames, chess_board):
    # Unpack the params.
    camera_params = params[:frames[0].n_total]
    extrinsics = params[frames[0].n_total:].reshape(-1, 6)
    
    # Update the camera parameters.
    frames[0].params = camera_params
    
    all_corners = np.array([frame.corners for frame in frames[1]])
    
    points_3d = chess_board.points
    
    projected_points = []
    
    # Calculate the reprojection error.
    for extr in extrinsics:
        rvec = extr[:3]
        tvec = extr[3:]
        
        camera_point = BasicConvertor.world_to_camera(points_3d, R=rvec, t=tvec)
        projected_point = frames[0].project(camera_point)
        
        projected_points.append(projected_point)
        
    projected_points = np.array(projected_points)
    
    error = (projected_points.reshape(-1, 2) - all_corners.reshape(-1, 2)).ravel()
    
    return error


if __name__ == '__main__':
    # From parser, get the config file path.
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    
    args = parser.parse_args()
    
    
    # From the config file, get the config.
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    is_mono         = config['is_mono']
    is_fisheye      = config['is_fisheye']
    image_path      = config['image_path']
    
    board_pattern   = config['board_pattern']
    board_size      = config['board_size']
    
    intr_type       = config['intrinsic_type']
    proj_type       = config['projection_type']
    dist_type       = config['distortion_type']
    
    is_visualize    = config['visualize_results']
    
    # Read the images.
    frames = read_mono_frames(intr_type, proj_type, dist_type, image_path)
    
    # Initialize the board.
    chess_board = Chessboard()
    chess_board.pattern = board_pattern
    chess_board.size = board_size
    
    # Initialize the camera and find corners in frame.
    good_frame_idxs = []
    
    for i, frame in enumerate(frames[1]):
        frame.is_fisheye = is_fisheye
        find, frame.corners = chess_board.find_corners(frame)

        if find and is_good_frame(frame):
            ### Initialize the camera parameters.
            height, width = frame.shape
                        
            ## 1. Initialize the camera parameters.
            cx, cy = (width - 1) / 2, (height - 1) / 2
            f = width * 0.8            
            
            camera = frames[0]
            camera.init_intrinsic(np.array([f, cx, cy]))
            
            # Use PnP method for estimate the extrinsic parameters.
            if chess_board.find_pose(frame, camera):
                good_frame_idxs.append(i)
    
    # Update frames with good frames.
    frames[1] = [frame for frame in frames[1] if frame.idx in good_frame_idxs]
    
    
    # Init params for optimization // params, args
    camera_params = frames[0].params
    extrinsics = np.array([frame.transform for frame in frames[1] if frame.idx in good_frame_idxs])
    
    params = np.concatenate([camera_params, extrinsics.ravel()])
    args = (frames, chess_board)
    
    # Optimize the parameters.
    camera_bound = np.array(frames[0].gen_bounds()).reshape(-1, 2)
    extrinsic_bound = np.array([frame.gen_bounds() for frame in frames[1]]).reshape(-1, 2)
    
    bounds = np.concatenate([camera_bound, extrinsic_bound]).T
    
    res = least_squares(object_function, params,loss='soft_l1' ,bounds=bounds, jac='2-point', args=args)
    
    res_camera_params = res.x[:frames[0].n_total]
    res_extrinsics = res.x[frames[0].n_total:].reshape(-1, 6)
    
    # Update the camera parameters.
    frames[0].params = res_camera_params
    
    # Update the extrinsic parameters.
    for extr, frame in zip(res_extrinsics, frames[1]):
        frame.transform = extr
        
    # Show the results.
    print("Camera parameters: ")
    print(res_camera_params)
    print(res.cost)
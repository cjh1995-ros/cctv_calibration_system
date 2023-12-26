from modules.board import Chessboard
from modules.frame import read_frames, is_good_frame
from argparse import ArgumentParser
import json



if __name__ == '__main__':
    # From parser, get the config file path.
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    
    args = parser.parse_args()
    
    
    # From the config file, get the config.
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    is_mono         = config['is_mono']
    image_path      = config['image_path']
    board_pattern   = config['board_pattern']
    board_size      = config['board_size']
    
    intr_type       = config['intrinsic_type']
    proj_type       = config['projection_type']
    dist_type       = config['distortion_type']
    
    is_visualize    = config['visualize_results']
    
    # Read the images.
    frames = read_frames(image_path)
    
    # Extract the corners from the image.
    chess_board = Chessboard()
    
    for i, frame in enumerate(frames):
        chess_board.find_corners(frame)

        if is_good_frame(frame):
            ### Initialize the camera parameters.
            height, width = frame.shape
                        
            ## 1. Initialize the camera parameters.
            cx, cy = (width - 1) / 2, (height - 1) / 2
            f = width * 0.8            
            
            ## 2. Initialize the distortion parameters.
            # camera.distortion_params = k1, k2, p1, p2
            
            ## 3. Initialize the extrinsic parameters.
            # Use PnP method for estimate the extrinsic parameters.
                
    
    ## 4. Optimize the parameters.
    # Init params for optimization.
    inititial_params = # np.concatenate([camera.initial_params, camera.distortion_params])
    
    ## 5. Visualize the results.
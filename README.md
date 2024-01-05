# This is repository for calibrating diverse camera model

There are diverse camera models.

- Brown-Conrady: perspective projection + polynomial/tangential distortion (k1, k2, p1, p2, k3)
- Kannala-Brandt: equidistant projection + polynomial distortion (k1, k2, k3, k4)
- Field-of-View: perspective projection + field of view distortion (w)
- Single Sphere Camera Model(SSCM): alpha
- Double Sphere Camera Model(DSCM): xi, alpha
- Triple Sphere Camera Model(TSCM): xi, lambda, alpha
- Quadra Sphere Camera Model(QSCM): gamma, xi, lambda, alpha

fx, fy, cx, cy, (distortion params), extrinsic

# How to build

This repository is composed of python only. 
So i recommends you to use anaconda env.

```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./DSCM_calib ../test.json
```

In the json file, you should specify keywords like below.

```json
{
    "data_path": "../data/chessboard",    
    "board_pattern": [6, 9],
    "board_size": [0.05, 0.05],
    // "camera_model": "SingleSphere",
    // "camera_model": "DoubleSphere",
    "camera_model": "QuadraSphere",
    // "camera_model": "TripleSphere",
    // "camera_model": "BrownConrady",
    // "camera_model": "KannalaBrandt",
    "output_path": "../data/",
    "output_name": "result.json"
}
```

## Further works

Find good initial values with fov.

### TODO - Develop

- [X] From Scipy to Ceres-Solver 
- [x] Chessboard
- [ ] Multi camera mode

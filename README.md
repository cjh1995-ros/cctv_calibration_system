# This is repository for calibrating diverse camera model


There are diverse camera models which explain how 3d point clouds project into an image plane.
There are so many projection method,

- Perspective projection
- Equidistant projection
- Stereographic projection
- Orthogonal projection
- Sphere projection
- ...

And there are some special projection like `Unified Camera Model(UCM)` which slightly move sphere for projection.
Following this work, there are some models which are composed of several moved spheres. 
`Double Sphere Camera Model(DSCM)`, `Triple Sphere Camera Model(TSCM)`... 
I made `Quadruple Sphere Camera Model` which is composed of 4 parameters for omni camera.

But there will be other distortion models which will describe or comportable for OpenCV etc.
Like ```polynomial distortion parameters``` like k_1, k_2, k_3. ```Field-of-View``` distortion parameters. 
And last case, for easy mode, equidistant distortion. 

## Supported Models

I specify camera matrix parameters into three cases like below.

- `FOCAL`: f
- `FCXCY`: f, cx, cy
- `FXYCXY`: fx, fy, cx, cy

You can choose the camera matrix version you want to calibrate.

### Supported Camera Projection Models

- Perspective Projection: `None`
- Equidistant Projection: `None`
- Single Sphere Projection (Unified Camera Model): `alpha`
- Double Sphere Projection (Double Sphere Camera Model): `alpha, xi`
- Triple Sphere Projection (Triple Sphere Camera Model): `alpha, xi, delta`
- Quadruple Sphere Projection (Quadruple Sphere Camera Model): `alpha, xi, delta, zeta`

### Supported Camera Distortion Models

- Polynomial Distortion: `k1, k2, k3, k4`
- Field-of-View Distortion: `w1`
- Equdistant Distortion: `None`

For OpenCV users, OpenCV supports two camera models, `Brown-Conrady`, `Kannala-Brandt`.
Each models will be same with `Perspective Projection` + `Polynomial Distortion`, `Equidistant Projection` + `Polynomial Distortion`.


## Keywords list of supported models with parameters

### Camera Matrix

- `FOCAL`: `f`
- `FCXCY`: `f, cx, cy`
- `FXYCXY`: `fx, fy, cx, cy`

### Projection Models

- `PERSPECTIVE`: None 
- `EQUIDISTANT`: None
- `SINGLE`: `alpha` (Single sphere projection)
- `DOUBLE`: `alpha, xi` (Double sphere projection)
- `TRIPLE`: `alpha, xi, delta` (Triple sphere projection)
- `QUADRA`: `alpha, xi, delta, zeta` (Quadraple sphere projection)

### Distortion Models

- `POLY_1`: `k1`
- `POLY_2`: `k1, k2`
- `POLY_3`: `k1, k2, k3`
- `POLY_4`: `k1, k2, k3, k4`
- `FOV`: `w` (Field-of-View distortion)
- `EQUIDISTANT`: None

## How to use

### How to setup.

This repository is composed of python only. 
So i recommends you to use anaconda env.

```bash
$ conda env create -f conda.yaml
$ conda activate diverse_camera_calib
```

```
$ python calib.py --calib_info info.json
```

In the json file, you should specify keywords like below.

```json
{
    "img_path":"path/to/image",
    "result_path":"path/to/result",
    "chessboard_pattern": [5, 5],
    "size_of_chessboard": 0.05, # unit [meter]
    "camera_matrix":"FCXCY",
    "projection":"PERSPECTIVE",
    "distortion":"POLY_1",
    "field_of_view": {
        "horizon": 120.0,
        "vertical": 90.0
    }
}
```

## Further works

There will be sample code for undistorting, triangulation, PnP method, how to use with OpenCV.

As the result of SciPy least square optimization is so bad, I decided to develop the code with Ceres-Solver.
There will be code for C++ soon.


### TODO - Develop

- [ ] From Scipy to Ceres-Solver 
- [x] Chessboard
- [ ] Multi camera mode

### TODO - Sample Code

- [x] Sample code for undistorting images
- [ ] Sample code for triangulation
- [ ] Sample code for PnP method
- [ ] Sample code for OpenCV

### TODO - README

- [ ] Add how to use
- [ ] Add research paper lists for each models
- [ ] Performance test of speed

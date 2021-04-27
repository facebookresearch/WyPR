## Getting Started
The code is structured for mainly three functionality: pre-processing (shape_det), proposal generation (gss), weakly-supervised recognition (wypr).

### Shape Detection
We use the open-source CGAL librabry to detecte shapes from points clouds. This pre-precessing step needs to be done before computing proposals or launch training.
```bash
# Complie our modified C++ code, this will require CGAL
# Clone the repo in recursice model so that cgal will be downloaded
# To learn more: https://cgal.geometryfactory.com/CGAL/doc/master/Shape_detection/index.html#Shape_detection_RegionGrowing
# Use Cmake 3.1 to 3.15 (e.g., module load cmake/3.13.3/gcc.7.3.0)
cd shape_det
mkdir build; cd build
cmake -DCGAL_DIR="$(realpath ../../3rd_party/cgal/)" -DCMAKE_BUILD_TYPE=Debug ../ 
make        
# Usage: ./region_growing_on_point_set_3 input(*.xyz) output(*.ply) output(*.txt)
# To test whether it's built correctly
./region_growing_on_point_set_3 ../data/point_set_3.xyz point_set_3.ply point_set_3.txt
# You can visualize ../data/point_set_3.xyz and point_set_3.ply using tools like meshlab.
# The index assignment is saved as point_set_3.txt where 
# each row represents one shape and the last row is the un-assigned points.
cd ../..
```
**Known Issues**
0. Make sure eigen is installed `sudo apt install libeigen3-dev`.
1. `Could NOT find GMP (missing: GMP_LIBRARIES GMP_INCLUDE_DIR)`, solve with `sudo apt-get install libgmp10 libgmp-dev`.
2. `Could NOT find MPFR (missing: MPFR_LIBRARIES MPFR_INCLUDE_DIR)`, solve with `sudo apt install libcgal-dev`. ([source](https://github.com/PyMesh/PyMesh/issues/96))
3. `fatal error: GL/gl.h: No such file or directory`, try `sudo apt install mesa-common-dev`

To pre-process the datasets we use (e.g., ScanNet), do
```bash
# 1st: Convert data from *.ply into *.xyz which CGAL can use
#      You should open some *.xyz files in meshlab to make sure things are correct
# 2nd: Generate running scripts
# Note: you need to change the `data_path` to be the absolute path of output
python shape_det/generate_scripts.py

# Running
cd shape_det/build
# Use the generated *.sh files here to detect shapes
sh *.sh
# Results will be saved in *.txt files under shape_det/build/

# Pre-compute the adjancency matrix between detected shapes
python shape_det/preprocess.py
```

### Geometric Selective Search (GSS)
We provide standalone code to compute 3D box proposals in an unsupervised manner, together with the evaluation, and visualizaztion code.

To generate proposals for ScanNet, do
```bash
cd gss
# Compute proposals for a single policy
# Change the setting in main funtion to different policies
python selective_search_3d_run.py

# Ensemble multiple runs
python selective_search_3d_ensemble.py

# Evaluate the MABO and AR
python selective_search_3d_eval.py --split val --policy size
```

The pre-computed 3D proposals using GSS can be found at:

| Dataset | Methods | MABO | AR | url | 
|---------|---------|------|----|-----|
| ScanNet | GSS (unsupervised) | 0.378 | 86.2 | [link]() |
| ScanNet | GSS                | 0.409 | 89.3 | [link]() |
|  S3DIS  | GSS (unsupervised) | 0.412 | 84.9 | [link]() |
|  S3DIS  | GSS                | 0.441 | 88.3 | [link]() |

### WyPR Running
We use [Hydra]() to for configuration. For a single-run of training (e.g., Scannet):
```bash
python tools/train.py model=seg_det_net_ts \
    distrib_backend=ddp backbone=pointnet2 num_point=40000 \
    batch_size=32 learning_rate=0.003 seg_pseudo_label_th=0.9 \
    hydra.run.dir=/path/to/outputs/
```

For sweeping parameters (e.g., batch_size)
```bash
python tools/train.py model=seg_det_net_ts \
    distrib_backend=ddp backbone=pointnet2 num_point=40000 \
    batch_size=32,24,48 learning_rate=0.003 seg_pseudo_label_th=0.9 \
    hydra.sweep.dir=/path/to/outputs/ -m
```


The pre-trained WyPR models are provided here.
Numbers are evaluated on validation set.

| Dataset | Methods | mIoU | AP@IoU=0.25 | url | scripts | 
|---------|---------|------|-------------|-----|---------|
| ScanNet | WyPR | 29.6 | 18.3 | [link](link) | [scripts](link) |
| S3DIS   | WyPR | 22.3 | 19.3 | [link](link) | [scripts](link) |

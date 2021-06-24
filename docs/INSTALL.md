## Installation

It is required that you have access to GPUs. 
The code is tested with Ubuntu 18.04, Pytorch v1.7, TensorFlow v1.14, CUDA 10.1.243, and cuDNN v7.6 
To install the dependecies, do the followings from a clean conda environment.

```bash
# Conda environment
conda create --name wypr python=3.7
conda activate wypr

# Install the dependencies
pip install seaborn matplotlib opencv-python plyfile 'trimesh>=2.35.39,<2.35.40' 'networkx>=2.2,<2.3' open3d scikit-image --user

# PyTorch (e.g., cuda-10.1)
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install tensorflow for TensorBoard
pip install tensorflow-cpu

# Install hydra and pytorch-lighting
pip install hydra-core pytorch-lightning
```

To install WyPR, pleae do
```bash
git clone --recursive https://github.com/facebookresearch/WyPR

# Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413),
# which we used as the backbone network
cd WyPR/wypr/modeling/backbone/pointnet2/
python setup.py install
cd ../../../..

# Install WyPR
pip install -e .
```

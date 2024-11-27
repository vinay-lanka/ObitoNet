# ObitoNet

## Install
PyTorch = 1.7.0 < 1.11.0; python = 3.10; CUDA = 11.8;

```
# Chamfer Distance
cd ./extensions/chamfer_dist
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
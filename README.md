# ObitoNet

## Overview
ObitoNet is a deep learning framework designed for advanced 3D point cloud and image-based tasks. It combines a Point Cloud Encoder, Cross Attention Decoder, and the potential for an Image Encoder (future integration). ObitoNet excels in tasks requiring high fidelity and precision, such as augmented reality, robotics, and 3D reconstruction.

A link to the video showcasing ObitoNet can be found [here](https://www.youtube.com/watch?v=xC1fuEzKreo).

<p align="center">
  <img src="media/obitonet.png" alt="ObitoNet Architecture" />
</p>

## Components
1. **Point Cloud Encoder:** Processes 3D point cloud data into meaningful tokens using Transformers.
2. **Cross Attention Decoder:** Combines encoded data for accurate reconstruction and representation.
3. **Image Encoder:** Planned component for processing 2D image data in conjunction with point clouds.


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

## 4. Training
To pretrain Point-MAE on ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config configs/config.yaml --exp_name <output_file_name>
```

## Visualization

Visulization of pre-trained model on ShapeNet validation set, run:

```
python main_vis.py --test --ckpts <path/to/pre-trained/model> --config configs/config.yaml--exp_name <name>
```

### Dataset Prep 

For uniform downsampling, I used different k_values for different data-subsets, as some values were giving us samples less than 16k (less than fps output)
k = 100 (for all)
k = 20 (for caterpillar)

### 
To run the terminal interactively:
```bash
bash-4.4$ srun --gres=gpu:rtxa5000:4 --account=class --partition=class --qos high -t 1-00:00:00 --mem-per-cpu=64gb --pty bash -i
```
# DeepBranchTracer

A Generally-Applicable Approach to Curvilinear Structure Reconstruction Using Multi-Feature Learning

[![Paper](https://img.shields.io/badge/Paper-AAAI%202024-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/28121)
[![Python](https://img.shields.io/badge/Python-3.7+-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red)](https://pytorch.org/)

DeepBranchTracer is a novel method for reconstructing curvilinear structures from 2D and 3D images. It leverages deep learning to extract both image and geometric features, enabling accurate and continuous reconstruction of line-like objects, such as roads, vessels, and neurons.

## Features

- Support for both **2D** and **3D** curvilinear structure reconstruction
- Multi-feature learning combining image features and geometric features
- LSTM-based direction and radius prediction
- Multiple tracing strategies: `centerline`, `angle`, `anglecenterline`
- Support for multiple datasets: DRIVE, CHASEDB1, ROAD (2D) and FMOST (3D)

## Project Structure

```
DeepBranchTracer/
├── configs/                    # Configuration files
│   ├── config_2d.py           # 2D model configuration
│   └── config_3d.py           # 3D model configuration
├── models/                     # Model definitions
│   ├── Models_2D.py           # CSFL_Net_2D model
│   └── Models_3D.py           # CSFL_Net_3D model
├── tools/                      # Utility tools
│   ├── tracing/               # Tracing algorithms
│   │   ├── tracing_tools_2D.py
│   │   └── tracing_tools_3D.py
│   ├── Data_Loader_2d.py      # 2D data loader
│   ├── Data_Loader_3d.py      # 3D data loader
│   ├── Image_Tools.py         # Image processing utilities
│   ├── Losses.py              # Loss functions
│   └── dataset.py             # Data augmentation
├── lib/                        # External libraries
│   ├── klib/                  # Basic I/O utilities
│   └── swclib/                # SWC file handling
├── data/                       # Dataset folder
│   ├── DRIVE/                 # DRIVE dataset
│   ├── CHASEDB1/              # CHASEDB1 dataset
│   └── ROAD/                  # ROAD dataset
├── prepare_datasets_2D.py     # 2D dataset preparation
├── prepare_datasets_3D.py     # 3D dataset preparation
├── train_2D.py                # 2D training and inference
└── train_3D.py                # 3D training and inference
```

## Requirements

```bash
# Core dependencies
torch>=1.8.0
torchvision
numpy
scipy
scikit-image
Pillow
opencv-python
tifffile
tensorboardX
matplotlib
rtree
pandas

# Install dependencies
pip install torch torchvision numpy scipy scikit-image Pillow opencv-python tifffile tensorboardX matplotlib rtree pandas
```

## Dataset Preparation

### Data Structure

Organize your data in the following structure:

```
data/
└── DATASET_NAME/
    ├── training/
    │   ├── images_color/      # Original color images (.tif)
    │   ├── labels/            # Ground truth labels (.tif)
    │   ├── mask/              # Image masks (.tif)
    │   └── swc/               # Ground truth SWC files (.swc)
    └── test/
        ├── images_color/
        ├── labels/
        ├── mask/
        └── swc/
```

### Generate Training Patches

**For 2D datasets:**

```bash
# DRIVE dataset
python prepare_datasets_2D.py --datasets_name DRIVE \
    --image_dir ./data/ \
    --train_dataset_root_dir ./data/DRIVE/training_data/

# CHASEDB1 dataset
python prepare_datasets_2D.py --datasets_name CHASEDB1 \
    --image_dir ./data/ \
    --train_dataset_root_dir ./data/CHASEDB1/training_data/

# ROAD dataset
python prepare_datasets_2D.py --datasets_name ROAD \
    --image_dir ./data/ \
    --train_dataset_root_dir ./data/ROAD/training_data/
```

**For 3D datasets:**

```bash
python prepare_datasets_3D.py --datasets_name FMOST \
    --image_dir ./data/ \
    --train_dataset_root_dir ./data/FMOST/training_data/
```

## Training

### 2D Model Training

Training consists of two stages:

**Stage 1: Train segmentation branch**

```bash
python train_2D.py --gpu_id 0 --train_or_test train --train_seg True
```

**Stage 2: Train tracing branch (freeze segmentation)**

```bash
python train_2D.py --gpu_id 0 --train_or_test train --lr 2e-4 --to_restore True
```

### 3D Model Training

```bash
# Stage 1: Train segmentation branch
python train_3D.py --gpu_id 0 --train_or_test train --train_seg True

# Stage 2: Train tracing branch
python train_3D.py --gpu_id 0 --train_or_test train --lr 2e-4 --to_restore True
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_id` | 0 | GPU device ID |
| `--batch_size` | 64 (2D) / 16 (3D) | Batch size |
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--train_seg` | False | Train segmentation branch only |
| `--to_restore` | False | Resume from checkpoint |

## Inference

### Step 1: Generate Segmentation Maps

```bash
python train_2D.py --gpu_id 0 --train_or_test inference_segmentation
```

### Step 2: Run Tracing

**Fast tracing (recommended):**

```bash
python train_2D.py --gpu_id 0 --train_or_test inference_fastdeepbranchtracer
```

**Full tracing:**

```bash
python train_2D.py --gpu_id 0 --train_or_test inference_deepbranchtracer
```

### 3D Inference

```bash
# Generate segmentation
python train_3D.py --gpu_id 0 --train_or_test inference_segmentation

# Run tracing
python train_3D.py --gpu_id 0 --train_or_test inference_fastdeepbranchtracer
```

## Configuration

Modify parameters in `configs/config_2d.py` or `configs/config_3d.py`:

```python
# Dataset paths
parser.add_argument('--dataset_img_path', default='path/to/training_datasets/')
parser.add_argument('--test_data_path', default='path/to/test/images/')

# Model settings
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--vector_bins', type=int, default=50)

# Dataset-specific settings
parser.add_argument('--resize_radio', type=float, default=2.0)  # DRIVE: 2.0, CHASEDB1: 1.5, ROAD: 1.0
parser.add_argument('--r_resize', type=float, default=15)
```

## Results Visualization

It is recommended to use **neuTube** software to visualize images and SWC files.

- Download neuTube: [https://neutracing.com/](https://neutracing.com/)

## Citation

If you found this project useful, please give us a star ⭐ or cite us in your work:

```bibtex
@inproceedings{liu2024deepbranchtracer,
  title={DeepBranchTracer: A Generally-Applicable Approach to Curvilinear Structure Reconstruction Using Multi-Feature Learning},
  author={Liu, Chao and Zhao, Ting and Zheng, Nenggan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3548--3557},
  year={2024}
}
```

## License

This project is released for academic research use only.

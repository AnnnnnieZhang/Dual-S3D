# Single-view 3D Scene Reconstruction with High-fidelity Shape and Texture (M3D Framework)

---
## Overview

Single-view 3D reconstruction aims to recover the complete 3D geometry and appearance of objects from a single RGB image and its corresponding camera parameters. Yet, the task remains challenging due to incomplete image information and inherent ambiguity. Existing methods primarily encounter two issues: balancing extracting local details with the construction of global topology and the interference caused by the early fusion of RGB and depth features in high-texture regions, destabilizing SDF optimization. We propose Dual-S3D, a novel single-view 3D reconstruction framework to address these challenges. Our method employs a hierarchical dual-path feature extraction strategy based on stages that utilize CNNs to anchor local geometric details.

---
## Workflow
The pipeline of M3D is shown below:

![Dual-S3D Reconstruction Process](readme/process.pdf)

---

## Results Show

### Reconstruction

![Dual-S3D Geometry Compare Results](readme/results.pdf)


## Prepare Start
### Environment Prepare
Follow these steps to set up the environment:

```bash
conda create -n DualS3D python=3.8 
conda activate DualS3D

pip install -r requirements.txt
```
---
If there are any environment problems, welcome to give us comments to discuss about it.

### Datasets Prepare

We use the **3D-FRONT** dataset as the main data source for training and testing.

**Dataset Reference**:
```bibtex
@article{fu20203dfront,
  title={3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics},
  author={Fu, Huan and Cai, Bowen and Gao, Lin and Zhang, Lingxiao and Li, Cao and Zeng, Qixun and Sun, Chengyue 
          and Fei, Yiyun and Zheng, Yu and Li, Ying and Liu, Yi and Liu, Peng and Ma, Lin and Weng, Le and Hu, Xiaohang
          and Ma, Xin and Qian, Qian and Jia, Rongfei and Zhao, Binqiang and Zhang, Hao},
  journal={arXiv preprint arXiv:2011.09127},
  year={2020}
}
```
**Dataset Split**  
We split the datasets as train / val / test into 'data_split' folder


**Dataset Prepare**  
Download the dataset from the following link:

The dataset can be downloaded from: [3DFRONT](https://drive.google.com/file/d/1j0n4J7XBqK1np5v7sxZGKBhqMg6qTG4Y/view)

After downloading, extract it to the 'data' folder in the project root directory.

---
## Depth Prior Generation

```bash
cd depth_pri
python run.py --encoder vitb --img-path ../data/FRONT3D/train/rgb --outdir ../depth_anything_png  --pred-only  --grayscale
python run.py --encoder vitb --img-path ../data/FRONT3D/val/rgb --outdir ../depth_anything_png  --pred-only  --grayscale
python run.py --encoder vitb --img-path ../data/FRONT3D/test/rgb --outdir ../depth_anything_png  --pred-only  --grayscale
cd ..
python utils/depth_utils/png_tran_npy.py
```
---
For more depth estimation command rules:
```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```
Arguments:
- ``--img-path``: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- ``--pred-only`` is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.
- ``--grayscale`` is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.
---
Because we use limited GPU, so we choose offline to get the depth prior information.
Here, we choose the DepthAnthing to generate the depth prior, and you could choose any other depth estimation method to get the depth information prior.
And after this step, we could get the 'depth_anything' folder, which saved the depth prior information

reference:
```bibtex
@inproceedings{depthanything,
      title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
      author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
      booktitle={CVPR},
      year={2024}
}
```

## Train
```bash
python train.py --config train.yaml
```
we set the batchsize = 30, lr = 0.00006, you can try more parameters.

set show_rendering=False

## Inference
```bash
python inference.py --config train.yaml
```

set show_rendering=False, eval.export_mesh=True, eval.export_color_mesh=True


## evaluation
In preparing......


## Project Status

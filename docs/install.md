# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n maptr python=3.10 -y
conda activate maptr
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Recommended torch>=1.9
```

**c. Install mmcv-full.**
```shell
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

**d. Install mmdet and mmseg.**
```shell
mim install mmdet
pip install "mmsegmentation>=1.0.0"
```

**e. Install timm.**
```shell
pip install timm
```

**f. Install mmdet3d and GKT**
```shell
cd /path/to/MapTR/mmdetection3d
pip install -v -e .

cd /path/to/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
pip install -v .

```

**g. Install other requirements.**
```shell
cd /path/to/MapTR
pip install -r requirement.txt
```

**h. Prepare pretrained models.**
```shell
cd /path/to/MapTR
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```


# SGI 2025: Motion Planning for Broken Snake Robots

This repository is a lighter version of the code for "Inverse Geometric Locomotion", trimmed for the needs of the summer school project "Motion Planning for Broken Snake Robots".

## Installation

First, clone the repository recursively as follows

```bash
git clone --recurse-submodules -j8 git@github.com:qbecky/sgi_igl.git
```

Or if you already cloned the repository, run

```bash
git submodule update --init --recursive
```

Simply create a conda environment and install the dependencies

```bash
conda create --name sgi_igl_env
conda activate sgi_igl_env
conda install -y scipy=1.10 matplotlib numpy=1 jupyterlab tqdm opencv absl-py
conda install -y pytorch::pytorch -c pytorch
conda install -y -c conda-forge nbstripout
pip install ffmpeg-python
pip install libigl
nbstripout --install
```

## Run code

Simply activate your environment and run 

```
jupyter lab
```

## 3D rendering

We use mayavi to render our 3D meshes. You can run the scripts located in `rendering/` after creating the following python environment

```bash
conda create --name sgi_igl_rendering_env python=3.9
conda activate sgi_igl_rendering_env
conda install -y scipy=1.10 matplotlib numpy=1.26 jupyterlab tqdm opencv absl-py mayavi=4.8.1 vtk=9.2.6 pyqt=5.15.7
pip install ffmpeg-python
```


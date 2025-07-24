<p align="center">

  <h1 align="center">SGI 2025: Motion Planning for Broken Snake Robots</h1>

  ![Teaser](./images/Teaser.jpg)

  <p align="center">
    <br />
    <a href="https://olligross.github.io/"><strong>Oliver Gross</strong></a>
    · 
    <a href="https://qbecky.github.io/"><strong>Quentin Becker</strong></a>
    <br />
  </p>
</p>

## About

This repository is a lighter version of the code for [Inverse Geometric Locomotion](https://go.epfl.ch/igl/), trimmed for the needs of the summer school project "Motion Planning for Broken Snake Robots".

## Code Structure

[TODO]

## Installation

First, clone the repository recursively as follows

```bash
git clone --recurse-submodules -j8 git@github.com:qbecky/sgi_igl.git
```

Or if you already cloned the repository, run

```bash
git submodule update --init --recursive
```

Create a conda environment and install the dependencies

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

## Run Code

Simply activate your environment and run 

```
jupyter lab
```


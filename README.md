# PDS
This repository provides the code for testing Streetwise on D4RL MuJoCo Gym datasets: HalfCheetah, Hopper, Walker2D.

To train models using Implicit Q-learning (IQL):
```
python main.py
```

To train the LSTM autoencoders:
```
bash LSTM-AE/run_all.sh
```

To evaluate Streetwise on disturbed variants of the MuJoCo Gym locomotion environments:
```
bash run_infer.sh
```

# MuJoCo Installation Guide

## Overview
MuJoCo (Multi-Joint dynamics with Contact) is a physics engine that facilitates research and development in robotics, biomechanics, graphics and animation, and machine learning. It was acquired and open-sourced by DeepMind, making it freely accessible to everyone.

## Prerequisites for Rendering (All MuJoCo Versions)
MuJoCo uses one of three backends for rendering: glfw, osmesa, or egl. Note that:
- glfw will not work in headless environments
- osmesa will not run on GPU
- egl is recommended for most use cases

### System Dependencies
If you have sudo access:
```bash
sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
```

Alternative installation via conda:
```bash
conda activate mujoco_env
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c anaconda mesa-libgl-cos6-x86_64
conda install -c menpo glfw3
```

### Setting Environment Variables
```bash
conda env config vars set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
conda deactivate && conda activate mujoco_env
```

## Installation Options

### 1. New Bindings (≥ 2.1.2)
This is the recommended method for new installations:

```bash
conda create -n mujoco_env python=3.9
conda activate mujoco_env
pip install mujoco
```

### 2. Old Bindings (≤ 2.1.1): mujoco-py

#### Initial Setup
```bash
conda create -n mujoco_env python=3.9
conda activate mujoco_env
mkdir ~/.mujoco
cd ~/.mujoco

# For version 2.1.0
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz

# For versions < 2.1.0, get the license file
wget http://roboti.us/file/mjkey.txt
```

#### Environment Configuration
```bash
# Configure environment variables
conda env config vars set MJLIB_PATH=/path/to/home/.mujoco/mujoco210/bin/libmujoco210.so \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/home/.mujoco/mujoco210/bin \
    MUJOCO_PY_MUJOCO_PATH=/path/to/home/.mujoco/mujoco210

# For versions < 2.1.0, add license path
conda env config vars set MUJOCO_PY_MJKEY_PATH=/path/to/home/.mujoco/mjkey.txt

# Reload environment
conda deactivate && conda activate mujoco_env
```

#### Installation Options

##### Option 1: Via pip (Basic)
```bash
pip install mujoco-py
```

##### Option 2: From source (Recommended)
```bash
cd path/to/where/mujoco-py/must/be/cloned
git clone https://github.com/openai/mujoco-py
cd mujoco-py
python setup.py develop
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

## Common Issues and Solutions

1. **GL/glew.h not found**
   ```bash
   # Ubuntu
   sudo apt-get install libglew-dev libglew
   # CentOS
   sudo yum install glew glew-devel
   # Conda
   conda install -c conda-forge glew
   ```

2. **GL/gl.h missing**
   ```bash
   conda install -y -c conda-forge mesalib
   ```

3. **GLIBCXX_3.4.29 not found**
   ```bash
   conda install libgcc -y
   export LD_PRELOAD=$LD_PRELOAD:/path/to/conda/envs/compile/lib/libstdc++.so.6
   ```

4. **patchelf missing**
   ```bash
   pip install patchelf
   ```

5. **libOpenGL.so.0 undefined symbol**
   ```bash
   conda install -y -c conda-forge libglvnd-glx-cos7-x86_64 --force-reinstall
   conda install -y -c conda-forge xvfbwrapper --force-reinstall
   conda env config vars set LD_PRELOAD=/path/to/conda/envs/mujoco_env/x86_64-conda-linux-gnu/sysroot/usr/lib64/libGLdispatch.so.0
   ```

6. **GLFW initialization error**
   - Set environment variable: `MUJOCO_GL=egl python myscript.py`

7. **Black rendered images**
   - Ensure you call `env.render()` before reading pixels

8. **X11/Xlib.h missing**
   ```bash
   # Ubuntu
   sudo apt install libx11-dev
   # CentOS
   sudo yum install libX11
   # Conda
   conda install -c conda-forge xorg-libx11
   ```

9. **GL/osmesa.h missing**
   ```bash
   # Ubuntu
   sudo apt-get install libosmesa6-dev
   # CentOS
   sudo yum install mesa-libOSMesa-devel
   # Conda
   conda install -c menpo osmesa
   ```

## Verification
After installation, verify your setup:
```python
import mujoco_py
print(mujoco_py.cymj)  # Should contain "linuxgpuextensionbuilder" if GPU is enabled
```

Note: For GPU-enabled environments using job schedulers like Slurm, you may need to set the `GPUS` environment variable to match the global device ID (can be obtained from `SLURM_STEP_GPUS`).

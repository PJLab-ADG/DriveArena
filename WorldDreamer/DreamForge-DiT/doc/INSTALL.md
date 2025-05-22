# Installation

All the codes are tested in the following environment:

Linux (tested on Ubuntu 22.04)

CUDA 12.1

```bash
# Install a virtual environment.
conda create -n dreamforge-DiT python=3.10
conda activate dreamforge-DiT
```

Install `torch==2.4.0`, `torchvision==0.19.0`, and `torchaudio==2.4.0`

```bash
# CUDA 12.1
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```



# Install third-party packages

1. Install Colossalai
```bash
git clone https://github.com/flymin/ColossalAI.git
cd ColossalAI
BUILD_EXT=1 pip install .
```

2. Install xformer
```bash
pip install xformers==0.0.27.post1
```

3. Install flash-attn
```bash
pip install flash-attn==2.7.2.post1
```

4. Install diffusers
```bash
pip install diffusers==0.30.0
```

5. Install the dependencies
```bash
cd ..
pip install -r requirements.txt
```

6. Install apex 
```bash
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

7. Install mmdet_plugin
```bash
cd dreamforgedit/
pip install -v -e .
```

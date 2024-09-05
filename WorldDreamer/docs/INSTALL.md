# Installation

All the codes are tested in the following environment:

Linux (tested on Ubuntu 22.04)

CUDA 11.3 or higher

```bash
# Install a virtual environment.
conda create -n dreamer python=3.9
conda activate dreamer
```

Install `nuplan-devkit` from source

```bash
cd WorldDreamer/third_party/nuplan-devkit
pip install -r requirements.txt
pip install -e .
```

Install `Pytorch==1.10.2` and `torchvision==0.11.3`

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install the source code for other third-party packages, with `cd ${FOLDER}; pip install -e .`

```
# Install third-party packages
third_party/
├── bevfusion -> minorly change on db75150
├── diffusers -> based on v0.17.1 (afcca3916)
└── xformers -> minorly change 0.0.19 to install with pytorch1.10.2
```

Install the dependencies of WorldDreamer
```bash
cd ..
pip install -r requirements.txt
```
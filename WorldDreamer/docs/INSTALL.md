# Installation

All the codes are tested in the following environment:

Linux (tested on Ubuntu 22.04)

CUDA 11.3 or higher

```
# Install a virtual environment.
conda create -n dreamer python=3.9
conda activate dreamer
```

Install Pytorch==1.10.2 and torchvision==0.11.3

```
# Install Dependencies
cd WorldDreamer
pip install -r requirements.txt
```

Install the source code for these third-party packages, with `cd ${FOLDER}; pip install -e .`

```
# Install third-party packages
third_party/
├── bevfusion -> based on db75150
├── diffusers -> based on v0.17.1 (afcca3916)
└── xformers -> minorly change 0.0.19 to install with pytorch1.10.2
```

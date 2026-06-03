## Installation 
my envir is cuda-12.4, the main function is `reconstruct_pc_GSA.py`. Others are some failed attempts.

```
conda create -n uist python=3.10 -y
conda activate uist

pip install torch torchvision torchaudio 
pip install trimesh opencv-python pyrealsense2
pip install IPython

## install pytorch3d (could be troublesome)
pip install iopath
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

## I found SAM solely is not very helpful
## install grounded-segment-anything is also hard

#pip install git+https://github.com/facebookresearch/segment-anything.git
#wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True

cd Grounded-Segment-Anything/
python -m pip install -e segment_anything

cd GroundingDINO
python -m pip install --no-build-isolation -e .

cd ../grounded-sam-osx 
bash install.sh

cd ..
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```

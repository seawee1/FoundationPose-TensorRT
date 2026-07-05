SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8
pip install trimesh imageio opencv-python open3d warp-lang==1.0.2 kornia==0.7.2 gdown

source "${SCRIPT_DIR}/deps.sh"
activate_deps  # We need correct nvcc for installed torch cuda version 
pip install git+https://github.com/facebookresearch/pytorch3d.git@d098beb7a7f92ee226de97b1b7905ee735aeed56 --no-build-isolation
pip install "$TENSORRT_HOME/python/tensorrt-10.9.0.34-cp310-none-linux_x86_64.whl"
pip install "$TENSORRT_HOME/python/tensorrt_lean-10.9.0.34-cp310-none-linux_x86_64.whl"
pip install "$TENSORRT_HOME/python/tensorrt_dispatch-10.9.0.34-cp310-none-linux_x86_64.whl"
deactivate_deps

pip install -e $SCRIPT_DIR/..

pip uninstall numpy -y
pip install "numpy<2" transformations Ninja

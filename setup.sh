conda create -n real2code2real python=3.10
conda activate real2code2real
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh xatlas pyvista pymeshfix igraph transformers
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu118.html

mkdir -p /tmp/extensions
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
pip install /tmp/extensions/nvdiffrast

        mkdir -p /tmp/extensions
        git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
        pip install /tmp/extensions/diffoctreerast

        mkdir -p /tmp/extensions
        git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
        pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/

        mkdir -p /tmp/extensions
        cp -r extensions/vox2seq /tmp/extensions/vox2seq
        pip install /tmp/extensions/vox2seq

pip install spconv-cu118pip 
pip install open3d
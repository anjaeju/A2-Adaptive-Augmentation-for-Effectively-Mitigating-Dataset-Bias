"""
2022-07-12
Part2: Training Generative Models (Source GAN and Target GAN)
This implementation is for understaning the overall structure of our methods.
"""

#################################
# 1. Training Source GAN
#################################

# Git clone stylegan2 that we used for our source GAN
git clone https://github.com/rosinality/stylegan2-pytorch.git

# 1. Making Source GAN Dataset
# LMDB_PATH: Path for output LMDB dataset that will be used for training Source GAN
# DATASET_PATH: Path for input dataset (e.g., CMNIST, CCIFAR10, BFFHQ, BAR) that will be used for training Source GAN
# n_worker: # of workers we will use
# size: (CMNIST & CCIFAR10: 32), (BFFHQ: 224), (BAR: 256)
python stylegan2-pytorch/prepare_data.py --out [LMDB_PATH] --n_worker [N_WORKER] --size [SIZE1] [DATASET_PATH]
# e.g., python stylegan2-pytorch/prepare_data.py --out=/sample/bffhq-training --n_worker=4 --size=224 /media/data/debiasing/BFFHQ

# 2. Training Source GAN
# N_GPU: # of GPUS
# PORT: Port for distributed learning
# BATCH_SIZE: # of batches for training Source GAN
# LMDB_PATH: Path for LMDB path that is made of training dataset
python -m torch.distributed.launch --nproc_per_node [N_GPU] --master_port [PORT] stylegan2-pytorch/train.py --batch [BATCH_SIZE] [LMDB_PATH]
# e.g., CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 stylegan2-pytorch/train.py /sample/bffhq-training

#################################
# 2. Training Target GAN
#################################

# Git clone few-show-gan-adaptation that we used for our target GAN
git clone https://github.com/utkarshojha/few-shot-gan-adaptation.git

# 1. Making Target GAN Dataset
# LMDB_PATH: Path for output LMDB dataset that will be used for training Target GAN
# BiasConflictPath: Path for the extracted bias-conflict dataset (e.g., CMNIST, CCIFAR10, BFFHQ, BAR) that will be used for training Target GAN
# n_worker: # of workers we will use
# size: (CMNIST & CCIFAR10: 32), (BFFHQ: 224), (BAR: 256)
python stylegan2-pytorch/prepare_data.py --out [LMDB_PATH] --n_worker [N_WORKER] --size [SIZE1] [BiasConflictPath]
# e.g., python stylegan2-pytorch/prepare_data.py --out=/home/sample/data --n_worker=4 --size=224 /sample/extracted-bias-conflict

# 2. Training Target GAN
# CHECKPOINTS: Pretrained weights for source GAN
# DATAPATH: Path for LMDB path that is made of extracted bias-conflict samples
# ExpName: Directory where trained network is stored 
python train.py --ckpt_source [CHECKPOINTS] --data_path [DATAPATH] --exp [ExpName]
# e.g., python train.py --ckpt_source ./checkpoints/source_bffhq.pt --data_path /sample/extracted-bias-conflict --exp target_bffhq


#################################
# 3. Adaptive Augmentation (A^2)
#################################

# 1. Projecting bias-align samples into latent spaces using pretrained source GAN
# Note that we slightly modified original projector file that is from StyleGAN2 repository
# CHECKPOINT: Pretrained source GAN
# GENERATOR_OUTPUT_SIZE: Output image sizes of the generator
python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] 
# e.g., python projector.py --ckpt=./checkpoints/source_bffhq.pt --size=224 --file_path=/media/data/debiasing/BFFHQ --save_path=/sample/latents/bffhq-latents.pt --step=1000

# 2. Translating bias-align samples into bias-conflict samples using Target GAN
# CHECKPOINT1: Pretrained source GAN
# CHECKPOINT2: Pretrained target GAN
# PROJECTED: projected latent vectors
python generate.py --ckpt_source [CHECKPOINT1] --ckpt_target [CHECKPOINT2] --load_learned [PROJECTED]
# e.g., python generate.py --ckpt_source=./checkpoints/source_bffhq.pt --ckpt_target=./checkpoints/target_bffhq.pt --size=224 --load_learned=/sample/latents/bffhq-latents.pt --train=/sample/bffhq-training
# HybridMIM
HybridMIM: A Hybrid Masked Image Modeling Framework for 3D Medical Image Segmentation

# Pre-training

We support two architecture: UNet and SwinUNETR.

We collect four datasets: Luna16, FLARE2021, Covid-19 and ATM22. You can search them at https://grand-challenge.org/challenges/all-challenges/.

## UNet pre-training
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main.py --batch_size=1 --num_steps=100000 --lrdecay --lr=1e-4 --decay=0.001 --logdir=./deepunet --model_name=deepunet_v2 --eval_num=500
```

## SwinUNETR pre-training

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main.py --batch_size=1 --num_steps=100000 --lrdecay --lr=6e-6 --decay=0.1 --logdir=./swin_pretrain --smartcache_dataset --model_name=swin --eval_num=500 --val_cache
```
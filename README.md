# HybridMIM
HybridMIM: A Hybrid Masked Image Modeling Framework for 3D Medical Image Segmentation

![](/imgs/framework.png)
# Pre-training

We design a self-supervised learning method to learn the spatial information from the high dimensional medical images at multiple levels, including: pixel-level, region-level and sample-level.

Compared with other self-supervised learning methods which evaluated on the single architecture, We support two architectures: UNet and SwinUNETR.

We collect four datasets: Luna16, FLARE2021, Covid-19 and ATM22 for a general pre-training. You can search them at https://grand-challenge.org/challenges/all-challenges/.

We evaluate HybridMIM using four segmentation datasets: BraTS2020, BTCV, MSD Liver. You can find them at:
1. https://www.med.upenn.edu/cbica/brats2020/data.html
2. https://www.synapse.org/#!Synapse:syn3193805
3. http://medicaldecathlon.com/

## UNet pre-training
```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main.py --batch_size=1 --num_steps=100000 --lrdecay --lr=1e-4 --decay=0.001 --logdir=./deepunet --model_name=deepunet_v2 --eval_num=500
```

## SwinUNETR pre-training

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=11223 main.py --batch_size=1 --num_steps=100000 --lrdecay --lr=6e-6 --decay=0.1 --logdir=./swin_pretrain --smartcache_dataset --model_name=swin --eval_num=500 --val_cache
```
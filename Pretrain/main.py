# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader

from losses.loss import * 

mse_loss = torch.nn.MSELoss()
def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, best_val=1e8):
        
        model.train()

        for step, batch in enumerate(train_loader):
            t1 = time()
            model.train()

            image = batch["image"].cuda()
            
            model_out = model(image)        
            x_rec = torch.sigmoid(model_out["logits"])
            labels = model_out['images']
            mask_images = model_out["x_mask"]
            pred_mask_region = model_out["pred_mask_region"]
            mask_labels = model_out["mask_labels"]
            contrast_pred_1 = model_out["contrast_pred_1"]
            contrast_pred_2 = model_out["contrast_pred_2"]
            pred_mask_region_position = model_out["pred_mask_region_position"]
            mask_region_position_label = model_out["mask_position_lables"]
            # loss_rec = forward_loss_reconstruct_mask(x_rec, labels, mask_images, mask_value=0.0)
            # loss_rec = forward_loss_reconstruct(x_rec, labels)
            loss_rec = mse_loss(x_rec, labels)
            loss_mask_region = forward_loss_mask(pred_mask_region, mask_labels)
            position_pred = (torch.sigmoid(pred_mask_region_position) > 0.5).float()
            position_pred_num_region = position_pred.sum(dim=-1)
          
            loss_consistency = (forward_loss_mask(pred_mask_region, position_pred_num_region.detach()) + nn.MSELoss()(position_pred_num_region, pred_mask_region.argmax(dim=-1).float().detach())) / 2

            loss_contrast = forward_constrast_loss(contrast_pred_1, contrast_pred_2)
            loss_position = forward_loss_mask_position(pred_mask_region_position, mask_region_position_label)

            if args.distributed:
                if args.rank == 0:
                    print(f"Step:{global_step}/{args.num_steps}, loss_rec is {loss_rec}, loss_mask_num is {loss_mask_region}, loss_mask_position is {loss_position}, loss_consistency is {loss_consistency}")
            else :
                print(f"loss_rec is {loss_rec}")
            
            loss = loss_rec + 0.1 * loss_mask_region + 0.1 * loss_position + 0.001 * loss_consistency + 0.001 * loss_contrast
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))
            else:
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))

            global_step += 1

            
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_losses, img_list = validation(args, test_loader)
                writer.add_scalar("Validation/loss_total", scalar_value=val_losses[0], global_step=global_step)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_losses[1], global_step=global_step)
                writer.add_scalar("Validation/loss_num", scalar_value=val_losses[2], global_step=global_step)
                writer.add_scalar("Validation/loss_position", scalar_value=val_losses[3], global_step=global_step)
                writer.add_scalar("Validation/loss_cl", scalar_value=val_losses[4], global_step=global_step)
                writer.add_scalar("train/loss_total", scalar_value=np.mean(loss), global_step=global_step)
                writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_rec), global_step=global_step)

                writer.add_image("Validation/x1_row", img_list[0], global_step, dataformats="HW")
                writer.add_image("Validation/x1_gt", img_list[1], global_step, dataformats="HW")
                writer.add_image("Validation/x1_recon", img_list[2], global_step, dataformats="HW")
                writer.add_image("Validation/x1_mask", img_list[3], global_step, dataformats="HW")
                writer.add_image("Validation/x1_recon_raw", img_list[4], global_step, dataformats="HW")

                val_loss_recon = val_losses[1]
                if val_loss_recon < best_val:
                    best_val = val_loss_recon
                    checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    # save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                    torch.save(model.state_dict(), logdir + "/model_bestValRMSE.pt")
                    print(
                        "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                            best_val, val_loss_recon
                        )
                    )
                else:
                    print(
                        "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                            best_val, val_loss_recon
                        )
                    )
        return global_step, best_val

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []
        loss_num_total = []
        loss_position_total = []
        loss_cl_total = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch["image"].cuda()
                model_out = model(val_inputs)

                x_rec = torch.sigmoid(model_out["logits"])
                labels = model_out['images']
                mask_images = model_out["x_mask"]
                pred_mask_region = model_out["pred_mask_region"]
                mask_labels = model_out["mask_labels"]
                contrast_pred_1 = model_out["contrast_pred_1"]
                contrast_pred_2 = model_out["contrast_pred_2"]
                pred_mask_region_position = model_out["pred_mask_region_position"]
                mask_region_position_label = model_out["mask_position_lables"]
                loss_rec = forward_loss_reconstruct_mask(x_rec, labels, mask_images, mask_value=0.0)
                loss_rec = forward_loss_reconstruct(x_rec, labels)
                loss_rec = mse_loss(x_rec, labels)

                num_pred = pred_mask_region.argmax(dim=-1)

                num_acc = (num_pred == mask_labels).float().sum() / (num_pred.shape[0] * num_pred.shape[1])
                

                loss_mask_region = forward_loss_mask(pred_mask_region, mask_labels)
                position_pred = (torch.sigmoid(pred_mask_region_position) > 0.5).float()
                position_pred_num_region = position_pred.sum(dim=-1)
                position_acc = (position_pred == mask_region_position_label).float().sum() / (position_pred.shape[0] * position_pred.shape[1] * position_pred.shape[2])

                loss_consistency = (forward_loss_mask(pred_mask_region, position_pred_num_region.detach()) + nn.MSELoss()(position_pred_num_region, pred_mask_region.argmax(dim=-1).float().detach())) / 2

                loss_contrast = forward_constrast_loss(contrast_pred_1, contrast_pred_2)
                loss_position = forward_loss_mask_position(pred_mask_region_position, mask_region_position_label)

                loss = loss_rec + 0.1 * loss_mask_region + 0.1 * loss_position + 0.001 * loss_consistency + 0.001 * loss_contrast
                loss_val.append(loss.item())
                loss_val_recon.append(loss_rec.item())
                loss_num_total.append(num_acc.item())
                loss_position_total.append(position_acc.item())
                loss_cl_total.append(loss_contrast.item())

                x_gt = labels.detach().cpu().numpy()
                x_gt = (x_gt - np.min(x_gt)) / (np.max(x_gt) - np.min(x_gt))
                xgt = x_gt[0][0][:, :, 48] * 255.0
                xgt = xgt.astype(np.uint8)

                x_mask = mask_images.detach().cpu().numpy()
                x_mask = (x_mask - np.min(x_mask)) / (np.max(x_mask) - np.min(x_mask))
                x_mask = x_mask[0][0][:, :, 48] * 255.0
                x_mask = x_mask.astype(np.uint8)

                x_row = val_inputs.detach().cpu().numpy()
                x_row = (x_row - np.min(x_row)) / (np.max(x_row) - np.min(x_row))
                x_row = x_row[0][0][:, :, 48] * 255.0
                x_row = x_row.astype(np.uint8)

                mask = (mask_images != 0.0).float()
                rec_x1 = mask_images.detach().cpu().numpy() + (1 - mask) * x_rec.detach().cpu().numpy()
                rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
                recon = rec_x1[0][0][:, :, 48] * 255.0
                recon = recon.astype(np.uint8)

                rec_row = x_rec.detach().cpu().numpy()
                rec_row = (rec_row - np.min(rec_row)) / (np.max(rec_row) - np.min(rec_row))
                rec_row = rec_row[0][0][:, :, 48] * 255.0
                rec_row = rec_row.astype(np.uint8)

                img_list = [x_row, xgt, recon, x_mask, rec_row]
                
                print("Validation step:{}, Loss:{:.4f}, Loss Reconstruction:{:.4f}".format(step, loss, loss_rec))

        return [np.mean(loss_val), np.mean(loss_val_recon), np.mean(loss_num_total), np.mean(loss_position_total), np.mean(loss_cl_total)], img_list

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--model_name", default="deepunet_v2", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=200, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--val_cache", action="store_true", help="use monai cache Dataset")

    args = parser.parse_args()
    logdir = "./runs/" + args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None

    if args.model_name == "deepunet_v2":
        from pretrain_models.deep_unet_v2 import DeepUNet
        model = DeepUNet(1, 1, features=[64, 64, 128, 256, 512], 
                        pool_size=((2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)), 
                        select_reconstruct_region=[[0, 0, 0], [12, 12, 12]],
                        dropout=0.2,
                        pretrain=True)

    elif args.model_name == "deppunet":
        from pretrain_models.deep_unet import DeepUNet
        model = DeepUNet(args.in_channels, args.out_channels, 
                features=[32, 32, 64, 128, 256, 512], 
                pool_size=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
                pretrain=False,
                select_reconstruct_region=[[0, 0, 0], [3, 3, 3]])
        if args.load_pretrain:
            checkpoints = torch.load(args.pretrain_model_path, map_location="cpu")
            new_checkpoints = {}
            
            for k, v in checkpoints.items():
                if "decoder_pred" not in k and "ups" not in k:
                # if "decoder_pred" not in k:
                    new_checkpoints[k] = v

            model.load_state_dict(new_checkpoints, strict=False)
            print("load params successed.")
    
    elif args.model_name == "swinunetr_8":
        from pretrain_models.swinunetr_8 import SwinUNETR
        model = SwinUNETR((96, 96, 96),
                      in_channels=1,
                      out_channels=1,
                      drop_rate=0.1,
                      feature_size=args.feature_size,
                      pretrain=True,
                      select_reconstruct_region=(0, 3))
    
    else :
        from pretrain_models.swinunetr import SwinUNETR
        model = SwinUNETR((96, 96, 96),
                      in_channels=1,
                      out_channels=1,
                      drop_rate=0.1,
                      feature_size=args.feature_size,
                      pretrain=True,
                      select_reconstruct_region=(0, 3))

    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        model.load_state_dict(model_dict["state_dict"])
        model.epoch = model_dict["epoch"]
        model.optimizer = model_dict["optimizer"]

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    train_loader, test_loader = get_loader(args)

    global_step = 0
    best_val = 1e8
    while global_step < args.num_steps:
        global_step, best_val = train(args, global_step, train_loader, best_val=best_val)

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "final_model.pth")


if __name__ == "__main__":
    main()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.datasets import Dynamic_Scenes_Dataset

from utils.utils import *
from skimage.metrics import structural_similarity as ssim

from moco.encoder import *
from moco.moco import *
from model import RFGViT

data_root = '/data/keuntek/HDR_KALANTARI'
batch_size = 1

val_dataset = Dynamic_Scenes_Dataset(root_dir=data_root, is_training=False, transform=None, crop=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = RFGViT().cuda()

moco = MoCo(base_encoder=Encoder,T=0.5).cuda()
moco.load_state_dict(torch.load("./checkpoints/encoder.pth"))
moco_encoder = moco.encoder_q.cuda()
moco_encoder.eval()

model.cuda()
#ee.cuda()
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

requires_grad(model, False)
requires_grad(moco_encoder, False)
total_psnr=0.
total_psnr_l=0.
total_ssim_l=0.
total_ssim_m=0.

model.load_state_dict(torch.load("./checkpoints/RFGVIT.pth"))
total_psnr=0.
total_psnr_l=0.
total_ssim_l=0.
total_ssim_m=0.
#print(idx)
for idx,i in enumerate(val_loader):
    #print(i['input0'].shape)
    ldr1 = i['input0'].cuda()
    ldr2 = i['input1'].cuda()
    ldr3 = i['input2'].cuda()
    ldr1_tm = i['input0_tm'].cuda()
    ldr2_tm = i['input1_tm'].cuda()
    ldr3_tm = i['input2_tm'].cuda()
    gt = i['label'].cuda()

    batch_size,ch,H,W = ldr1.shape

    ldr1_cat = torch.cat([ldr1,ldr1_tm], dim=1)
    ldr2_cat = torch.cat([ldr2,ldr2_tm], dim=1)
    ldr3_cat = torch.cat([ldr3,ldr3_tm], dim=1)
    #-1~1 for encoder
    local_1, global_1, _ = moco_encoder(ldr1*2-1)
    local_2, global_2, _ = moco_encoder(ldr2*2-1)
    local_3, global_3, _ = moco_encoder(ldr3*2-1)
    pred = model(ldr1_cat,ldr2_cat,ldr3_cat,[local_1,local_2,local_3],[global_1,global_2,global_3])
    pred = pred[0,:,:,:]#.astype(np.float32)
    # pred = (pred+1.)/2.
    pred_img = pred.permute(1,2,0).cpu().numpy()
    gt_img = gt[0].permute(1,2,0).cpu().numpy()

    mse_l = torch.mean((gt-pred)**2)
    psnr_l = -10.*math.log10(mse_l)
    total_psnr_l+=psnr_l
    ssim_l = ssim(pred.squeeze().permute(1,2,0).cpu().numpy(), gt.squeeze().permute(1,2,0).cpu().numpy(), data_range=1, multichannel=True)
    total_ssim_l+=ssim_l
    pred = range_compressor_tensor(pred)
    pred_img = pred.permute(1,2,0).cpu().numpy()

    gt = range_compressor_tensor(gt)

    mse = torch.mean((gt-pred)**2)
    psnr = -10.*math.log10(mse)
    total_psnr+=psnr
    ssim_m = ssim(pred.squeeze().permute(1,2,0).cpu().numpy(), gt.squeeze().permute(1,2,0).cpu().numpy(), data_range=1, multichannel=True)
    total_ssim_m+=ssim_m
            #print(psnr)
    avg_psnr_mu = total_psnr/len(val_loader)
    avg_psnr_l = total_psnr_l/len(val_loader)
    avg_ssim_mu = total_ssim_m/len(val_loader)
    avg_ssim_l = total_ssim_l/len(val_loader)
    print(len(val_loader))
print("Test PSNR M: ",avg_psnr_mu)
print("Test PSNR L: ",avg_psnr_l)
print("Test SSIM M: ",avg_ssim_mu)
print("Test SSIM L: ",avg_ssim_l)

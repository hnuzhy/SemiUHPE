# ------------------------------------------------------------------------------
# Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
# ------------------------------------------------------------------------------

import torch

import numpy as np
from PIL import Image
    
# arxiv2017.08 (Cutout) Improved Regularization of Convolutional Neural Networks with Cutout
# https://github.com/uoguelph-mlrg/Cutout
def random_cutout_tensor(image, mask_holes_num=3, normal=False):
    N, _, width, height = image.shape
    
    if not normal:  # even distribution sampling
        center_x = torch.randint(20, width-20, (N, mask_holes_num)).int().cuda()
        center_y = torch.randint(20, height-20, (N, mask_holes_num)).int().cuda()
    else:  # normal distribution sampling
        img_cx, sigma_x = width / 2.0, width / 6.0  # 224 / 6.0 = 37, 224 / 8.0 = 28
        img_cy, sigma_y = height / 2.0, height / 6.0  # 224 / 6.0 = 37, 224 / 8.0 = 28
        center_x = torch.normal(img_cx, sigma_x, size=(N, mask_holes_num)).int().cuda()
        center_y = torch.normal(img_cy, sigma_y, size=(N, mask_holes_num)).int().cuda()
        
    size = torch.randint(10, 20, (N, mask_holes_num, 2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0], 0, width)
    y0 = torch.clamp_(center_y-size[...,1], 0, height)

    x1 = torch.clamp_(center_x+size[...,0], 0, width)
    y1 = torch.clamp_(center_y+size[...,1], 0, height)

    for i in range(N):
        for j in range(mask_holes_num):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = 0
    return image


def random_cutout_pil(image, mask_holes_num=3, normal=False):
    width, height = image.size
    
    if not normal:  # even distribution sampling
        center_x = np.random.randint(20, width-20, (mask_holes_num,))
        center_y = np.random.randint(20, height-20, (mask_holes_num,))
    else:  # normal distribution sampling
        img_cx, sigma_x = width / 2.0, width / 6.0
        img_cy, sigma_y = height / 2.0, height / 6.0
        center_x = np.int32(np.random.normal(img_cx, sigma_x, (mask_holes_num,)))
        center_y = np.int32(np.random.normal(img_cy, sigma_y, (mask_holes_num,)))
    
    size = np.random.randint(10, 20, (mask_holes_num, 2))
    
    x0 = np.clamp_(center_x-size[...,0], 0, width)
    y0 = np.clamp_(center_y-size[...,1], 0, height)

    x1 = np.clamp_(center_x+size[...,0], 0, width)
    y1 = np.clamp_(center_y+size[...,1], 0, height)

    for j in range(mask_holes_num):
        image_mask = Image.new("RGB", ( x1[j]-x0[j], y1[j]-y0[j]), (127, 127, 127))
        image.paste(image_mask, (x0[j], y0[j]))
    return image



# ICCV2019 CutMix - Regularization Strategy to Train Strong Classifiers With Localizable Features
# https://github.com/clovaai/CutMix-PyTorch
def random_cutmix_tensor(image, mask_holes_num=3, normal=False):
    N, _, width, height = image.shape
    
    if not normal:  # even distribution sampling
        center_x = torch.randint(20, width-20, (N, mask_holes_num)).int().cuda()
        center_y = torch.randint(20, height-20, (N, mask_holes_num)).int().cuda()
        
    else:  # normal distribution sampling
        img_cx, sigma_x = width / 2.0, width / 6.0  # 224 / 6.0 = 37, 224 / 8.0 = 28
        img_cy, sigma_y = height / 2.0, height / 6.0  # 224 / 6.0 = 37, 224 / 8.0 = 28
        center_x = torch.normal(img_cx, sigma_x, size=(N, mask_holes_num)).int().cuda()
        center_y = torch.normal(img_cy, sigma_y, size=(N, mask_holes_num)).int().cuda()
    
    size = torch.randint(10, 20, (N, mask_holes_num, 2)).int().cuda()
    
    x0 = torch.clamp_(center_x-size[...,0], 0, width)
    y0 = torch.clamp_(center_y-size[...,1], 0, height)

    x1 = torch.clamp_(center_x+size[...,0], 0, width)
    y1 = torch.clamp_(center_y+size[...,1], 0, height)
    
    rand_index = torch.randperm(N).cuda()
    image_rand = image[rand_index]

    for i in range(N):
        for j in range(mask_holes_num):
            image[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]] = image_rand[i, :, y0[i,j]:y1[i,j], x0[i,j]:x1[i,j]]
    return image


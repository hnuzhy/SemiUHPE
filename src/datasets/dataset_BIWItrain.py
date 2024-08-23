'''
Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
'''

import os
import random
import cv2
import torch
import scipy.io as sio
import numpy as np
import torchvision.transforms as tfs

from tqdm import tqdm
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

from src.utils import get_6DRepNet_Rot


def process_annotated_image(im, yaw, pitch, roll, augment_strong=False, config=None):
    
    # Inputs are in degrees, convert to rad.
    yaw = yaw * np.pi / 180.0  # similar to azimuth / az
    pitch = pitch * np.pi / 180.0  # similar to theta / th
    roll = roll * np.pi / 180.0  # similat elevation / el


    if np.random.uniform(0, 1) < 0.5:  # Flip
        yaw = -yaw
        roll = -roll
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    if np.random.uniform(0, 1) < 0.05:  # Blur
        im = im.filter(ImageFilter.BLUR)
        
    # im = tfs.ColorJitter(brightness=0.1, contrast=0.5, hue=0.2, saturation=0.5)(im)  # weak
    # im = tfs.ColorJitter(brightness=0.2, contrast=0.6, hue=0.3, saturation=0.4)(im)  # strong
    
    pad, scale_min, scale_max = 0, 0.8, 1.25
    aug_weak = tfs.Compose([
        tfs.Pad((pad, pad), padding_mode='edge'),
        tfs.RandomResizedCrop(size=(224, 224), scale=(scale_min, scale_max), ratio=(1., 1.))
    ])
    im_weak = aug_weak(im.copy()) 
    
    rot_angle = 0
    if augment_strong:
        if config.rotate_aug:
            rot_angle = round(np.random.rand()*60 - 30, 3)  # (0, 1)*60 - 30 --> (-30, 30)
            im = im.rotate(rot_angle, center=(112, 112), expand=True)
            im_rot_w, im_rot_h = im.size  # must big than (224,224)
            new_x_min, new_y_min = im_rot_w//2 - 112, im_rot_h//2 - 112
            new_x_max, new_y_max = new_x_min + 224, new_y_min + 224
            im = im.crop((new_x_min, new_y_min, new_x_max, new_y_max))
        pad, scale_min, scale_max = 0, 0.6, 1.5
        aug_strong = tfs.Compose([
            tfs.Pad((pad, pad), padding_mode='edge'),
            tfs.RandomResizedCrop(size=(224, 224), scale=(scale_min, scale_max), ratio=(1., 1.)),
        ])
        im_strong = aug_strong(im.copy())
    else:
        im_strong = None
        
    im_weak_tensor = tfs.ToTensor()(im_weak)
    im_weak_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_weak_tensor)
    if im_strong is not None:
        im_strong_tensor = tfs.ToTensor()(im_strong)
        im_strong_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_strong_tensor)
    else:
        im_strong_tensor = None

    trans_euler = [pitch * 180.0 / np.pi, yaw * 180.0 / np.pi, roll * 180.0 / np.pi]
    # trans_euler = [(v + 90.0) / 180.0 for v in trans_euler]  # norm to (0,1)
    trans_euler = torch.FloatTensor(trans_euler)
    
    Rot = get_6DRepNet_Rot(pitch, yaw, roll)  # in radians
    Rot = torch.FloatTensor(Rot)
 
    return im_weak, im_weak_tensor, im_strong, im_strong_tensor, trans_euler, Rot, rot_angle
    

class Dataset_BIWItrain(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None, img_size=224):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        self.img_size = img_size
        print('Load BIWItrain Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size
        
        img_cont = self.img_files[idx]
        cont_labels = self.anno_files[idx]
        yaw, pitch, roll = cont_labels  # degrees
        
        while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
        while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
        while abs(roll) > 180: roll = roll - roll/abs(roll)*360
        
        pitch = np.clip(pitch, -89.99, 89.99)
        yaw = np.clip(yaw, -89.99, 89.99)
        roll = np.clip(roll, -89.99, 89.99)
        
        # Convert Opencv image to PIL image
        color_coverted = cv2.cvtColor(img_cont, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_coverted)
        pil_image = pil_image.resize([self.img_size, self.img_size])

        im_weak, im_weak_tensor, im_strong, im_strong_tensor, trans_euler, Rot, rot_angle = process_annotated_image(
            pil_image, yaw, pitch, roll, augment_strong=self.augment_strong, config=self.config)

        ra = rot_angle * np.pi / 180.0
        aug_rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
        # rotation_matrix_new = aug_rot_mat @ rotation_matrix  # for the config.rotate_aug
        aug_rot_mat = torch.FloatTensor(aug_rot_mat)
    
        Rot = aug_rot_mat @ Rot  # for the config.rotate_aug
        
        assert im_strong is not None, "im_strong should not be None for the BIWItrain Dataset!!!"
        # if im_strong is None:
            # im_strong_tensor = torch.zeros_like(im_weak_tensor)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "BIWItrain_weak_" in name]
        save_jpg_name = self.config.log_dir+"/BIWItrain_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "BIWItrain_strong_" in name]
        save_jpg_name = self.config.log_dir+"/BIWItrain_strong_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)

        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        # sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor)  # BIWItrain dose no need labels
        sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor, aug_rot_mat=aug_rot_mat)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_BIWItrain(phase, config):
    
    assert phase in ["ulb_train"], "unsupport phase of BIWItrain! " + phase
    
    db_path = config.data_dir_BIWItrain  # path of BIWI_train.npz which is created by FSA-Net
    db_dict = np.load(db_path)
    img_size = db_dict['img_size']  # image size is 64*64
    
    imgs_list, pose_list = [], []
    for img, cont_labels in tqdm(zip(db_dict['image'], db_dict['pose'])):
        [yaw, pitch, roll] = cont_labels  # in degrees
        imgs_list.append(img)
        pose_list.append([yaw, pitch, roll])

    batch_size = round(config.batch_size * config.ulb_batch_ratio)
    shuffle = True
    augment_strong = True

    dset = Dataset_BIWItrain(imgs_list, pose_list, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, 
        pin_memory=True, drop_last=True)
    return dloader


if __name__ == '__main__':
    print("testing BIWItrain Dataset...")


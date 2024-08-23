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
from PIL import Image
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

from src.utils import get_6DRepNet_Rot


def process_annotated_image(im, yaw, pitch, roll):
    # Inputs are in degrees, convert to rad.
    yaw = yaw * np.pi / 180.0  # similar to azimuth / az
    pitch = pitch * np.pi / 180.0  # similar to theta / th
    roll = roll * np.pi / 180.0  # similat elevation / el

    Rot = get_6DRepNet_Rot(pitch, yaw, roll)  # in radians
    Rot = torch.FloatTensor(Rot)

    euler = [pitch * 180.0 / np.pi, yaw * 180.0 / np.pi, roll * 180.0 / np.pi]
    # euler = [(v + 90.0) / 180.0 for v in euler]  # norm to (0,1)
    euler = torch.FloatTensor(euler)
    
    im_tensor = tfs.ToTensor()(im)
    im_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_tensor)

    return im, im_tensor, euler, Rot
    

class Dataset_BIWItest(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None, img_size=224):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        self.img_size = img_size
        print('Load BIWItest Dataset with phase %s, Length:%d'%(phase, self.size))

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

        img, im_tensor, trans_euler, Rot = process_annotated_image(pil_image, yaw, pitch, roll)
        
        im_strong_tensor = torch.zeros_like(im_tensor)
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "BIWItest_" in name]
        save_jpg_name = self.config.log_dir+"/BIWItest_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: img.save(save_jpg_name)

        sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_tensor, img_strong=im_strong_tensor)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_BIWItest(phase, config):
    
    assert phase in ["test"], "unsupport phase of BIWItest! " + phase
    
    db_path = config.data_dir_BIWItest  # path of BIWI_test.npz which is created by FSA-Net
    db_dict = np.load(db_path)
    img_size = db_dict['img_size']  # image size is 64*64
    
    imgs_list, pose_list = [], []
    for img, cont_labels in tqdm(zip(db_dict['image'], db_dict['pose'])):
        [yaw, pitch, roll] = cont_labels  # in degrees
        imgs_list.append(img)
        pose_list.append([yaw, pitch, roll])

    batch_size = config.batch_size
    shuffle = False
    augment_strong = False

    dset = Dataset_BIWItest(imgs_list, pose_list, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, 
        pin_memory=True, drop_last=False)
    return dloader


if __name__ == '__main__':
    print("testing BIWItest Dataset...")


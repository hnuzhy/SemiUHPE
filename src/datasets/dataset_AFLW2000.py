'''
Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
'''

import os
import cv2
import random
import torch
import scipy.io as sio
import numpy as np
import torchvision.transforms as tfs

from PIL import Image
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

from src.utils import get_6DRepNet_Rot

def process_ori_mat_anno(img_path, anno_path):
    img_ori = Image.open(img_path).convert('RGB')
    img_w, img_h = img_ori.size
    
    mat_contents = sio.loadmat(anno_path)
    pose_para = mat_contents['Pose_Para'][0]
    pitch = pose_para[0] * 180 / np.pi
    yaw = pose_para[1] * 180 / np.pi
    roll = pose_para[2] * 180 / np.pi
    
    while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
    while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
    while abs(roll) > 180: roll = roll - roll/abs(roll)*360
    
    cont_labels = np.array([pitch, yaw, roll])
    
    pt2d = mat_contents['pt2d'] 
    
    pt2d_x, pt2d_y = pt2d[0,:], pt2d[1,:]
    pt2d_idx, pt2d_idy = pt2d_x>0.0, pt2d_y>0.0  # remove negative value.
    pt2d_id = pt2d_idx
    if sum(pt2d_idx) > sum(pt2d_idy):
        pt2d_id = pt2d_idy
    pt2d_x = pt2d_x[pt2d_id]
    pt2d_y = pt2d_y[pt2d_id]
    
    # Crop the face loosely
    x_min = int(min(pt2d_x))
    x_max = int(max(pt2d_x))
    y_min = int(min(pt2d_y))
    y_max = int(max(pt2d_y))

    h = y_max-y_min
    w = x_max-x_min
    h = w = max(h, w)  # hint 1: we want get a squared face bbox <<<------
    # h = w = (h + w)/2.0  # hint 1: we want get a squared face bbox <<<------
    
    # padding head or face (Following the FSA-Net which sets ad = 0.6)
    ad = 0.2
    new_x_min = max(int(x_min - ad * w), 0)
    new_x_max = min(int(x_max + ad * w), img_w - 1)
    new_y_min = max(int(y_min - ad * h * 2), 0)  # hint 2: give more area above the top face <<<------
    new_y_max = min(int(y_max + ad * h), img_h - 1)
    
    left, top, right, bottom = new_x_min, new_y_min, new_x_max, new_y_max
    # bbox = [left, top, right, bottom]  # drawback 1: get an affine transformed face image <<<------
    # return [img_ori, cont_labels, bbox]

    # hint 3: we do not want to change the face shape very much <<<------
    temph, tempw = bottom - top, right - left
    if temph > tempw:
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, int((temph-tempw)/2), int((temph-tempw)/2)
    else:
        pad_top, pad_bottom, pad_left, pad_right = int((tempw-temph)/2), int((tempw-temph)/2), 0, 0   
    
    if left-pad_left < 0: new_x_min, new_left = 0, abs(left-pad_left)
    else: new_x_min, new_left = left-pad_left, 0
    
    if top-pad_top < 0: new_y_min, new_top = 0, abs(top-pad_top)
    else: new_y_min, new_top = top-pad_top, 0
    
    if right+pad_right > img_w-1: new_x_max, new_right = img_w-1, right+pad_right-img_w+1
    else: new_x_max, new_right = right+pad_right, 0
    
    if bottom+pad_bottom > img_h-1: new_y_max, new_bottom = img_h-1, bottom+pad_bottom-img_h+1
    else: new_y_max, new_bottom = bottom+pad_bottom, 0

    img_crop = img_ori.crop((new_x_min, new_y_min, new_x_max, new_y_max))
    width, height = img_crop.size
    new_width = width + new_right + new_left
    new_height = height + new_top + new_bottom
    img_padded = Image.new(img_crop.mode, (new_width, new_height), (0, 0, 0))
    img_padded.paste(img_crop, (new_left, new_top))  
    img_padded = img_padded.resize([224, 224])
    
    return img_padded, cont_labels
    
    # img_cv2 = np.array(img_ori)  # PIL to cv2
    # img_cv2 = img_cv2[:, :, ::-1]  # Convert RGB to BGR 
    # img_crop = img_cv2[new_y_min:new_y_max, new_x_min:new_x_max]
    # img_crop = cv2.copyMakeBorder(img_crop, new_top, new_bottom, new_left, new_right, cv2.BORDER_CONSTANT, value=(0,0,0))
    # img_crop = cv2.resize(img_crop, (224, 224))
    # img_pil = img_crop[:, :, ::-1]  # Convert BGR to RGB 
    # img_pil = Image.fromarray(img_pil)

    # return img_pil, cont_labels


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
    

class Dataset_AFLW2000(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load AFLW2000 Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size
        
        img_path = self.img_files[idx]
        anno_path = self.anno_files[idx]

        img, cont_labels = process_ori_mat_anno(img_path, anno_path)
        pitch, yaw, roll = cont_labels  # degrees, X-Y-Z
        
        img, im_tensor, trans_euler, Rot = process_annotated_image(img, yaw, pitch, roll)
        
        im_strong_tensor = torch.zeros_like(im_tensor)
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "AFLW2000_" in name]
        save_jpg_name = self.config.log_dir+"/AFLW2000_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: img.save(save_jpg_name)

        sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_tensor, img_strong=im_strong_tensor)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_AFLW2000(phase, config, debug=False):
    
    assert phase in ["test"], "unsupport phase of AFLW2000! " + phase
    
    mypath = config.data_dir_AFLW2000
    files_mat = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.mat')]
    files_jpg = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.jpg')]
    files_mat.sort()  # <--- important
    files_jpg.sort()  # <--- important
    
    files_mat_left, files_jpg_left = [], []
    for file_mat, file_jpg in zip(files_mat, files_jpg):

        mat_contents = sio.loadmat(file_mat)
        pose_para = mat_contents['Pose_Para'][0]
        pitch = pose_para[0] * 180 / np.pi
        yaw = pose_para[1] * 180 / np.pi
        roll = pose_para[2] * 180 / np.pi
        while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
        while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
        while abs(roll) > 180: roll = roll - roll/abs(roll)*360
        
        if abs(pitch) < 90 and abs(yaw) < 90 and abs(roll) < 90:
            files_mat_left.append(file_mat)
            files_jpg_left.append(file_jpg)
        else:
            if debug: print("Remove this illegal face instance:", file_jpg, pitch, yaw, roll) # remove 36 faces
            
    print("Left faces number in AFLW2000:", len(files_jpg_left))

    batch_size = config.batch_size
    shuffle = False
    augment_strong = False

    dset = Dataset_AFLW2000(files_jpg_left, files_mat_left, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, 
        pin_memory=True, drop_last=False)
    return dloader


if __name__ == '__main__':
    print("testing AFLW2000 Dataset...")


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

from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

from src.utils import get_6DRepNet_Rot

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # for fixing bug "OSError: image file is truncated."


def process_ori_mat_anno(img_path, anno_dict):
    img_ori = Image.open(img_path).convert('RGB')
    img_w, img_h = img_ori.size
    
    gt_bbox = anno_dict["bbox"]  # format xyxy, not used here
    pt2d = anno_dict["landmarks"]  # shape is (2, 19)
    
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
    ad = np.random.random_sample() * 0.2 + 0.2
    new_x_min = max(int(x_min - ad * w), 0)
    new_x_max = min(int(x_max + ad * w), img_w - 1)
    new_y_min = max(int(y_min - ad * h * 2), 0)  # hint 2: give more area above the top face <<<------
    new_y_max = min(int(y_max + ad * h), img_h - 1)
    
    left, top, right, bottom = new_x_min, new_y_min, new_x_max, new_y_max
    # bbox = [left, top, right, bottom]  # drawback 1: get an affine transformed face image <<<------
    # return [img_ori, bbox]

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
    
    return img_padded
    
    # img_cv2 = np.array(img_ori)  # PIL to cv2
    # img_cv2 = img_cv2[:, :, ::-1]  # Convert RGB to BGR 
    # img_crop = img_cv2[new_y_min:new_y_max, new_x_min:new_x_max]
    # img_crop = cv2.copyMakeBorder(img_crop, new_top, new_bottom, new_left, new_right, cv2.BORDER_CONSTANT, value=(0,0,0))
    # img_crop = cv2.resize(img_crop, (224, 224))
    # img_pil = img_crop[:, :, ::-1]  # Convert BGR to RGB 
    # img_pil = Image.fromarray(img_pil)

    # return img_pil 


def process_not_annotated_image(im, augment_strong=False, config=None):

    if np.random.uniform(0, 1) < 0.5:  # Flip
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
    
    # trans_euler = torch.FloatTensor([0, 0, 0])
    # Rot = torch.FloatTensor([[0,0,0], [0,0,0], [0,0,0]])

    # return im_weak, im_weak_tensor, im_strong, im_strong_tensor, trans_euler, Rot
    return im_weak, im_weak_tensor, im_strong, im_strong_tensor, rot_angle


class Dataset_AFLWFace(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load AFLWFace Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size

        img_path = self.img_files[idx]
        anno_dict = self.anno_files[idx]

        img_padded = process_ori_mat_anno(img_path, anno_dict)
        
        im_weak, im_weak_tensor, im_strong, im_strong_tensor, rot_angle = process_not_annotated_image(
            img_padded, augment_strong=self.augment_strong, config=self.config)

        ra = rot_angle * np.pi / 180.0
        aug_rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
        # rotation_matrix_new = aug_rot_mat @ rotation_matrix  # for the config.rotate_aug
        aug_rot_mat = torch.FloatTensor(aug_rot_mat)

        assert im_strong is not None, "im_strong should not be None for the AFLWFace Dataset!!!"
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "AFLWFace_weak_" in name]
        save_jpg_name = self.config.log_dir+"/AFLWFace_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "AFLWFace_strong_" in name]
        save_jpg_name = self.config.log_dir+"/AFLWFace_strong_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)

        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        # sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor)  # AFLWFace has no pose labels
        sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor, aug_rot_mat=aug_rot_mat)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_AFLWFace(phase, config):
    
    assert phase in ["ulb_train"], "unsupport phase of AFLWFace! " + phase

    mypath = config.data_dir_AFLWFace
    mat_gt = sio.loadmat(join(mypath, 'AFLWinfo_release.mat'))
    
    total_images, total_faces = 21123, 24386
    names_list, idx_gt_dict = [], {}
    
    mat_mask = mat_gt['mask_new'].copy()
    mat_lms = mat_gt['data'].reshape((total_faces, 2, 19))
    # mat_lms = np.transpose(mat_lms, (0,2,1))
    mat_bbox = mat_gt['bbox'].copy()
    
    for i in range(total_faces):
        name = mat_gt['nameList'][i, 0][0]  # "id/img_name.jpg" or "id/img_name.png", id is 0, 2 or 3
        names_list.append(name)  # names_list may have repeated ones for some images have more than one face
        idx_gt_dict[str(i)] = {
            "name": name,
            "mask": mat_mask[i],
            "landmarks": mat_lms[i],
            "bbox": mat_bbox[i]
        }
    assert len(set(names_list)) == total_images, "the total images number is not right!"
    aflw2000_names = open(join(mypath, 'AFLW2000.txt'), "r").readlines()
    aflw2000_names = [img_name.strip() for img_name in aflw2000_names]
    names_list_left = [name for name in names_list if name.split("/")[-1] not in aflw2000_names]
    print('total_faces: {:05d}, left_total_faces: {:05d}, left_total_images: {:05d}'.format(
        len(names_list), len(names_list_left), len(set(names_list_left))))  # 24386, 22246, 19123

    files_gt_dict, files_jpg = [], []
    for idx_str, gt_dict in idx_gt_dict.items():
        img_name = gt_dict["name"].split("/")[-1]
        if img_name not in aflw2000_names:
            img_path = join(mypath, 'images', img_name)
            files_jpg.append(img_path)
            files_gt_dict.append(gt_dict) 

    batch_size = round(config.batch_size * config.ulb_batch_ratio)
    shuffle = True
    augment_strong = True

    dset = Dataset_AFLWFace(files_jpg, files_gt_dict, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, 
        pin_memory=True, drop_last=True, persistent_workers=True)
    return dloader


if __name__ == '__main__':
    print("testing AFLWFace Dataset...")

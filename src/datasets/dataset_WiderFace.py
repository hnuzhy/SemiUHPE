'''
Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
'''

import os
import cv2
import json
import random
import shutil
import torch
import scipy.io as sio
import numpy as np
import torchvision.transforms as tfs

from tqdm import tqdm
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader


def process_ori_mat_anno(img_path, anno_list, ad_base=0.1, debug_vis=False):
    img_ori = Image.open(img_path).convert('RGB')
    img_w, img_h = img_ori.size
    
    f_bbox = anno_list
    
    bbox = f_bbox  # [x1,y1,w,h]
    [ori_x, ori_y, ori_w, ori_h] = [bbox[0], bbox[1], bbox[2], bbox[3]]
    cx, cy = ori_x+ori_w/2, ori_y+ori_h/2
    # pad_len = max(ori_w, ori_h)  # hint 1: we want get a squared face bbox <<<------
    pad_len = (ori_w + ori_h) / 2.0  # hint 1: we want get a squared face bbox <<<------
    
    ad = np.random.random_sample() * 0.1 + ad_base
    new_x_min = max(int(cx - (0.5 + ad) * pad_len), 0)
    new_x_max = min(int(cx + (0.5 + ad) * pad_len), img_w - 1)
    new_y_min = max(int(cy - (0.5 + ad*2) * pad_len), 0)  # hint 2: give more area above the top face <<<------
    new_y_max = min(int(cy + (0.5 + ad) * pad_len), img_h - 1)

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

    if debug_vis:
        img_cv2 = cv2.cvtColor(np.array(img_padded), cv2.COLOR_RGB2BGR)
        return img_padded, img_cv2

    return img_padded


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


class Dataset_WiderFace(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load WiderFace Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size

        img_path = self.img_files[idx]
        anno_list = self.anno_files[idx]

        img_padded = process_ori_mat_anno(img_path, anno_list)
        
        im_weak, im_weak_tensor, im_strong, im_strong_tensor, rot_angle = process_not_annotated_image(
            img_padded, augment_strong=self.augment_strong, config=self.config)

        ra = rot_angle * np.pi / 180.0
        aug_rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
        # rotation_matrix_new = aug_rot_mat @ rotation_matrix  # for the config.rotate_aug
        aug_rot_mat = torch.FloatTensor(aug_rot_mat)

        assert im_strong is not None, "im_strong should not be None for the WiderFace Dataset!!!"
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "WiderFace_weak_" in name]
        save_jpg_name = self.config.log_dir+"/WiderFace_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "WiderFace_strong_" in name]
        save_jpg_name = self.config.log_dir+"/WiderFace_strong_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)

        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        # sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor)  # WiderFace has no pose labels
        sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor, aug_rot_mat=aug_rot_mat)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_WiderFace(phase, config):

    assert phase in ["ulb_train"], "unsupport phase of WiderFace! " + phase

    db_path = config.data_dir_WiderFace
    # is_full_range = config.is_full_range  # front_range for labled 300WLP
    # assert is_full_range == False, "We now only apply WiderFace for front_range HPE."
    print("Dataset WiderFace with phase:", phase)

    imgs_root_train = os.path.join(db_path, "images_original/train")
    imgs_root_val = os.path.join(db_path, "images_original/val")
    anno_path_train = os.path.join(db_path, "wider_face_split/wider_face_train_bbx_gt.txt")
    anno_path_val = os.path.join(db_path, "wider_face_split/wider_face_val_bbx_gt.txt")

    min_head_size_thre = 25  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes

    for (imgs_root, anno_path) in [(imgs_root_val, anno_path_val), (imgs_root_train, anno_path_train)]:
        txt_lines = open(anno_path, "r").readlines()
        current_img_name = ""
        imgs_list_temp, anno_list_temp = [], [] 
        for txt_line in txt_lines:
            txt_line = txt_line.strip()
            if ".jpg" in txt_line:
                current_img_name = txt_line
                continue  # the line of one image name
            txt_elements = txt_line.split(" ")
            if len(txt_elements) == 1:
                continue  # the line of face number in the image
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = txt_elements
            
            '''
            blur: clear->0 normal blur->1 heavy blur->2
            expression: typical expression->0 exaggerate expression->1
            illumination: normal illumination->0 extreme illumination->1
            occlusion: no occlusion->0 partial occlusion->1 heavy occlusion->2
            pose: typical pose->0 atypical pose->1
            invalid: false->0(valid image) true->1(invalid image)
            '''
            if int(w) < min_head_size_thre or int(h) < min_head_size_thre:
                continue  # this face is too small
            # if int(blur) == 2:
                # continue  # this face is heavy blur
            # if int(illumination) == 1:
                # continue  # this face is under extreme illumination
            # if int(pose) == 1:
                # continue  # this face has an atypical pose
            # if int(invalid) == 1:
                # continue  # this face has an invalid image
            
            face_bbox = [int(x1), int(y1), int(w), int(h)]
            imgs_list_temp.append(os.path.join(imgs_root, current_img_name))
            anno_list_temp.append(face_bbox)

        print("Finished! left total face number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp

    print("Finished All! left total face number:", len(anno_list_all))

    batch_size = round(config.batch_size * config.ulb_batch_ratio)
    shuffle = True
    augment_strong = True

    dset = Dataset_WiderFace(imgs_list_all, anno_list_all, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=True, drop_last=True)
 
    return dloader    


if __name__ == '__main__':
    print("testing WiderFace Dataset...")
    
    """Please put this file under the main folder for running test."""
    
    db_path = "/datasdc/zhouhuayi/dataset/WiderFace"

    if os.path.exists("./debug_WiderFace"):
        shutil.rmtree("./debug_WiderFace")
    os.mkdir("./debug_WiderFace")
    
    min_head_size_thre = 25  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes
    
    for phase in ["val", "train"]:
        imgs_root = os.path.join(db_path, f"images_original/{phase}")
        anno_path = os.path.join(db_path, f"wider_face_split/wider_face_{phase}_bbx_gt.txt")

        txt_lines = open(anno_path, "r").readlines()
        current_img_name = ""
        imgs_list_temp, anno_list_temp = [], [] 
        for txt_line in txt_lines:
            txt_line = txt_line.strip()
            if ".jpg" in txt_line:
                current_img_name = txt_line
                continue  # the line of one image name
            txt_elements = txt_line.split(" ")
            if len(txt_elements) == 1:
                continue  # the line of face number in the image
            x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose = txt_elements
            
            '''
            blur: clear->0 normal blur->1 heavy blur->2
            expression: typical expression->0 exaggerate expression->1
            illumination: normal illumination->0 extreme illumination->1
            occlusion: no occlusion->0 partial occlusion->1 heavy occlusion->2
            pose: typical pose->0 atypical pose->1
            invalid: false->0(valid image) true->1(invalid image)
            '''
            if int(w) < min_head_size_thre or int(h) < min_head_size_thre:
                continue  # this face is too small
            # if int(blur) == 2:
                # continue  # this face is heavy blur
            # if int(illumination) == 1:
                # continue  # this face is under extreme illumination
            # if int(pose) == 1:
                # continue  # this face has an atypical pose
            # if int(invalid) == 1:
                # continue  # this face has an invalid image
            
            face_bbox = [int(x1), int(y1), int(w), int(h)]
            imgs_list_temp.append(os.path.join(imgs_root, current_img_name))
            anno_list_temp.append(face_bbox)

        print("Finished! left total face number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp

    print("Finished All! left total face number:", len(anno_list_all))
    

    vis_img_num = 300
    random.seed(666)
    index_arr = np.arange(len(imgs_list_all))
    random.shuffle(index_arr)
    imgs_list = [imgs_list_all[i] for i in index_arr[:vis_img_num]]
    anno_list = [anno_list_all[i] for i in index_arr[:vis_img_num]]
    for idx, (img_path, anno_info) in enumerate(zip(imgs_list, anno_list)):
        img_padded, img_cv2 = process_ori_mat_anno(img_path, anno_info, ad_base=0.1, debug_vis=True)
        
        [face_x, face_y, face_w, face_h] = anno_info
        save_img_name =f"{str(idx).zfill(4)}_{face_w}x{face_h}.jpg"
        cv2.imwrite(f"./debug_WiderFace/{save_img_name}", img_cv2)
    
    '''
    Finished! left total face number: 9381
    Finished! left total face number: 36030
    Finished All! left total face number: 45411
    
    with comment out int(blur), int(illumination), int(pose) and int(invalid)
    Finished! left total face number: 12737
    Finished! left total face number: 49363
    Finished All! left total face number: 62100
    '''
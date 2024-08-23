'''
Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
'''

import os
import cv2
import json
import shutil
import random
import torch
import scipy.io as sio
import numpy as np
import torchvision.transforms as tfs

from typing import Dict, List
from tqdm import tqdm
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

def process_ori_mat_anno(img_path, anno_list, ad_base=0.1, debug_vis=False):
    img_ori = Image.open(img_path).convert('RGB')
    img_w_pil, img_h_pil = img_ori.size
    
    [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, is_front_flag] = anno_list
    # assert img_w_pil == img_w and img_h_pil == img_h, "this image's size info is not right."
    
    bbox = h_bbox[:4]  # [x1,y1,x2,y2,v]
    [ori_x, ori_y, ori_w, ori_h] = [bbox[0], bbox[1], head_w, head_h]
    cx, cy = ori_x+ori_w/2, ori_y+ori_h/2
    # pad_len = max(ori_w, ori_h)  # hint 1: we want get a squared face bbox <<<------
    pad_len = (ori_w + ori_h) / 2.0  # hint 1: we want get a squared face bbox <<<------
    
    ad = np.random.random_sample() * 0.1 + ad_base
    new_x_min = max(int(cx - (0.5 + ad) * pad_len), 0)
    new_x_max = min(int(cx + (0.5 + ad) * pad_len), img_w - 1)
    new_y_min = max(int(cy - (0.5 + ad) * pad_len), 0)  # hint 2: give more area above the top face <<<------
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


class Dataset_CrowdHuman(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load CrowdHuman Dataset with phase %s, Length:%d'%(phase, self.size))

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

        assert im_strong is not None, "im_strong should not be None for the CrowdHuman Dataset!!!"
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "CrowdHuman_weak_" in name]
        save_jpg_name = self.config.log_dir+"/CrowdHuman_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "CrowdHuman_strong_" in name]
        save_jpg_name = self.config.log_dir+"/CrowdHuman_strong_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)

        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        # sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor)  # CrowdHuman has no pose labels
        sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor, aug_rot_mat=aug_rot_mat)
        return sample
        
    def __len__(self):
        return self.size


def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def get_dataloader_CrowdHuman(phase, config):
    assert phase in ["ulb_train"], "unsupport phase of CrowdHuman! " + phase

    db_path = config.data_dir_CrowdHuman
    # is_full_range = config.is_full_range  # full_range for labled DAD3DHeads
    # assert is_full_range == True, "We now only apply CrowdHuman for full_range HPE."
    print("Dataset CrowdHuman with phase:", phase)
    
    imgs_root_train = os.path.join(db_path, "yolov5_format/images/train")
    imgs_root_val = os.path.join(db_path, "yolov5_format/images/val")
    anno_path_train = os.path.join(db_path, "BFJDet/instances_train_full_bhf_new.json")
    anno_path_val = os.path.join(db_path, "BFJDet/instances_val_full_bhf_new.json")
    
    rm_ignore = True  # remove instances which have been labeled as ignore
    min_head_size_thre = 30  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes

    for (imgs_root, anno_path) in [(imgs_root_val, anno_path_val), (imgs_root_train, anno_path_train)]:
        anno_dict = json.load(open(anno_path, "r"))
        imgs_dict_list = anno_dict['images']
        annos_dict_list = anno_dict['annotations']
        images_labels_dict = sort_labels_by_image_id(annos_dict_list)
        print("Processing annotations of CrowdHuman by BFJDet \n[%s][instance number: %d]..."%(
            anno_path, len(annos_dict_list)))
        imgs_list_temp, anno_list_temp = [], [] 
        for imgs_dict in tqdm(imgs_dict_list):
            img_name = imgs_dict["file_name"]
            img_h, img_w = imgs_dict["height"], imgs_dict["width"]
            image_id = str(imgs_dict['id'])
            if image_id not in images_labels_dict:
                continue  # this image has no person instances
        
            img_path_src = os.path.join(imgs_root, img_name)
            assert os.path.exists(img_path_src), "original image missing :%s"%(img_path_src)
            
            anno_BFJDet_list = images_labels_dict[image_id]
            assert len(anno_BFJDet_list) != 0, "Each image has at least one anno by BFJDet! --> "+img_path_src
            ''' coco format of an instance_id in BFJDet
            anno_BFJDet_instance= {
                "segmentation": [], 
                "vbox": [959, 599, 38, 160],   # visible body [x,y,w,h] 
                "area": 6400, 
                "iscrowd": 0, 
                "height": 160, 
                "ignore": 0, 
                "image_id": 21, 
                "vis_ratio": 0.95, 
                "bbox": [959, 598, 40, 160],   # full body [x,y,w,h], x or y may be negative
                "category_id": 1, 
                "id": 567, 
                "h_bbox": [977, 600, 19, 21],  # head bbox [x,y,w,h], if not visible, the w and h will be 0
                "f_bbox": [978, 603, 11, 14]  # face bbox [x,y,w,h], if not visible, the w and h will be 0
            }
            '''
            for anno_BFJDet_instance in anno_BFJDet_list:
            
                if rm_ignore and anno_BFJDet_instance["ignore"] == 1:
                    # remove this person instance with "ignore" flag is 1 
                    # This may be a pitcure of pitcure, a traffic sign or a super tiny person
                    # print("[warning] ignore this instance:", img_name, anno_BFJDet_instance["id"])
                    continue
                    
                instance_id = str(anno_BFJDet_instance["id"])
                p_bbox = anno_BFJDet_instance["bbox"]
                h_bbox = anno_BFJDet_instance["h_bbox"]
                f_bbox = anno_BFJDet_instance["f_bbox"]
                
                if h_bbox[2] == 1 and h_bbox[3] == 1:  # illegal head bbox are labeled as [xx, xx, 1, 1] by BFJDet
                    continue  # this person has no labeled head or face
                head_h, head_w = h_bbox[3], h_bbox[2]
                if head_h < min_head_size_thre or head_w < min_head_size_thre:  # this head is too small
                    continue
                h_bbox = [h_bbox[0], h_bbox[1], h_bbox[0]+head_w, h_bbox[1]+head_h]  # xywh --> xyxy
                [x0, y0, x1, y1] = h_bbox
                if (x0<0 and x1<0) or (y0<0 and y1<0) or (x0>img_w and x1>img_w) or (y0>img_h and y1>img_h):
                    continue  # some head bboxes are not rightly annotated
                    
                h_bbox = [h_bbox[0], h_bbox[1], h_bbox[0]+head_w, h_bbox[1]+head_h]  # xywh --> xyxy
                if f_bbox[2] == 1 and f_bbox[3] == 1:  
                    f_bbox = [0,0,0,0]  # this person has only labeled head (back maybe), front_flag = False (not reliable)
                    anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, False]
                else:  # this person has both labeled head and face (front maybe), front_flag = True (not reliable)
                    f_bbox = [f_bbox[0], f_bbox[1], f_bbox[0]+f_bbox[2], f_bbox[1]+f_bbox[3]]  # xywh --> xyxy
                    anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, True]
                imgs_list_temp.append(img_path_src)
                anno_list_temp.append(anno_info)
            
        print("Finished! left total head number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp
        
    print("Finished All! left total head number:", len(anno_list_all))


    batch_size = round(config.batch_size * config.ulb_batch_ratio)
    shuffle = True
    augment_strong = True

    dset = Dataset_CrowdHuman(imgs_list_all, anno_list_all, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=True, drop_last=True)
 
    return dloader    


if __name__ == '__main__':
    print("testing CrowdHuman Dataset...")
    
    """Please put this file under the main folder for running test."""
    
    db_path = "/datasdc/zhouhuayi/dataset/CrowdHuman"
    
    if os.path.exists("./debug_CrowdHuman"):
        shutil.rmtree("./debug_CrowdHuman")
    os.mkdir("./debug_CrowdHuman")
    
    rm_ignore = True  # remove instances which have been labeled as ignore
    min_head_size_thre = 30  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes

    for phase in ["val", "train"]:
        imgs_root = os.path.join(db_path, f"yolov5_format/images/{phase}")
        anno_path = os.path.join(db_path, f"BFJDet/instances_{phase}_full_bhf_new.json")
        
        anno_dict = json.load(open(anno_path, "r"))
        imgs_dict_list = anno_dict['images']
        annos_dict_list = anno_dict['annotations']
        images_labels_dict = sort_labels_by_image_id(annos_dict_list)
        print("Processing annotations of CrowdHuman by BFJDet \n[%s][instance number: %d]..."%(
            anno_path, len(annos_dict_list)))
        imgs_list_temp, anno_list_temp = [], [] 
        for imgs_dict in tqdm(imgs_dict_list):
            img_name = imgs_dict["file_name"]
            img_h, img_w = imgs_dict["height"], imgs_dict["width"]
            image_id = str(imgs_dict['id'])
            if image_id not in images_labels_dict:
                continue  # this image has no person instances
        
            img_path_src = os.path.join(imgs_root, img_name)
            assert os.path.exists(img_path_src), "original image missing :%s"%(img_path_src)

            anno_BFJDet_list = images_labels_dict[image_id]
            assert len(anno_BFJDet_list) != 0, "Each image has at least one anno by BFJDet! --> "+img_path_src

            for anno_BFJDet_instance in anno_BFJDet_list:
            
                if rm_ignore and anno_BFJDet_instance["ignore"] == 1:
                    # remove this person instance with "ignore" flag is 1 
                    # This may be a pitcure of pitcure, a traffic sign or a super tiny person
                    # print("[warning] ignore this instance:", img_name, anno_BFJDet_instance["id"])
                    continue
                    
                instance_id = str(anno_BFJDet_instance["id"])
                p_bbox = anno_BFJDet_instance["bbox"]
                h_bbox = anno_BFJDet_instance["h_bbox"]
                f_bbox = anno_BFJDet_instance["f_bbox"]
                
                if h_bbox[2] == 1 and h_bbox[3] == 1:  # illegal head bbox are labeled as [xx, xx, 1, 1] by BFJDet
                    continue  # this person has no labeled head or face
                head_h, head_w = h_bbox[3], h_bbox[2]
                if head_h < min_head_size_thre or head_w < min_head_size_thre:  # this head is too small
                    continue
                h_bbox = [h_bbox[0], h_bbox[1], h_bbox[0]+head_w, h_bbox[1]+head_h]  # xywh --> xyxy
                [x0, y0, x1, y1] = h_bbox
                if (x0<0 and x1<0) or (y0<0 and y1<0) or (x0>img_w and x1>img_w) or (y0>img_h and y1>img_h):
                    continue  # some head bboxes are not rightly annotated
                    
                if f_bbox[2] == 1 and f_bbox[3] == 1:  
                    f_bbox = [0,0,0,0]  # this person has only labeled head (back maybe), front_flag = False (not reliable)
                    anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, False]
                else:  # this person has both labeled head and face (front maybe), front_flag = True (not reliable)
                    f_bbox = [f_bbox[0], f_bbox[1], f_bbox[0]+f_bbox[2], f_bbox[1]+f_bbox[3]]  # xywh --> xyxy
                    anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, True]
                imgs_list_temp.append(img_path_src)
                anno_list_temp.append(anno_info)

        print("Finished! left total head number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp
        
    print("Finished All! left total head number:", len(anno_list_all))
    
    
    vis_img_num = 300
    random.seed(666)
    index_arr = np.arange(len(imgs_list_all))
    random.shuffle(index_arr)
    imgs_list = [imgs_list_all[i] for i in index_arr[:vis_img_num]]
    anno_list = [anno_list_all[i] for i in index_arr[:vis_img_num]]
    for idx, (img_path, anno_info) in enumerate(zip(imgs_list, anno_list)):
        try:
            img_padded, img_cv2 = process_ori_mat_anno(img_path, anno_info, ad_base=0.1, debug_vis=True)
        except:
            print(img_path, "\n", anno_info)
        [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, is_front_flag] = anno_info
        save_img_name =f"{str(idx).zfill(4)}_{str(instance_id).zfill(8)}_{head_w}x{head_h}_{is_front_flag}.jpg"
        cv2.imwrite(f"./debug_CrowdHuman/{save_img_name}", img_cv2)
    
    """
    min_head_size_thre = 30
    
    Processing annotations of CrowdHuman by BFJDet
    [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_val_full_bhf_new.json][instance number: 127716]...
    100%|█████████████████████████████████| 4370/4370 [00:00<00:00, 24558.30it/s]
    Finished! left total head number: 37995
    Processing annotations of CrowdHuman by BFJDet
    [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_train_full_bhf_new.json][instance number: 438792]...
    100%|█████████████████████████████████| 15000/15000 [00:00<00:00, 25228.89it/s]
    Finished! left total head number: 126875
    Finished All! left total head number: 164870
    
    min_head_size_thre = 0
    
    Processing annotations of CrowdHuman by BFJDet
    [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_val_full_bhf_new.json][instance number: 127716]...
    100%|█████████████████████████████████| 4370/4370 [00:00<00:00, 7929.60it/s]
    Finished! left (original) total head number: 99128 (99481)
    Processing annotations of CrowdHuman by BFJDet
    [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_train_full_bhf_new.json][instance number: 438792]...
    100%|█████████████████████████████████| 15000/15000 [00:02<00:00, 5502.19it/s]
    Finished! left (original) total head number: 339565 (338440)
    Finished All! left (original) total head number: 439046 (437568)
    """


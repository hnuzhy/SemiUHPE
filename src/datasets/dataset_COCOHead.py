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


class Dataset_COCOHead(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load COCOHead Dataset with phase %s, Length:%d'%(phase, self.size))

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
        
        assert im_strong is not None, "im_strong should not be None for the COCOHead Dataset!!!"
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "COCOHead_weak_" in name]
        save_jpg_name = self.config.log_dir+"/COCOHead_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)

        jpg_names = [name for name in os.listdir(self.config.log_dir) if "COCOHead_strong_" in name]
        save_jpg_name = self.config.log_dir+"/COCOHead_strong_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)

        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        # sample = dict(idx=idx, img=im_weak_tensor, img_strong=im_strong_tensor)  # COCOHead has no pose labels
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

def get_dataloader_COCOHead(phase, config):
    
    assert phase in ["ulb_train"], "unsupport phase of COCOHead! " + phase

    db_path = config.data_dir_COCOHead
    # is_full_range = config.is_full_range  # full_range for labled DAD3DHeads
    # assert is_full_range == True, "We now only apply COCOHead for full_range HPE."
    print("Dataset COCOHead with phase:", phase)

    imgs_root_train = os.path.join(db_path, "images/train2017")
    imgs_root_val = os.path.join(db_path, "images/val2017")
    anno_path_train = os.path.join(db_path, "annotations_HumanParts/person_humanparts_train2017.json")
    anno_path_val = os.path.join(db_path, "annotations_HumanParts/person_humanparts_val2017.json")

    min_head_size_thre = 30  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes

    for (imgs_root, anno_path) in [(imgs_root_val, anno_path_val), (imgs_root_train, anno_path_train)]:
        anno_dict = json.load(open(anno_path, "r"))
        imgs_dict_list = anno_dict['images']
        annos_dict_list = anno_dict['annotations']
        images_labels_dict = sort_labels_by_image_id(annos_dict_list)
        print("Processing annotations of HumanParts by Hier-R-CNN \n[%s][instance number: %d]..."%(
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

            anno_HierRCNN_list = images_labels_dict[image_id]
            assert len(anno_HierRCNN_list) != 0, "Each image has at least one anno by HierRCNN! --> "+img_path_src
            ''' coco format of an instance_id in HierRCNN
            anno_HierRCNN_instance = {
                "hier": [426, 157, 443, 178, 1, 426, 167, 431, 176, 1, 0, 0, 0, 0, 0, 412, 213, 420, 216, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 5*6, (head, face, lefthand, righthand, leftfoot, rightfoot), [x1,y1,x2,y2,v]
                "segmentation": [], 
                "difficult": 0, 
                "id": 1, 
                "bbox": [412.0, 157.0, 54.0, 139.0],  # full body [x,y,w,h], x or y may be negative
                "image_id": 1, 
                "iscrowd": 0, 
                "category_id": 1,  # person category
                "area": 7506.0
            }
            the original defination in HumanParts by HierRCNN
            "categories": [
                {"id": 1, "supercategory": "person", "name": "person"}, 
                {"id": 2, "supercategory": "head", "name": "head"}, 
                {"id": 3, "supercategory": "face", "name": "face"}, 
                {"id": 4, "supercategory": "lefthand", "name": "lefthand"}, 
                {"id": 5, "supercategory": "righthand", "name": "righthand"}, 
                {"id": 6, "supercategory": "leftfoot", "name": "leftfoot"}, 
                {"id": 7, "supercategory": "rightfoot", "name": "rightfoot"}
            ]      
            '''
            for anno_HierRCNN_instance in anno_HierRCNN_list:
                instance_id = anno_HierRCNN_instance["id"]
                p_bbox = anno_HierRCNN_instance["bbox"]
                all_bbox_list = anno_HierRCNN_instance["hier"]
                h_bbox, f_bbox = all_bbox_list[:5], all_bbox_list[5:10]

                if h_bbox[-1] == 0:  # this person has no labeled head or face
                    continue
                head_h, head_w = h_bbox[3]-h_bbox[1], h_bbox[2]-h_bbox[0]
                if head_h < min_head_size_thre or head_w < min_head_size_thre:  # this head is too small
                    continue
                anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w]
                if h_bbox[-1] == 1 and f_bbox[-1] == 0:  # this person has only labeled head (back maybe)
                    imgs_list_temp.append(img_path_src)
                    anno_list_temp.append(anno_info + [False])  # is_front_flag = False (not reliable)
                if h_bbox[-1] == 1 and f_bbox[-1] == 1:  # this person has both labeled head and face (front maybe)
                    imgs_list_temp.append(img_path_src)
                    anno_list_temp.append(anno_info + [True])  # is_front_flag = True (not reliable)
            
        print("Finished! left total head number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp
        
    print("Finished All! left total head number:", len(anno_list_all))


    batch_size = round(config.batch_size * config.ulb_batch_ratio)
    shuffle = True
    augment_strong = True

    dset = Dataset_COCOHead(imgs_list_all, anno_list_all, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=True, drop_last=True)
 
    return dloader    
    

if __name__ == '__main__':
    print("testing COCOHead Dataset...")
    
    """Please put this file under the main folder for running test."""
    
    db_path = "/datasdc/zhouhuayi/dataset/coco"

    if os.path.exists("./debug_COCOHead"):
        shutil.rmtree("./debug_COCOHead")
    os.mkdir("./debug_COCOHead")

    min_head_size_thre = 30  # tiny head with size (width or height) smaller than this threshold will be removed
    imgs_list_all, anno_list_all = [], []  # person with both head and/or face bboxes

    for phase in ["val", "train"]:
        imgs_root = os.path.join(db_path, f"images/{phase}2017")
        anno_path = os.path.join(db_path, f"annotations_HumanParts/person_humanparts_{phase}2017.json")
        
        anno_dict = json.load(open(anno_path, "r"))
        imgs_dict_list = anno_dict['images']
        annos_dict_list = anno_dict['annotations']
        images_labels_dict = sort_labels_by_image_id(annos_dict_list)
        print("Processing annotations of HumanParts by Hier-R-CNN \n[%s][instance number: %d]..."%(
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

            anno_HierRCNN_list = images_labels_dict[image_id]
            assert len(anno_HierRCNN_list) != 0, "Each image has at least one anno by HierRCNN! --> "+img_path_src

            for anno_HierRCNN_instance in anno_HierRCNN_list:
                instance_id = anno_HierRCNN_instance["id"]
                p_bbox = anno_HierRCNN_instance["bbox"]
                all_bbox_list = anno_HierRCNN_instance["hier"]
                h_bbox, f_bbox = all_bbox_list[:5], all_bbox_list[5:10]

                if h_bbox[-1] == 0:  
                    continue  # this person has no labeled head or face
                head_h, head_w = h_bbox[3]-h_bbox[1], h_bbox[2]-h_bbox[0]
                if head_h < min_head_size_thre or head_w < min_head_size_thre:  
                    continue  # this head size is too small
                anno_info = [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w]
                if h_bbox[-1] == 1 and f_bbox[-1] == 0:  # this person has only labeled head (back maybe)
                    imgs_list_temp.append(img_path_src)
                    anno_list_temp.append(anno_info + [False])  # is_front_flag = False (not reliable)
                if h_bbox[-1] == 1 and f_bbox[-1] == 1:  # this person has both labeled head and face (front maybe)
                    imgs_list_temp.append(img_path_src)
                    anno_list_temp.append(anno_info + [True])  # is_front_flag = True (not reliable)

        print("Finished! left total head number:", len(anno_list_temp))
        imgs_list_all += imgs_list_temp
        anno_list_all += anno_list_temp
        
    print("Finished All! left total head number:", len(anno_list_all))
    

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--rotate_aug', action='store_true', help='whether to use Rotation strong_aug')
    args_config = parser.parse_args()
    args_config.rotate_aug = True
    vis_img_num = 300
    random.seed(666)
    index_arr = np.arange(len(imgs_list_all))
    random.shuffle(index_arr)
    imgs_list = [imgs_list_all[i] for i in index_arr[:vis_img_num]]
    anno_list = [anno_list_all[i] for i in index_arr[:vis_img_num]]
    for idx, (img_path, anno_info) in enumerate(zip(imgs_list, anno_list)):
        img_padded, img_cv2 = process_ori_mat_anno(img_path, anno_info, ad_base=0.1, debug_vis=True)
        
        [h_bbox, f_bbox, image_id, img_h, img_w, instance_id, head_h, head_w, is_front_flag] = anno_info
        save_img_name =f"{str(idx).zfill(4)}_{str(instance_id).zfill(8)}_{head_w}x{head_h}_{is_front_flag}.jpg"
        cv2.imwrite(f"./debug_COCOHead/{save_img_name}", img_cv2)
        
        im_weak, im_weak_tensor, im_strong, im_strong_tensor, rot_angle = process_not_annotated_image(
            img_padded, augment_strong=True, config=args_config)
        
        img_cv2_weak = cv2.cvtColor(np.array(im_weak), cv2.COLOR_RGB2BGR)
        save_img_name =f"{str(idx).zfill(4)}_{str(instance_id).zfill(8)}_{head_w}x{head_h}_weak.jpg"
        cv2.imwrite(f"./debug_COCOHead/{save_img_name}", img_cv2_weak)
        img_cv2_rot = cv2.cvtColor(np.array(im_strong), cv2.COLOR_RGB2BGR)
        save_img_name =f"{str(idx).zfill(4)}_{str(instance_id).zfill(8)}_{head_w}x{head_h}_{rot_angle}.jpg"
        cv2.imwrite(f"./debug_COCOHead/{save_img_name}", img_cv2_rot)
    
    '''
    min_head_size_thre = 30 
    
    Processing annotations of HumanParts by Hier-R-CNN
    [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_val2017.json][instance number: 10777]...
    100%|████████████████████████████| 2693/2693 [00:00<00:00, 97153.50it/s]
    Finished! left total head number: 3013
    Processing annotations of HumanParts by Hier-R-CNN
    [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_train2017.json][instance number: 257306]...
    100%|████████████████████████████| 64115/64115 [00:01<00:00, 33256.10it/s]
    Finished! left total head number: 71115
    Finished All! left total head number: 74128
    
    min_head_size_thre = 0 
    
    Processing annotations of HumanParts by Hier-R-CNN
    [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_val2017.json][instance number: 10777]...
    100%|████████████████████████████| 2693/2693 [00:00<00:00, 86013.91it/s]
    Finished! left total head number: 9351
    Processing annotations of HumanParts by Hier-R-CNN
    [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_train2017.json][instance number: 257306]...
    100%|████████████████████████████| 64115/64115 [00:02<00:00, 25503.02it/s]
    Finished! left total head number: 223049
    Finished All! left total head number: 232400
    '''



        
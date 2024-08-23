'''
Written by Huayi Zhou (sjtu_zhy@sjtu.edu.cn)
'''

import os
import cv2
import random
import torch
import shutil
import scipy.io as sio
import numpy as np
import torchvision.transforms as tfs

from PIL import Image, ImageFilter
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
    cont_labels = np.array([pitch, yaw, roll])
    
    lms_path_tmp = anno_path[:-4] + "_pts.mat"
    lms_path = lms_path_tmp.replace("300W_LP", "300W_LP/landmarks")
    if '_Flip' in lms_path:
        lms_path = lms_path.replace("_Flip", "")
    lms_contents = sio.loadmat(lms_path)
    pt2d = lms_contents["pts_2d"]  # array shape [68, 2]
    pt2d = pt2d.T  # shape [68, 2] --> [2, 68]
    if '_Flip' in lms_path_tmp: 
        pt2d[0, :] = img_w - pt2d[0, :]

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
    

class Dataset_300WLP(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_strong=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_strong = augment_strong
        self.config = config
        self.size = len(img_files)
        print('Load 300WLP Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size
        
        img_path = self.img_files[idx]
        anno_path = self.anno_files[idx]

        img, cont_labels = process_ori_mat_anno(img_path, anno_path)
        pitch, yaw, roll = cont_labels  # degrees, X-Y-Z
        while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
        while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
        while abs(roll) > 180: roll = roll - roll/abs(roll)*360
        pitch = np.clip(pitch, -89.99, 89.99)
        yaw = np.clip(yaw, -89.99, 89.99)
        roll = np.clip(roll, -89.99, 89.99)
        # assert pitch > -90.0 and pitch < 90.0, "an illegal pitch label value" + str(pitch)
        # assert yaw > -90.0 and yaw < 90.0, "an illegal yaw label value" + str(yaw)
        # assert roll > -90.0 and roll < 90.0, "an illegal roll label value" + str(roll)
        
        im_weak, im_weak_tensor, im_strong, im_strong_tensor, trans_euler, Rot, rot_angle = process_annotated_image(
            img, yaw, pitch, roll, augment_strong=self.augment_strong, config=self.config)

        ra = rot_angle * np.pi / 180.0
        aug_rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
        # rotation_matrix_new = aug_rot_mat @ rotation_matrix  # for the config.rotate_aug
        aug_rot_mat = torch.FloatTensor(aug_rot_mat)
        
        Rot = aug_rot_mat @ Rot  # for the config.rotate_aug

        if im_strong is None:
            im_strong_tensor = torch.zeros_like(im_weak_tensor)
            
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "300WLP_weak_" in name]
        save_jpg_name = self.config.log_dir+"/300WLP_weak_"+str(idx).zfill(8)+".jpg"
        if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)
        if im_strong is not None:
            jpg_names = [name for name in os.listdir(self.config.log_dir) if "300WLP_strong_" in name]
            save_jpg_name = self.config.log_dir+"/300WLP_strong_"+str(idx).zfill(8)+".jpg"
            if len(jpg_names) < 10 and idx%10 == 0: im_strong.save(save_jpg_name)
        
        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, 
            img=im_weak_tensor, img_strong=im_strong_tensor, aug_rot_mat=aug_rot_mat)
        return sample
        
    def __len__(self):
        return self.size


def get_dataloader_300WLP(phase, config):
    
    assert phase in ["train", "ulb_train", "train_all"], "unsupport phase of 300WLP! " + phase
    
    use_flip_part = True
    
    sub_folders = ["AFW", "HELEN", "IBUG", "LFPW"]
    mat_jpg_dict = {}
    for sub_folder in sub_folders:
        mat_jpg_dict[sub_folder] = {"files_mat": [], "files_jpg": [], "files_mat_flip": [], "files_jpg_flip": []}

        mypath = join(config.data_dir_300WLP, sub_folder)
        files_mat = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.mat')]
        files_jpg = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.jpg')]

        files_mat.sort()
        files_jpg.sort()
        print(sub_folder, "\t", len(files_mat), len(files_jpg))
        
        mat_jpg_dict[sub_folder]["files_mat"] = files_mat
        mat_jpg_dict[sub_folder]["files_jpg"] = files_jpg
        
        if use_flip_part:
            files_mat_flip = [file_mat.replace("LP/"+sub_folder, "LP/"+sub_folder+"_Flip") for file_mat in files_mat]
            files_jpg_flip = [file_jpg.replace("LP/"+sub_folder, "LP/"+sub_folder+"_Flip") for file_jpg in files_jpg]
            mat_jpg_dict[sub_folder]["files_mat_flip"] = files_mat_flip
            mat_jpg_dict[sub_folder]["files_jpg_flip"] = files_jpg_flip


    if phase == 'train_all':
        # fully labeled data
        sampled_jpg_files, sampled_mat_files = [], []
        for sub_folder in sub_folders:
            sampled_jpg_files += mat_jpg_dict[sub_folder]["files_jpg"]
            sampled_mat_files += mat_jpg_dict[sub_folder]["files_mat"]
            if use_flip_part:
                sampled_jpg_files += mat_jpg_dict[sub_folder]["files_jpg_flip"]
                sampled_mat_files += mat_jpg_dict[sub_folder]["files_mat_flip"] 
        batch_size = config.batch_size
        shuffle = True
        augment_strong = False
        
    if phase == 'train':
        # partially labeled data
        assert config.ss_ratio <= 1., "we do not use all labeled 300WLP dataset!!!"
        sampled_jpg_files, sampled_mat_files = [], []
        for sub_folder in sub_folders:
            jpg_files = mat_jpg_dict[sub_folder]["files_jpg"]
            mat_files = mat_jpg_dict[sub_folder]["files_mat"]
            total_num = len(jpg_files)
            labeled_num = int(total_num * config.ss_ratio)
            index_arr = np.arange(total_num)
            random.seed(666)
            random.shuffle(index_arr)
            sampled_jpg_files += [jpg_files[index_arr[i]] for i in range(labeled_num)]
            sampled_mat_files += [mat_files[index_arr[i]] for i in range(labeled_num)]
            if use_flip_part:
                jpg_flip_files = mat_jpg_dict[sub_folder]["files_jpg_flip"]
                mat_flip_files = mat_jpg_dict[sub_folder]["files_mat_flip"]
                sampled_jpg_files += [jpg_flip_files[index_arr[i]] for i in range(labeled_num)]
                sampled_mat_files += [mat_flip_files[index_arr[i]] for i in range(labeled_num)]  
        batch_size = config.batch_size
        shuffle = True
        augment_strong = False

    if phase == 'ulb_train':
        # left unlabeled data 
        sampled_jpg_files, sampled_mat_files = [], []
        for sub_folder in sub_folders:
            jpg_files = mat_jpg_dict[sub_folder]["files_jpg"]
            mat_files = mat_jpg_dict[sub_folder]["files_mat"]
            total_num = len(jpg_files)
            labeled_num = int(total_num * config.ss_ratio)
            unlabeled_num = total_num - labeled_num
            index_arr = np.arange(total_num)
            random.seed(666)
            random.shuffle(index_arr)
            sampled_jpg_files += [jpg_files[index_arr[labeled_num + i]] for i in range(unlabeled_num)]
            sampled_mat_files += [mat_files[index_arr[labeled_num + i]] for i in range(unlabeled_num)]
            if use_flip_part:
                jpg_flip_files = mat_jpg_dict[sub_folder]["files_jpg_flip"]
                mat_flip_files = mat_jpg_dict[sub_folder]["files_mat_flip"]
                sampled_jpg_files += [jpg_flip_files[index_arr[labeled_num + i]] for i in range(unlabeled_num)]
                sampled_mat_files += [mat_flip_files[index_arr[labeled_num + i]] for i in range(unlabeled_num)]  
        batch_size = round(config.batch_size * config.ulb_batch_ratio)
        shuffle = True
        augment_strong = True

    dset = Dataset_300WLP(sampled_jpg_files, sampled_mat_files, phase, augment_strong, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers, 
        pin_memory=True, drop_last=True)
    return dloader


if __name__ == '__main__':
    print("testing 300WLP Dataset...")

    data_dir_300WLP = "/datasdc/zhouhuayi/dataset/headpose/300W_LP"
    
    sub_folders = ["AFW", "HELEN", "IBUG", "LFPW"]
    mat_jpg_dict = {}
    for sub_folder in sub_folders:
        mat_jpg_dict[sub_folder] = {"files_mat": [], "files_jpg": []}

        mypath = join(data_dir_300WLP, sub_folder)
        files_mat = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.mat')]
        files_jpg = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('.jpg')]
        files_mat.sort()
        files_jpg.sort()
        print(sub_folder, "\t", len(files_mat), len(files_jpg))
        
        mat_jpg_dict[sub_folder]["files_mat"] = files_mat
        mat_jpg_dict[sub_folder]["files_jpg"] = files_jpg
        
    # fully labeled data
    sampled_jpg_files, sampled_mat_files = [], []
    for sub_folder in sub_folders:
        sampled_jpg_files += mat_jpg_dict[sub_folder]["files_jpg"]
        sampled_mat_files += mat_jpg_dict[sub_folder]["files_mat"]


    if os.path.exists("./debug_300WLP"):
        shutil.rmtree("./debug_300WLP")
    os.mkdir("./debug_300WLP")
    
    from src.vis_plot import convert_euler_bbox_to_6dof
    from src.vis_plot import convert_rotmat_bbox_to_6dof
    from src.renderer import Renderer
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy", 
        triangles_path="pose_references/triangles.npy"
    )

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--rotate_aug', action='store_true', help='whether to use Rotation strong_aug')
    args_config = parser.parse_args()
    args_config.rotate_aug = True
    vis_img_num = 50
    random.seed(666)
    index_arr = np.arange(len(sampled_jpg_files))
    random.shuffle(index_arr)
    imgs_list = [sampled_jpg_files[i] for i in index_arr[:vis_img_num]]
    anno_list = [sampled_mat_files[i] for i in index_arr[:vis_img_num]]
    for idx, (img_path, anno_info) in enumerate(zip(imgs_list, anno_list)):
        img, cont_labels = process_ori_mat_anno(img_path, anno_info)
        pitch, yaw, roll = cont_labels  # degrees, X-Y-Z
        while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
        while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
        while abs(roll) > 180: roll = roll - roll/abs(roll)*360
        pitch = np.clip(pitch, -89.99, 89.99)
        yaw = np.clip(yaw, -89.99, 89.99)
        roll = np.clip(roll, -89.99, 89.99)
        
        im_weak, im_weak_tensor, im_strong, im_strong_tensor, trans_euler, Rot, rot_angle = process_annotated_image(
            img, yaw, pitch, roll, augment_strong=True, config=args_config)

        ra = rot_angle * np.pi / 180.0
        aug_rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
        
        rot_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # one more step for 300WLP labels
        Rot_trans = rot_180 @ np.transpose(Rot.numpy())  # one more step for 300WLP labels

        Rot_trans_new = aug_rot_mat @ Rot_trans  # for the config.rotate_aug


        """Case 1 img_3dvis: original image with 3d vis by img2pose"""
        im_weak_cv2 = cv2.cvtColor(np.array(im_weak), cv2.COLOR_RGB2BGR)
        (h, w, c) = im_weak_cv2.shape
        global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        scaled_bbox = [20, 20, w-20, h-20]
        global_pose = convert_rotmat_bbox_to_6dof(Rot_trans, scaled_bbox, global_intrinsics)
        trans_vertices = renderer.transform_vertices(im_weak_cv2, [global_pose])
        img_3dvis = renderer.render(im_weak_cv2, trans_vertices, alpha=1.0)

        """Case 2 img_3dvisRot: original image with 3d vis by img2pose"""
        im_strong_cv2 = cv2.cvtColor(np.array(im_strong), cv2.COLOR_RGB2BGR)
        (h, w, c) = im_strong_cv2.shape
        global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        scaled_bbox = [20, 20, w-20, h-20]
        global_pose = convert_rotmat_bbox_to_6dof(Rot_trans_new, scaled_bbox, global_intrinsics)
        trans_vertices = renderer.transform_vertices(im_strong_cv2, [global_pose])
        img_3dvisRot = renderer.render(im_strong_cv2, trans_vertices, alpha=1.0)


        img_name = os.path.split(img_path)[-1]
        cv2.imwrite(f"./debug_300WLP/ulb_{str(idx).zfill(4)}_weak_{img_name}", im_weak_cv2)
        cv2.imwrite(f"./debug_300WLP/ulb_{str(idx).zfill(4)}_weak3dvis_{img_name}", img_3dvis)
        cv2.imwrite(f"./debug_300WLP/ulb_{str(idx).zfill(4)}_strong_{img_name}", im_strong_cv2)
        cv2.imwrite(f"./debug_300WLP/ulb_{str(idx).zfill(4)}_strong3dvis_{img_name}", img_3dvisRot)



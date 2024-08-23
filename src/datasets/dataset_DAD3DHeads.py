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

from scipy.spatial.transform import Rotation
from tqdm import tqdm
from PIL import Image, ImageFilter
from os import listdir
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader

from src.vis_plot import draw_axis_ypr
from src.vis_plot import convert_euler_bbox_to_6dof
from src.vis_plot import convert_rotmat_bbox_to_6dof


def process_ori_img_anno(img_path, anno_path, phase="train", ad_base=0.1, debug_vis=False):
    img_ori = Image.open(img_path).convert('RGB')
    img_w, img_h = img_ori.size

    [bbox, rotation_matrix, euler_angles] = anno_path
    [pitch, yaw, roll] = euler_angles
    cont_labels = np.array([pitch, yaw, roll])
    
    [ori_x, ori_y, ori_w, ori_h] = bbox
    cx, cy = ori_x+ori_w/2, ori_y+ori_h/2
    # pad_len = max(ori_w, ori_h)  # hint 1: we want get a squared face bbox <<<------
    pad_len = (ori_w + ori_h) / 2.0  # hint 1: we want get a squared face bbox <<<------
    
    if phase == "train":  # for the DAD3DHeads train-set (a changed ad)
        ad = np.random.random_sample() * 0.1 + ad_base
    else:
        ad = 0.05 + ad_base  # for the DAD3DHeads val-set (an unchanged ad)
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
        # img_cv2 = np.array(img_padded)  # PIL to cv2
        # img_cv2 = img_cv2[:, :, ::-1]  # Convert RGB to BGR 
        img_cv2 = cv2.cvtColor(np.array(img_padded), cv2.COLOR_RGB2BGR)
        
        img_cv2_vis = draw_axis_ypr(img_cv2, yaw, pitch, roll, size=50)
        return img_padded, cont_labels, rotation_matrix, img_cv2_vis
        
    return img_padded, cont_labels, rotation_matrix


def process_annotated_image(im, augment_weak=False, config=None):
    
    if augment_weak:  # for the DAD3DHeads train-set
        if np.random.uniform(0, 1) < 0.05:  # Blur
            im = im.filter(ImageFilter.BLUR)
            
        pad, scale_min, scale_max = 0, 0.8, 1.25
        aug_weak = tfs.Compose([
            tfs.Pad((pad, pad), padding_mode='edge'),
            tfs.RandomResizedCrop(size=(224, 224), scale=(scale_min, scale_max), ratio=(1., 1.))
        ])
        im_weak = aug_weak(im.copy()) 
    else:  # for the DAD3DHeads train-set
        im_weak = im
        
    im_weak_tensor = tfs.ToTensor()(im_weak)
    im_weak_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(im_weak_tensor)
 
    return im_weak, im_weak_tensor


class Dataset_DAD3DHeads(Dataset):
    def __init__(self, img_files, anno_files, phase, augment_weak=False, config=None):
        self.img_files = img_files
        self.anno_files = anno_files
        self.phase = phase
        self.augment_weak = augment_weak
        self.config = config
        self.size = len(img_files)
        print('Load DAD3DHeads Dataset with phase %s, Length:%d'%(phase, self.size))

    def __getitem__(self, idx):
        idx = idx % self.size
        
        img_path = self.img_files[idx]
        anno_path = self.anno_files[idx]

        img, cont_labels, gt_rot_mat = process_ori_img_anno(img_path, anno_path, phase=self.phase)
        pitch, yaw, roll = cont_labels  # degrees, X-Y-Z
        
        im_weak, im_weak_tensor = process_annotated_image(
            img, augment_weak=self.augment_weak, config=self.config)
        
        # DAD3DHeads only provides full-range rotation_matrix, thus we do not use euler_angles
        # trans_euler = torch.FloatTensor([pitch, yaw, roll])
        Rot = torch.FloatTensor(gt_rot_mat)
        
        im_strong_tensor = torch.zeros_like(im_weak_tensor)  # DAD3DHeads does not need augment_strong
        
        if self.phase == "train":
            jpg_names = [name for name in os.listdir(self.config.log_dir) if "DAD3DHeads_train_weak_" in name]
            save_jpg_name = self.config.log_dir+"/DAD3DHeads_train_weak_"+str(idx).zfill(8)+".jpg"
            if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)
        else:
            jpg_names = [name for name in os.listdir(self.config.log_dir) if "DAD3DHeads_val_" in name]
            save_jpg_name = self.config.log_dir+"/DAD3DHeads_val_"+str(idx).zfill(8)+".jpg"
            if len(jpg_names) < 10 and idx%10 == 0: im_weak.save(save_jpg_name)
            
        # sample = dict(idx=idx, rot_mat=Rot, euler_angles=trans_euler, img=im_weak_tensor, img_strong=im_strong_tensor)
        sample = dict(idx=idx, rot_mat=Rot, img=im_weak_tensor, img_strong=im_strong_tensor)
        return sample
        
    def __len__(self):
        return self.size

#################################################################################

def get_2d_keypoints(data, img_height):
    flame_vertices3d = np.array(data["vertices"], dtype=np.float32)
    model_view_matrix = np.array(data["model_view_matrix"], dtype=np.float32)
    projection_matrix = np.array(data["projection_matrix"], dtype=np.float32)

    flame_vertices3d_homo = np.concatenate((flame_vertices3d, np.ones_like(flame_vertices3d[:, [0]])), -1)
    flame_vertices3d_world_homo = np.transpose(np.matmul(model_view_matrix, np.transpose(flame_vertices3d_homo)))

    flame_vertices2d_homo = np.transpose(
        np.matmul(projection_matrix, np.transpose(flame_vertices3d_world_homo))
    )
    flame_vertices2d = flame_vertices2d_homo[:, :2] / flame_vertices2d_homo[:, [3]]
    return np.stack((flame_vertices2d[:, 0], (img_height - flame_vertices2d[:, 1])), -1).astype(int)

def draw_annotated_mesh(predictions, image, subset, proj_path):
    if subset != "face" and subset != "head":
        ValueError("Invalid FLAME mesh vertices subset provided.\n"
                   "Available options are: face, head")

    FLAME_IDICES_DIR = os.path.join(proj_path, "model_training/model/static/flame_indices/")
    EDGE_COLOR = (218, 48, 39)
    OPACITY = .6

    mesh_vis = image.copy()
    output = image.copy()
    projected_vertices = predictions["projected_vertices"]
    edges = np.load(os.path.join(FLAME_IDICES_DIR, f"{subset}_edges.npy"))

    for edge in edges:
        pt1, pt2 = edge[0], edge[1]
        cv2.line(mesh_vis, projected_vertices[pt1], projected_vertices[pt2], EDGE_COLOR, 1, cv2.LINE_AA)

    cv2.addWeighted(mesh_vis, OPACITY, output, 1 - OPACITY, 0, output)
    return mesh_vis

######################################################################

# DAD-3DHeads-codes/model_training/model/flame.py
def limit_angle(angle, pi = 180.0):
    """
    Angle should be in degrees, not in radians.
    If you have an angle in radians - use the function radians_to_degrees.
    """
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi
    return angle

def raw_pose_labels_filter(raw_data_list, is_full_range):
    imgs_list, anno_list = [], []
    
    for raw_data in tqdm(raw_data_list):    
        [item_id, img_path, annotation_path, bbox] = raw_data
        
        anno_json_dict = json.load(open(annotation_path, "r"))
        vertices3d = np.array(anno_json_dict["vertices"], dtype=np.float32)  # (5023, 3)
        # please refer https://github.com/PinataFarms/DAD-3DHeads for vertices3d --> vertices2d (68, 2)
        model_view_matrix = np.array(anno_json_dict["model_view_matrix"], dtype=np.float32)  # (4, 4)
        projection_matrix = np.array(anno_json_dict["projection_matrix"], dtype=np.float32)  # (4, 4)
        
        # Following the DAD-3DHeads Benchmarks https://github.com/PinataFarms/DAD-3DHeads/issues/13
        # https://github.com/PinataFarms/DAD-3DHeads/blob/main/dad_3dheads_benchmark/benchmark.py#L68
        rot_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # around X-axis
        mv = rot_180 @ model_view_matrix
        rotation_matrix = mv[:3, :3]
        
        '''(pitch, yaw, roll) in degrees (rot_mat ==> euler_angles is not accurate for the gimbal lock)'''
        rot_mat_2 = np.transpose(rotation_matrix.copy())
        angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
        roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
        
        while abs(pitch) > 180: pitch = pitch - pitch/abs(pitch)*360
        while abs(yaw) > 180: yaw = yaw - yaw/abs(yaw)*360
        while abs(roll) > 180: roll = roll - roll/abs(roll)*360
        euler_angles = [pitch, yaw, roll]
        
        is_front_face = (abs(pitch) < 90) and (abs(yaw) < 90) and (abs(roll) < 90)
        if not is_full_range and is_front_face:  # front face with each euler angle in range (-90, 90)
            continue
        
        imgs_list.append(img_path)
        anno_list.append([bbox, rotation_matrix.tolist(), euler_angles])

    return imgs_list, anno_list


def get_dataloader_DAD3DHeads(phase, config):
    
    assert phase in ["train", "val"], "unsupport phase of DAD3DHeads! " + phase

    db_path = config.data_dir_DAD3DHeads
    is_full_range = config.is_full_range
    assert is_full_range == True, "We now only apply DAD3DHeads for full_range HPE."
    print("Dataset DAD3DHeads with phase:", phase, "; is_full_range:", is_full_range)
    
    if phase == 'train':
        processed_json_path = os.path.join(db_path, "train", "train_HeadPoseEstimation.json")
        
        if os.path.exists(processed_json_path):
            with open(processed_json_path, "r") as json_file:
                json_dict = json.load(json_file)
            imgs_list, anno_list = json_dict["images"], json_dict["annotations"]
        
        else:        
            raw_train_list = []
            train_json_dict_list = json.load(open(os.path.join(db_path, "train", "train.json"), "r"))
            for train_json_dict in train_json_dict_list:
                item_id = train_json_dict["item_id"]
                # img_path = train_json_dict["img_path"]
                # annotation_path = train_json_dict["annotation_path"]
                bbox = train_json_dict["bbox"]  # [x, y, w, h] format
                # keys: {"quality","gender","expression","age","occlusions","pose","standard light"}
                # attr = train_json_dict["attributes"]  
                img_path = os.path.join(db_path, "train", "images", item_id+".png")
                annotation_path = os.path.join(db_path, "train", "annotations", item_id+".json")
                raw_train_list.append([item_id, img_path, annotation_path, bbox])

            imgs_list, anno_list = raw_pose_labels_filter(raw_train_list, is_full_range)
            print("Origimal/Left heads number in DAD3DHeads with phase train:", len(raw_train_list), len(imgs_list))
            
            json_dict = {"images": imgs_list, "annotations": anno_list}
            with open(processed_json_path, "w") as json_file:
                json.dump(json_dict, json_file)
            
        batch_size = config.batch_size
        shuffle = True
        augment_weak = False
        drop_last=True

    if phase == 'val':
        processed_json_path = os.path.join(db_path, "val", "val_HeadPoseEstimation.json")
        
        if os.path.exists(processed_json_path):
            with open(processed_json_path, "r") as json_file:
                json_dict = json.load(json_file)
            imgs_list, anno_list = json_dict["images"], json_dict["annotations"]
        
        else:
            raw_val_list = []
            val_json_dict_list = json.load(open(os.path.join(db_path, "val", "val.json"), "r"))
            for val_json_dict in val_json_dict_list:
                item_id = val_json_dict["item_id"]
                # img_path = val_json_dict["img_path"]
                # annotation_path = val_json_dict["annotation_path"]
                bbox = val_json_dict["bbox"]  # [x, y, w, h] format
                img_path = os.path.join(db_path, "val", "images", item_id+".png")
                annotation_path = os.path.join(db_path, "val", "annotations", item_id+".json")
                raw_val_list.append([item_id, img_path, annotation_path, bbox])
            
            imgs_list, anno_list = raw_pose_labels_filter(raw_val_list, is_full_range)
            print("Origimal/Left heads number in DAD3DHeads with phase val:", len(raw_val_list), len(imgs_list))

            json_dict = {"images": imgs_list, "annotations": anno_list}
            with open(processed_json_path, "w") as json_file:
                json.dump(json_dict, json_file)

        batch_size = config.batch_size
        shuffle = False
        augment_weak = False
        drop_last = False


    dset = Dataset_DAD3DHeads(imgs_list, anno_list, phase, augment_weak, config)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_workers,
        pin_memory=True, drop_last=drop_last)
 
    return dloader

if __name__ == '__main__':
    print("testing DAD3DHeads Dataset...")
    
    """Please put this file under the main folder for running test."""
    
    is_full_range = True
    db_path = "/datasdc/zhouhuayi/dataset/DAD-3DHeadsDataset"
    proj_path = "/datasdc/zhouhuayi/face_related/DAD-3DHeads-codes/"

    from src.renderer import Renderer
    renderer = Renderer(
        vertices_path="pose_references/vertices_trans.npy", 
        triangles_path="pose_references/triangles.npy"
    )
    
    if os.path.exists("./debug_DAD3DHeads"):
        shutil.rmtree("./debug_DAD3DHeads")
    os.mkdir("./debug_DAD3DHeads")
    
    for phase in ["val", "train"]:
        raw_val_list = []
        val_json_dict_list = json.load(open(os.path.join(db_path, phase, f"{phase}.json"), "r"))
        for img_id, val_json_dict in enumerate(val_json_dict_list):
            item_id = val_json_dict["item_id"]
            bbox = val_json_dict["bbox"]  # [x, y, w, h] format
            img_path = os.path.join(db_path, phase, "images", item_id+".png")
            annotation_path = os.path.join(db_path, phase, "annotations", item_id+".json")
            raw_val_list.append([item_id, img_path, annotation_path, bbox])
            if img_id > 200: break
        
        '''By visualization, we can find that full-range euler_angles produced by rot_euler_6DRepNet() is not reliable'''
        imgs_list, anno_list = raw_pose_labels_filter(raw_val_list, is_full_range)
        print("Origimal/Left faces/heads number in DAD3DHeads with phase val:", len(raw_val_list), len(imgs_list))
       
        for idx, (img_path, annos) in enumerate(zip(imgs_list, anno_list)):
        
            [bbox, rotation_matrix, euler_angles] = annos
            
            """Case 1 img_mesh: original image with head mesh"""
            img_ori = cv2.imread(img_path)
            json_path = img_path.replace('images', 'annotations').replace('.png', '.json')
            with open(json_path) as json_data:
                mesh_data = json.load(json_data)
            keypoints_2d = get_2d_keypoints(mesh_data, img_ori.shape[0])
            # img_mesh = draw_annotated_mesh({"projected_vertices": keypoints_2d}, img_ori, "face", proj_path)
            img_mesh = draw_annotated_mesh({"projected_vertices": keypoints_2d}, img_ori, "head", proj_path)
            
            """Case 2 img_3dvis: original image with 3d vis by img2pose"""
            (h, w, c) = img_ori.shape
            global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
            [x0, y0, bw, bh] = bbox  # xywh
            bbox = [x0-0.1*bw, y0-0.1*bh, x0+1.1*bw, y0+1.1*bh]  # xywh --> xyxy
            global_pose = convert_rotmat_bbox_to_6dof(rotation_matrix, bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(img_ori, [global_pose])
            img_3dvis = renderer.render(img_ori, trans_vertices, alpha=1.0)
            
            """Case 3 img_vis: cropped image with plottd three euler angles"""
            img_pil_input, cont_labels, rotation_matrix, img_vis = process_ori_img_anno(img_path, annos, debug_vis=True)
            img_name = os.path.split(img_path)[-1]
            
            """Case 4 img_3dvis_v2: cropped image with 3d vis by img2pose"""
            (h, w, c) = img_vis.shape  # (224, 224, 3)
            global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
            scaled_bbox = [20, 20, w-20, h-20]
            global_pose = convert_euler_bbox_to_6dof(euler_angles, scaled_bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(img_vis, [global_pose])
            img_3dvis_v2 = renderer.render(img_vis, trans_vertices, alpha=1.0)
            
            """Case 5 img_pil_rot: rotate the plottd image with a random rot_angle"""
            img_pil = img_vis[:, :, ::-1]  # Convert BGR to RGB 
            img_pil = Image.fromarray(img_pil)
            rot_angle = round(np.random.rand()*60 - 30, 3)  # (0, 1)*60 - 30 --> (-30, 30)
            img_pil_rot = img_pil.rotate(rot_angle, center=(112, 112), expand=True)
            im_rot_w, im_rot_h = img_pil_rot.size  # must big than (224,224)
            new_x_min, new_y_min = im_rot_w//2 - 112, im_rot_h//2 - 112
            new_x_max, new_y_max = new_x_min + 224, new_y_min + 224
            img_pil_rot = img_pil_rot.crop((new_x_min, new_y_min, new_x_max, new_y_max))
            img_cv2_rot = cv2.cvtColor(np.array(img_pil_rot), cv2.COLOR_RGB2BGR)
            
            """Case 6 img_rot_3dvis: rotated image with the 3d vis by rotating rotation_matrix"""
            ra = rot_angle * np.pi / 180.0
            # https://www.cnblogs.com/meteoric_cry/p/7987548.html
            rot_mat = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])  # around Z-axis
            rotation_matrix_new = rot_mat @ rotation_matrix
            (h, w, c) = img_cv2_rot.shape
            global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
            scaled_bbox = [20, 20, w-20, h-20]
            global_pose = convert_rotmat_bbox_to_6dof(rotation_matrix_new, scaled_bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(img_cv2_rot.copy(), [global_pose])
            img_rot_3dvis = renderer.render(img_cv2_rot.copy(), trans_vertices, alpha=1.0)
            
            if idx < 50:
                img_cv2_input = cv2.cvtColor(np.array(img_pil_input), cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_origin_{img_name}", img_cv2_input)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_mesh_{img_name}", img_mesh)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_3dRotMat_{img_name}", img_3dvis)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_euler_{img_name}", img_vis)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_3dEuler_{img_name}", img_3dvis_v2)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_randomRot_{img_name}", img_cv2_rot)
                cv2.imwrite(f"./debug_DAD3DHeads/{phase}_{str(idx).zfill(4)}_randomRot3d_{img_name}", img_rot_3dvis)
            else:
                break
            
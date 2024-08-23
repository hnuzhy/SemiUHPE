
import os
import cv2
import json
import math
import time
import torch
import shutil
import imageio
import torchvision

import numpy as np
import torch.nn as nn
import torchvision.transforms as tfs

from scipy.spatial.transform import Rotation
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from src.config import get_config
from src.agent import get_agent
from src.fisher.fisher_utils import batch_torch_A_to_R
from src.utils import compute_euler_angles_from_rotation_matrices
from src.utils import limit_angle
from src.vis_plot import convert_euler_bbox_to_6dof
from src.vis_plot import convert_rotmat_bbox_to_6dof

from src.renderer import Renderer
renderer = Renderer(
    vertices_path="pose_references/vertices_trans.npy", 
    triangles_path="pose_references/triangles.npy"
)

###############################################################################
# BPJDet utils (the body-head model trained on CrowdHuman and using YOLOv5-L)
###############################################################################

from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

colors_list = [
        # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
        # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

def cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    # calculate small rectangle inside big box ratio, calSmallBoxInsideRatio
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        # return crossArea/(areaA + areaB - crossArea)
        return crossArea/areaB  # range [0, 1]

def post_process_batch(imgs, paths, shapes, body_dets, part_dets, num_offsets, match_iou_thres):

    batch_bboxes, batch_points, batch_scores = [], [], []

    # process each image in batch
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]
        
        if nbody:  # one batch
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]

            scores = bdet[:, 4].cpu().numpy()  # body detection score
            bboxes = scale_coords(imgs[si].shape[1:], bdet[:, :4], shape).round().cpu().numpy()

            points = scale_coords(imgs[si].shape[1:], bdet[:, -num_offsets:], shape).cpu().numpy()
            points = points.reshape((nbody, -1, 2))
            # points = np.concatenate((points, np.zeros((nbody, points.shape[1], 1))), axis=-1)  # n*c*2 --> n*c*3
            points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5))), axis=-1)  # n*c*2 --> n*c*7
                
            # batch_parts_dict[str(img_id)] = []
            if npart:
                pdet[:, :4] = scale_coords(imgs[si].shape[1:], pdet[:, :4].clone(), shape)
                pdet_slim = pdet[:, :6].cpu().numpy()

                left_pdet = []
                matched_part_ids = [-1 for i in range(points.shape[0])]  # points shape is n*c*7, add in 2022-12-09
                for id, (x1, y1, x2, y2, conf, cls) in enumerate(pdet_slim):
                    p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))  # the body-part's part bbox center point
                    part_pts = points[:, int(cls - 1)]
                    dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                    pt_match = np.argmin(dist)

                    '''association alg version 4.0'''   
                    tmp_iou = cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])  # add in 2022-12-11, body-part must inside the body
                    if conf > part_pts[pt_match][2] and tmp_iou > match_iou_thres:  # add in 2022-12-09, we fetch the part bbox with highest conf
                        part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        matched_part_ids[pt_match] = id  # only for dataset BodyHands without left/right label of hands

            batch_bboxes.extend(bboxes)
            batch_points.extend(points)
            batch_scores.extend(scores)

        # else:
            # print("This image has no object detected!")
        
    return batch_bboxes, batch_points, batch_scores

###############################################################################

if __name__ == '__main__':
    
    alpha_value = 1.0
    edges_scale = -0.05
    
    config = get_config('test')
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)

    """Step 0: get single image / multiple images"""
    # folder_name, imgs_name = "test_DetHPE", "SCUTHead"  # classrooms, meeting rooms
    # folder_name, imgs_name = "test_DetHPE", "CroHD"  # crossing, stations
    folder_name, imgs_name = "test_DetHPE", "CrowdHuman"  # crowded persons
    image_path = f"./{folder_name}/{imgs_name}"

    """Step 1: load model weights"""
    weights_path = "weights/ch_head_l_1536_e150_best_mMR.pt"
    conf_thres = 0.5
    iou_thres = 0.75
    match_iou_thres = 0.6
    scales = [1]
    imgsz = 1536
    num_offsets = 2
    
    body_bbox, head_bbox = False, True
    
    device = 'cuda:0'
    print('Using device: {}'.format(device))
    BPJDet_model = attempt_load(weights_path, map_location=device)  # load FP32 model
    stride = int(BPJDet_model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    """Step 2: processing images"""
    print(image_path, len(dataset))
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        if '_res' in single_path or '_vis' in single_path:
            continue
        
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = BPJDet_model(img, augment=True, scales=scales)[0]
        body_dets = non_max_suppression(out_ori, conf_thres, iou_thres, 
            classes=[0], num_offsets=num_offsets)
        part_dets = non_max_suppression(out_ori, conf_thres, iou_thres, 
            classes=list(range(1, 1 + num_offsets//2)), num_offsets=num_offsets)

        # Post-processing of body and part detections
        bboxes, points, scores = post_process_batch(
            img, [], [[im0.shape[:2]]], body_dets, part_dets, num_offsets, match_iou_thres)
        
        line_thick = max(im0.shape[:2]) // 1000 + 3
        
        im0_ori = im0.copy()
        img_h, img_w, img_c = im0_ori.shape
        
        head_bbox_list = []
        for ind, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
            if f_score != 0:  # for the body-head pair, we must have a detected head
                color = colors_list[ind%len(colors_list)]
                head_bbox_list.append(f_bbox)
                if body_bbox:
                    [x1, y1, x2, y2] = bbox
                    cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=line_thick)
                if head_bbox:
                    [px1, py1, px2, py2] = f_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=line_thick)

        for [px1, py1, px2, py2] in head_bbox_list:
            pcx, pcy = (px1 + px2)/2.0, (py1 + py2)/2.0
            head_size = max(px2 - px1, py2 - py1)
            new_px1 = max(0, int(pcx - (0.5 - edges_scale) * head_size))
            new_px2 = min(img_w-1, int(pcx + (0.5 - edges_scale) * head_size))
            new_py1 = max(0, int(pcy - (0.5 - edges_scale) * head_size))
            new_py2 = min(img_h-1, int(pcy + (0.5 - edges_scale) * head_size))

            img_head_cv2 = im0_ori[new_py1:new_py2, new_px1:new_px2]
            img_head_pil = img_head_cv2[:, :, ::-1]  # Convert BGR to RGB 
            img_head_pil = Image.fromarray(img_head_pil)
            img_input = img_head_pil.resize([224, 224])
            
            img_tensor = tfs.ToTensor()(img_input)
            img_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
            with torch.no_grad(): 
                agent.net.eval()
                fisher_out = agent.net(img_tensor.reshape((1,3,224,224)).cuda())
                pd_m = batch_torch_A_to_R(fisher_out)

            rot_mat = pd_m.detach().cpu().numpy()[0]
            rot_mat_2 = np.transpose(rot_mat)
            angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
            roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))

            head_bbox = [new_px1, new_py1, new_px2, new_py2]
            global_intrinsics = np.array([[img_w+img_h, 0, img_w // 2], [0, img_w+img_h, img_h // 2], [0, 0, 1]])
            global_pose = convert_rotmat_bbox_to_6dof(rot_mat, head_bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(im0, [global_pose])
            im0 = renderer.render(im0, trans_vertices, alpha=alpha_value)
        
        cv2.putText(im0, '{:2d} Heads'.format(len(head_bbox_list)), (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
      
        cv2.imwrite(single_path[:-4]+"_res_BPJDetSemiUHPE.jpg", im0)


'''
python image.py SSL1.0_r0.05_ce_tDyna0.75_b16_ema_RO_CO_CM_full/Sep20_195132/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1

python image.py SSL1.0_r0.05_ce_tDyna0.75_b32_ema_RO_CO_CM_full/Sep30_130637/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
'''
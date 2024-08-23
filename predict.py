
import os
import cv2
import json
import torch
import shutil
import numpy as np

import torchvision.transforms as tfs
from scipy.spatial.transform import Rotation
from PIL import Image

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

if __name__ == '__main__':

    alpha_value = 1.0
    edges_scale = 0.05
    # subset_name = "COCOHead"
    # subset_name = "Comparing"
    # subset_name = "WiderFace"
    # subset_name = "CrowdHuman"
    # subset_name = "DAD3DNetFailed"
    subset_name = "DAD3DHeads"
    
    test_imgs_dir = f"./test_imgs/{subset_name}/"  # well-cropped head images
    save_path_dir = test_imgs_dir.replace(subset_name, f"{subset_name}_results")
    if os.path.exists(save_path_dir):
        shutil.rmtree(save_path_dir)
    os.mkdir(save_path_dir)

    config = get_config('test')
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)
    
    img_names = os.listdir(test_imgs_dir)
    for index, img_name in enumerate(img_names):
        img_path = os.path.join(test_imgs_dir, img_name)

        img_ori = Image.open(img_path).convert('RGB')
        
        w, h = img_ori.size
        bbox = [int(edges_scale*w), int(edges_scale*h), int((1-edges_scale)*w), int((1-edges_scale)*h)]
        global_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
        img_ori_cv2 = cv2.cvtColor(np.array(img_ori), cv2.COLOR_RGB2BGR)
        
        img_input = img_ori.resize([224, 224])
        
        
        if "DAD3D" in subset_name:
            item_id = img_name[:-4]
            db_path = "/datasdc/zhouhuayi/dataset/DAD-3DHeadsDataset"
            test_json_dict_list = json.load(open(os.path.join(db_path, "test", "test.json"), "r"))
            for test_json_dict in test_json_dict_list:
                if item_id == test_json_dict["item_id"]:
                    ori_bbox = test_json_dict["bbox"]  # [x, y, w, h] format
                    print(w, h, ori_bbox)
                    break
                    
            from eval_DAD3DHeads import process_ori_img
            img_input = process_ori_img(img_path, ori_bbox)
            

        img_tensor = tfs.ToTensor()(img_input)
        img_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        with torch.no_grad(): 
            agent.net.eval()
            fisher_out = agent.net(img_tensor.reshape((1,3,224,224)).cuda())
            pd_m = batch_torch_A_to_R(fisher_out)

        if config.train_labeled == "DAD3DHeads":  # trained on DAD3DHeads for full-range HPE
            rot_mat = pd_m.detach().cpu().numpy()[0]
            rot_mat_2 = np.transpose(rot_mat)
            angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
            roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))

            global_pose = convert_rotmat_bbox_to_6dof(rot_mat, bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(img_ori_cv2, [global_pose])
            img_3dvis = renderer.render(img_ori_cv2, trans_vertices, alpha=alpha_value)

        if config.train_labeled == "300WLP":  # trained on 300WLP for front-range HPE
            pd_e = compute_euler_angles_from_rotation_matrices(pd_m, full_range=False)*180/np.pi
            [pitch, yaw, roll] = pd_e.detach().cpu().numpy()[0]
            euler_angles = [pitch, yaw, roll]
            
            global_pose = convert_euler_bbox_to_6dof(euler_angles, bbox, global_intrinsics)
            trans_vertices = renderer.transform_vertices(img_ori_cv2, [global_pose])
            img_3dvis = renderer.render(img_ori_cv2, trans_vertices, alpha=alpha_value)
        
        new_img_name = img_name[:-4]+"_"+config.train_labeled+".jpg"
        save_img_path = os.path.join(save_path_dir, new_img_name)
        cv2.imwrite(save_img_path, img_3dvis)
        
        print(index, "\t", img_path, "\t", pitch, yaw, roll)
        # if index > 30: break

'''
python predict.py SSL1.0_r0.05_ce_tDyna0.75_b16_ema_RO_CO_CM_full/Sep20_195132/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1
    
python predict.py SSL1.0_r0.05_ce_tDyna0.75_b32_ema_RO_CO_CM_full/Sep30_130637/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
    
python predict.py SSL1.0_r0.05_ce_effinetv2_tDyna0.75_b32_ema_RO_CO_CM_full/Jul18_100557/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network effinetv2 --gpu_ids 1

python predict.py SSL1.0_r0.05_ce_effinetv2_tDyna0.75_b64_ema_RO_CO_CM_full/Jul22_122502/best \
    --is_full_range --config settings/DAD3DHeads_WildHead.yml --network effinetv2 --gpu_ids 1

'''
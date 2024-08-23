import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from os.path import dirname, abspath, join
from pytorch3d import transforms as trans

from src.config import get_config
from src.agent import get_agent
from src.datasets.dataset_AFLW2000 import get_dataloader_AFLW2000
from src.datasets.dataset_BIWItest import get_dataloader_BIWItest
from src.datasets.dataset_DAD3DHeads import get_dataloader_DAD3DHeads

from src.utils import compute_euler_angles_from_rotation_matrices
from src.utils import limit_angle

def test():
    # create experiment config containing all hyperparameters
    config = get_config('test')

    # create dataloader
    if config.test_set == "AFLW2000":
        test_loader = get_dataloader_AFLW2000('test', config)
    if config.test_set == "BIWItest":
        test_loader = get_dataloader_BIWItest('test', config)
    if config.test_set == "DAD3DHeads":
        test_loader = get_dataloader_DAD3DHeads('val', config)
        
    # create network and eval agent
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)

    evaluate(config, test_loader, agent)
    evaluate(config, test_loader, agent, eval_ema=True)


def evaluate(config, test_loader, agent, eval_ema=False):
    ema_name = 'EMA_' if eval_ema else ''

    # err_deg_lst = []
    err_deg_list_pitch, err_deg_list_yaw, err_deg_list_roll = [], [], []
    err_rot_list, err_relative_deg_list = [], []
    
    gt_deg_list_yaw = []
    
    testbar = tqdm(test_loader)
    for i, data in enumerate(testbar):
        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data, eval_ema=eval_ema)
        # err_deg_lst.append(fisher_dict['err_deg'].detach().cpu().numpy())
        
        pd_m = fisher_dict['pred_orth']
        gt_m = data.get('rot_mat').cuda()
        
        if 'euler_angles' in data:
            if config.train_labeled == "DAD3DHeads":  # trained on DAD3DHeads, and test on AFLW2000 / BIWItest
                # gt_e = data.get('euler_angles').cuda()
                # rot_180 = torch.FloatTensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).unsqueeze(0)  # (1, 3, 3)
                # rot_mat = rot_180.repeat(pd_m.shape[0], 1, 1).cuda() @ pd_m  # (bs, 3 ,3)
                # pd_e = compute_euler_angles_from_rotation_matrices(rot_mat, full_range=False)*180/np.pi
                # pd_e = -pd_e
                # err_deg = abs((gt_e - pd_e).detach().cpu().numpy())
                
                gt_e = data.get('euler_angles')
                rot_mats = pd_m.detach().cpu().numpy()
                temp_err_deg_list = []
                for rot_mat in rot_mats:
                    rot_mat_2 = np.transpose(rot_mat)
                    angle = Rotation.from_matrix(rot_mat_2).as_euler("xyz", degrees=True)
                    roll, pitch, yaw = list(map(limit_angle, [angle[2], angle[0] - 180, angle[1]]))
                    temp_err_deg_list.append([pitch, yaw, roll])
                err_deg = abs(gt_e - np.array(temp_err_deg_list))
                    
            else:  # trained on 300WLP, and test on AFLW2000 / BIWItest
                gt_e = data.get('euler_angles')
                pd_e = compute_euler_angles_from_rotation_matrices(pd_m, full_range=False)*180/np.pi
                err_deg = abs((gt_e.cuda() - pd_e).detach().cpu().numpy())
            
            err_deg_list_pitch.append(err_deg[:, 0])
            err_deg_list_yaw.append(err_deg[:, 1])
            err_deg_list_roll.append(err_deg[:, 2])
            
            gt_deg_list_yaw.append(gt_e[:, 1])
            
        else: # trained on DAD3DHeads, and test on DAD3DHeads
            err_rad = trans.so3_relative_angle(pd_m, gt_m)
            err_deg = torch.rad2deg(err_rad).detach().cpu().numpy()  # (bs,)
            err_relative_deg_list.append(err_deg)
            
            # https://github.com/PinataFarms/DAD-3DHeads/blob/main/dad_3dheads_benchmark/benchmark.py#L74
            R_dist = (pd_m @ gt_m.transpose(1, 2)).detach().cpu().numpy()  # (bs, 3, 3)
            temp_err_rot_list = []
            for idx in range(R_dist.shape[0]):
                err_rot = np.linalg.norm(np.eye(3) - R_dist[idx], "fro")
                temp_err_rot_list.append(err_rot)
            err_rot_list.append(temp_err_rot_list)  # (bs,)
            
            gt_e = compute_euler_angles_from_rotation_matrices(gt_m, full_range=True)*180/np.pi
            gt_deg_list_yaw.append(gt_e.detach().cpu().numpy()[:, 1])
            
    # err_deg_lst = np.concatenate(err_deg_lst, 0)

    print(f'==== {ema_name}exp: {config.exp_name} ====')
    # print(f'{ema_name}Fisher mean: {np.mean(err_deg_lst):.4f}')

    if len(err_rot_list) != 0:  # only for full-range rotation_matrix based datasets like DAD-3DHeads
        err_relative_deg_list = np.concatenate(err_relative_deg_list, 0)
        err_rot_list = np.concatenate(err_rot_list, 0)
        e_deg = np.mean(err_relative_deg_list)
        e_rot = np.mean(err_rot_list)
        print(f'{ema_name}Relative Angle mean: {e_deg:.4f}; {ema_name}Rotation Matrix mean: {e_rot:.4f}')
        
        if not eval_ema:  # for Performance Improvement Analysis
            final_detailed_errors_list = []
            gt_deg_list_yaw = np.concatenate(gt_deg_list_yaw, 0)
            for err_deg, err_rot, gt_y in zip(err_relative_deg_list, err_rot_list, gt_deg_list_yaw):
                final_detailed_errors_list.append([float(gt_y), float(err_deg), float(err_rot)])
            save_json_name = f"./ERR_{config.exp_name}_{config.network}_{config.test_set}.json"
            with open(save_json_name, "w") as json_file:
                json.dump(final_detailed_errors_list, json_file)
        
    else:
        err_deg_list_pitch = np.concatenate(err_deg_list_pitch, 0)
        err_deg_list_yaw = np.concatenate(err_deg_list_yaw, 0)
        err_deg_list_roll = np.concatenate(err_deg_list_roll, 0)
        err_deg_list_all = (err_deg_list_pitch + err_deg_list_yaw + err_deg_list_roll) / 3
        p_e = np.mean(err_deg_list_pitch)
        y_e = np.mean(err_deg_list_yaw)
        r_e = np.mean(err_deg_list_roll)
        m_e = np.mean(err_deg_list_all)
        print(f'{ema_name}Euler MAE (pitch, yaw, roll, mean): {p_e:.4f}, {y_e:.4f}, {r_e:.4f}, {m_e:.4f}')
        
        if not eval_ema:  # for Performance Improvement Analysis
            final_detailed_errors_list = []
            gt_deg_list_yaw = np.concatenate(gt_deg_list_yaw, 0)
            for gt_y, pd_p, pd_y, pd_r in zip(gt_deg_list_yaw, err_deg_list_pitch, err_deg_list_yaw, err_deg_list_roll):
                final_detailed_errors_list.append([float(gt_y), float(pd_p), float(pd_y), float(pd_r)])
            save_json_name = f"./exps/MAE_{config.exp_name}_{config.network}_{config.test_set}.json"
            with open(save_json_name, "w") as json_file:
                json.dump(final_detailed_errors_list, json_file)

        
if __name__ == '__main__':
    with torch.no_grad():
        test()


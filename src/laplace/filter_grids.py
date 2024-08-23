
import os
import math
import numpy as np

# from scipy.spatial.transform import Rotation as R

grids = np.load("./eq_grids3.npy")  # (36864, 3, 3)
eps = 1e-7

# R.from_matrix([[1,0,0], [0,1,0], [0,0,1]]).as_euler('xyz', degrees=True) --> array([0., 0., 0.])
# basis_front = np.array([[1,0,0], [0,1,0], [0,0,1]])  # normal sampling

# R.from_matrix([[1,0,0], [0,-1,0], [0,0,-1]]).as_euler('xyz', degrees=True) --> array([180., 0., 0.]  
basis_front = np.array([[1,0,0], [0,-1,0], [0,0,-1]])  # used sampling

basis_rot90 = np.array([[0,0,0], [0,1,0], [0,0,0]])  # theta = np.pi / 2
basis_rot180 = np.array([[-1,0,0], [0,1,0], [0,0,-1]])  # theta = np.pi

theta_list = []
for grid in grids:
    # m_c = grid @ grid.T
    # m_det = m_c[0,0] + m_c[1,1] + m_c[2,2]  # m_det = 3
    m = basis_front @ grid.T 
    cos_v = (m[0,0] + m[1,1] + m[2,2] - 1)/2
    theta = math.acos(np.clip(cos_v, -1+eps, 1-eps))  # [0, np.pi]
    theta_list.append(theta)
print(np.min(theta_list), np.max(theta_list), np.mean(theta_list), np.median(theta_list))

print(theta_list)

# filter_thre_theta = np.pi / 2  # we only keep front grids
# save_grids_name = "./eq_grids3_front.npy"  # 6656 grids (18% grids)

filter_thre_theta = (100 / 180) * np.pi  # we keep large-view grids
save_grids_name = "./eq_grids3_large.npy"  # 8872 grids (24% grids)

# filter_thre_theta = np.median(theta_list)  # we remove out half grids
# save_grids_name = "./eq_grids3_half.npy"  # 18432 grids (50% grids)

front_grid_count = 0
final_left_grids = []
for theta, grid in zip(theta_list, grids):
    if theta < filter_thre_theta:
        front_grid_count += 1
        final_left_grids.append(grid)
print(len(grids), front_grid_count, front_grid_count/len(grids))

np.save(save_grids_name, np.array(final_left_grids))
import os
import cv2
import json
import logging
import shutil
import csv
import torch
import math
import numpy as np

class TrainClock(object):
    """ Clock object to track epoch and iteration during training
    """

    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.iteration = 0
        # used for ema
        self.scratch_iter = 0

    def tick(self):
        self.minibatch += 1
        self.iteration += 1
        # used for ema
        self.scratch_iter += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'iteration': self.iteration
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.iteration = clock_dict['iteration']


class KSchedule(object):
    """ linear interpolation of k
    """

    def __init__(self, k_init, k_safe, max_iters):
        self.k_init = k_init
        self.k_safe = k_safe
        self.max_iters = max_iters

    def get_k(self, cur_iter):
        ratio = min(cur_iter // (self.max_iters // 10), 9) / 9
        k = self.k_init + ratio * (self.k_safe - self.k_init)
        return k


class Table(object):
    def __init__(self, filename):
        '''
        create a table to record experiment results that can be opened by excel
        :param filename: using '.csv' as postfix
        '''
        assert '.csv' in filename
        self.filename = filename

    @staticmethod
    def merge_headers(header1, header2):
        # return list(set(header1 + header2))
        if len(header1) > len(header2):
            return header1
        else:
            return header2

    def write(self, ordered_dict):
        '''
        write an entry
        :param ordered_dict: something like {'name':'exp1', 'acc':90.5, 'epoch':50}
        :return:
        '''
        if os.path.exists(self.filename) == False:
            headers = list(ordered_dict.keys())
            prev_rec = None
        else:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                prev_rec = [row for row in reader]
            headers = self.merge_headers(headers, list(ordered_dict.keys()))

        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, headers)
            writer.writeheader()
            if not prev_rec == None:
                writer.writerows(prev_rec)
            writer.writerow(ordered_dict)


class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'params.json')

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'\nMaking directory: {path}...\n')


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def requires_grad(xs, req=False):
    if not (isinstance(xs, tuple) or isinstance(xs, list)):
        xs = tuple(xs)
    for x in xs:
        x.requires_grad_(req)


def dict_get(dict, key, default, default_device='cuda'):
    v = dict.get(key)
    default_tensor = torch.tensor([default]).float().to(default_device)
    if v is None or v.nelement() == 0:
        return default_tensor
    else:
        return v


def acc(x, thres):
    return (x <= thres).sum() / len(x)



'''6DRepNet utils'''

def get_6DRepNet_Rot(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


#input batch*4*4 or batch*3*3
#output torch batch*3 x, y, z in radiant
#the rotation is in the sequence of x,y,z
# https://learnopencv.com/rotation-matrix-to-euler-angles/
def compute_euler_angles_from_rotation_matrices(rotation_matrices, full_range=False, use_gpu=True, gpu_id=0):
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:,0,0]*R[:,0,0]+R[:,1,0]*R[:,1,0])
    singular = sy<1e-6
    singular = singular.float()
    
    '''2023.01.15'''
    for i in range(len(sy)):  # expand y (yaw angle) range into (-180, 180)
        if R[i,0,0] < 0 and full_range:
            sy[i] = -sy[i]
    
    x=torch.atan2(R[:,2,1], R[:,2,2])
    y=torch.atan2(-R[:,2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    z=torch.atan2(R[:,1,0],R[:,0,0])
    
    xs=torch.atan2(-R[:,1,2], R[:,1,1])
    ys=torch.atan2(-R[:,2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    zs=R[:,1,0]*0
        
    if use_gpu:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3).cuda(gpu_id))
    else:
        out_euler=torch.autograd.Variable(torch.zeros(batch,3))  
    out_euler[:,0]=x*(1-singular)+xs*singular
    out_euler[:,1]=y*(1-singular)+ys*singular
    out_euler[:,2]=z*(1-singular)+zs*singular
        
    return out_euler


def rot_euler_6DRepNet(rotation_matrices, full_range=False):
    R = rotation_matrices
    sy = np.sqrt(R[0,0]*R[0,0]+R[1,0]*R[1,0])
    singular = sy<1e-6

    if R[0,0] < 0 and full_range:  # expand y (yaw angle) range into (-180, 180)
        sy = -sy
    
    x=math.atan2(R[2,1], R[2,2])
    y=math.atan2(-R[2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    z=math.atan2(R[1,0], R[0,0])
    
    xs=math.atan2(-R[1,2], R[1,1])
    ys=math.atan2(-R[2,0], sy)  # sy > 0, so y (yaw angle) is always in range (-90, 90)
    zs=R[1,0]*0
        
    out_euler=np.zeros(3,)  
    out_euler[0]=x*(1-singular)+xs*singular
    out_euler[1]=y*(1-singular)+ys*singular
    out_euler[2]=z*(1-singular)+zs*singular
        
    return out_euler
    


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

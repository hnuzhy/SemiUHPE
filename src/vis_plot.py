import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def bbox_is_dict(bbox):
    # check if the bbox is a not dict and convert it if needed
    if not isinstance(bbox, dict):
        temp_bbox = {}
        temp_bbox["left"] = bbox[0]
        temp_bbox["top"] = bbox[1]
        temp_bbox["right"] = bbox[2]
        temp_bbox["bottom"] = bbox[3]
        bbox = temp_bbox

    return bbox

def get_bbox_intrinsics(image_intrinsics, bbox):
    # crop principle point of view
    bbox_center_x = bbox["left"] + ((bbox["right"] - bbox["left"]) // 2)
    bbox_center_y = bbox["top"] + ((bbox["bottom"] - bbox["top"]) // 2)

    # create a camera intrinsics from the bbox center
    bbox_intrinsics = image_intrinsics.copy()
    bbox_intrinsics[0, 2] = bbox_center_x
    bbox_intrinsics[1, 2] = bbox_center_y

    return bbox_intrinsics

def pose_bbox_to_full_image(pose, image_intrinsics, bbox):
    # check if bbox is np or dict
    bbox = bbox_is_dict(bbox)

    # rotation vector
    rvec = pose[:3].copy()

    # translation and scale vector
    tvec = pose[3:].copy()

    # get camera intrinsics using bbox
    bbox_intrinsics = get_bbox_intrinsics(image_intrinsics, bbox)

    # focal length
    focal_length = image_intrinsics[0, 0]

    # bbox_size
    bbox_width = bbox["right"] - bbox["left"]
    bbox_height = bbox["bottom"] - bbox["top"]
    bbox_size = bbox_width + bbox_height
    bbox_size *= 0.5 * 0.5

    # adjust scale
    tvec[2] *= focal_length / bbox_size

    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(tvec.T)

    # reverse the projected points using the full image camera intrinsics
    tvec = projected_point.dot(np.linalg.inv(image_intrinsics.T))

    # same for rotation
    rmat = Rotation.from_rotvec(rvec).as_matrix()
    # project crop points using the crop camera intrinsics
    projected_point = bbox_intrinsics.dot(rmat)
    # reverse the projected points using the full image camera intrinsics
    rmat = np.linalg.inv(image_intrinsics).dot(projected_point)
    rvec = Rotation.from_matrix(rmat).as_rotvec()

    return np.concatenate([rvec, tvec])


"""Suitable to the popular HPE datasets like 300WLP, AFLW2000, BIWI and our newly bulit DAD3DHeads"""
def convert_euler_bbox_to_6dof(euler_angle, bbox, global_intrinsics):
    [pitch, yaw, roll] = euler_angle
    ideal_angle = [pitch, -yaw, -roll]
    
    rot_mat = Rotation.from_euler('xyz', ideal_angle, degrees=True).as_matrix()
    rot_mat = np.transpose(rot_mat)
    rotvec = Rotation.from_matrix(rot_mat).as_rotvec()
    
    local_pose = np.array([rotvec[0], rotvec[1], rotvec[2], 0, 0, 1])
    
    global_pose_6dof = pose_bbox_to_full_image(local_pose, global_intrinsics, bbox_is_dict(bbox))

    return global_pose_6dof.tolist()


"""Only suitable to the gt_rot_mat in DAD3DHeads dataset. rot_mat is much better than euler_angle"""
def convert_rotmat_bbox_to_6dof(rot_mat, bbox, global_intrinsics):
    
    rot_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rot_mat = rot_180 @ rot_mat
    
    rot_mat = np.transpose(rot_mat)
    rotvec = Rotation.from_matrix(rot_mat).as_rotvec()
    
    # local_pose = np.array([rotvec[0], rotvec[1], rotvec[2], 0, 0, 1])
    local_pose = np.array([-rotvec[0], rotvec[1], rotvec[2], 0, 0, 1])
    
    global_pose_6dof = pose_bbox_to_full_image(local_pose, global_intrinsics, bbox_is_dict(bbox))

    return global_pose_6dof.tolist()


def draw_axis_ypr(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    # Referenced from HopeNet https://github.com/natanielruiz/deep-head-pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),2)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)
    return img
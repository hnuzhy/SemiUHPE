
import os
import csv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import cv2
import shutil
head_imgs_path = "./head_images_wild_30"
if os.path.exists(head_imgs_path):
    shutil.rmtree(head_imgs_path)
os.mkdir(head_imgs_path)
scale_ratio = 1.2
head_imgs_cnt = 0
name_prefix = "OpenImagesV6_"


min_head_size_thre = 30
csv_file_list = ["validation/labels/detections.csv", "test/labels/detections.csv", "train/labels/detections.csv"]
img_file_path_list = ["validation/data/", "test/data/", "train/data/"]

target_label = "/m/04hgtk"  # /m/04hgtk,Human head in file $tag$/metadata/classes.csv/

imgsizes = []
for (imgs_root, anno_path) in zip(img_file_path_list, csv_file_list):

    filelines = csv.reader(open(anno_path, "r"))
    titles = next(filelines)
    print(imgs_root, "\t", titles)
    
    total_lines, total_cnt = 0, 0
    while True:
        try:
            fileline = next(filelines)
            ImageID, Source, LabelName, Confidence = fileline[:4]
            XMin, XMax, YMin, YMax = fileline[4:8]
            IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside = fileline[8:13]
            image_path = os.path.join(imgs_root, ImageID+".jpg")
            total_lines += 1
        except:
            break

        if LabelName != target_label:
            continue  # filter out unwanted image labels
        if int(IsDepiction) != 0 or int(IsGroupOf) != 0 or int(IsInside) != 0:
            continue  # unwanted samples
        
        img = cv2.imread(image_path)
        img_h, img_w, img_c = img.shape
        XMin, XMax = max(0, float(XMin)*img_w), min(img_w-1, float(XMax)*img_w)
        YMin, YMax = max(0, float(YMin)*img_h), min(img_h-1, float(YMax)*img_h)
        
        head_h, head_w = YMax - YMin, XMax - XMin
        if head_h < min_head_size_thre or head_w < min_head_size_thre:  
            continue  # this head size is too small
        
        # imgsizes.append(min(head_h, head_w))
        imgsizes.append(min(min(head_h, head_w), 512))
        
        total_cnt += 1
    

        xc, yc = XMin+head_w/2,YMin+head_h/2
        x1_new = int(xc - head_w/2*scale_ratio)
        y1_new = int(yc - head_h/2*scale_ratio)
        x2_new = int(xc + head_w/2*scale_ratio)
        y2_new = int(yc + head_h/2*scale_ratio)
        pad_left = -x1_new if x1_new < 0 else 0
        x1_new = max(0, x1_new)
        pad_top = -y1_new if y1_new < 0 else 0
        y1_new = max(0, y1_new)
        pad_right = x2_new-img_w+1 if x2_new > img_w-1 else 0
        x2_new = min(img_w-1, x2_new)
        pad_bottom = y2_new-img_h+1 if y2_new > img_h-1 else 0
        y2_new = min(img_h-1, y2_new)
        img_head = img[y1_new:y2_new, x1_new:x2_new]
        img_head = cv2.copyMakeBorder(img_head, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0,0,0))
        head_imgs_cnt += 1
        xc_adjusted, yc_adjusted = img_head.shape[1]/2, img_head.shape[0]/2
        x1_adjusted, y1_adjusted = int(xc_adjusted - head_w/2), int(yc_adjusted - head_h/2)
        head_loc_info = "_%d,%d,%d,%d"%(x1_adjusted, y1_adjusted, head_w, head_h)
        save_img_name = name_prefix + str(head_imgs_cnt).zfill(6) + head_loc_info + ".jpg"
        save_img_path = os.path.join(head_imgs_path, save_img_name)
        cv2.imwrite(save_img_path, img_head)


    print(imgs_root, "\t", total_lines, total_cnt)
    
imgsizes = np.array(imgsizes)
print(len(imgsizes), np.min(imgsizes), np.max(imgsizes), np.mean(imgsizes), np.median(imgsizes))


plt.hist(imgsizes, bins=50)
# plt.show()
plt.savefig("./OpenImagesV6_imgsize_pixel_%d.jpg"%(min_head_size_thre))

'''
# min_head_size_thre = 60 (for 3d head reconstruction)
validation/data/    303980 5065
test/data/          937327 15672
train/data/         14610229 96316
117053 60.0 512.0 166.66045354391738 130.85286399999995

# min_head_size_thre = 30 (for head pose estimation)
validation/data/    303980 6581
test/data/          937327 21098
train/data/         14610229 138118
165797 
'''
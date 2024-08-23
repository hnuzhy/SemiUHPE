
import os
import cv2
import json
import torch
import shutil
import numpy as np
import torchvision.transforms as tfs
from PIL import Image
from tqdm import tqdm

from src.config import get_config
from src.agent import get_agent
from src.fisher.fisher_utils import batch_torch_A_to_R


def process_ori_img(img_path, bbox):
    img_ori = Image.open(img_path).convert('RGB')
    img_w, img_h = img_ori.size

    [ori_x, ori_y, ori_w, ori_h] = bbox
    cx, cy = ori_x+ori_w/2, ori_y+ori_h/2
    # pad_len = max(ori_w, ori_h)  # hint 1: we want get a squared face bbox <<<------
    pad_len = (ori_w + ori_h) / 2.0  # hint 1: we want get a squared face bbox <<<------
    

    ad = 0.15  # for the DAD3DHeads test-set (an unchanged ad)
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
    
    return img_padded

if __name__ == '__main__':

    config = get_config('test')
    agent = get_agent(config)
    agent.load_ckpt(config.ckpt)
    
    json_dict_result = {}

    db_path = config.data_dir_DAD3DHeads
    test_json_dict_list = json.load(open(os.path.join(db_path, "test", "test.json"), "r"))
    for test_json_dict in tqdm(test_json_dict_list):
        item_id = test_json_dict["item_id"]
        bbox = test_json_dict["bbox"]  # [x, y, w, h] format
        # keys: {"quality","gender","expression","age","occlusions","pose","standard light"}
        attr = test_json_dict["attributes"]
        
        img_path = os.path.join(db_path, "test", "images", item_id+".png")
        annotation_path = os.path.join(db_path, "test", "annotations", item_id+".json")  # not available
        
        img_input = process_ori_img(img_path, bbox)
        img_tensor = tfs.ToTensor()(img_input)
        img_tensor = tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img_tensor)
        with torch.no_grad(): 
            agent.net.eval()
            fisher_out = agent.net(img_tensor.reshape((1,3,224,224)).cuda())
            pd_m = batch_torch_A_to_R(fisher_out)
        
        rot_mat = pd_m.detach().cpu().numpy()[0]  # 3×3

        # https://github.com/PinataFarms/DAD-3DHeads/tree/main/dad_3dheads_benchmark#evaluation
        json_dict_result[item_id] = {
            '68_landmarks_2d': [], 'N_landmarks_3d': [], '7_landmarks_3d': [],
            'rotation_matrix': rot_mat.tolist()}
    
    current_exp_name = config.network + "_"
    if "Dyna" in config.exp_detail:  current_exp_name += "SemiUHPE"
    elif "r0.05" in config.exp_detail:  current_exp_name += "Baseline"
    else:  current_exp_name += "Supervised"

    predicted_json_path = os.path.join("exps", f"testset_results_{current_exp_name}.json") 
    with open(predicted_json_path, "w") as json_file:
        json.dump(json_dict_result, json_file)
 
'''
# SemiUHPE + effinetv2-s (WildHead)
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_effinetv2_tDyna0.75_b64_ema_RO_CO_CM_full/Jul22_122502/best \
    --is_full_range --config settings/DAD3DHeads_WildHead.yml --network effinetv2 --gpu_ids 1

# SemiUHPE + resnet50
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_tDyna0.75_b16_ema_RO_CO_CM_full/Sep20_195132/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1
# SemiUHPE + repvgg
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_tDyna0.75_b32_ema_RO_CO_CM_full/Sep30_130637/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
# SemiUHPE + effinetv2-s
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_effinetv2_tDyna0.75_b32_ema_RO_CO_CM_full/Jul18_100557/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network effinetv2 --gpu_ids 1
    
# Baseline + resnet50
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_t-4.3_b16_ema_full/Oct04_231242/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1
# Baseline + repvgg
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_t-4.3_b32_ema_full/Oct06_090346/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
# Baseline + effinetv2-s
python eval_DAD3DHeads.py SSL1.0_r0.05_ce_effinetv2_t-4.8_b32_ema_full/Jul20_223732/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network effinetv2 --gpu_ids 1
    
# Supervised + resnet50
python eval_DAD3DHeads.py SSL1.0_r1.0_ce_t-5.3_b32_ema_full/Sep30_124334/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network resnet50 --gpu_ids 1
# Supervised + repvgg
python eval_DAD3DHeads.py SSL1.0_r1.0_ce_t-5.3_b32_ema_full/Sep30_124814/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network repvgg --gpu_ids 1
# Supervised + effinetv2-s
python eval_DAD3DHeads.py SSL1.0_r1.0_ce_effinetv2_t-5.3_b32_ema_full/Jul17_160508/best \
    --is_full_range --config settings/DAD3DHeads_COCOHead.yml --network effinetv2 --gpu_ids 1

'''
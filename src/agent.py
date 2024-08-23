import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch3d import transforms as trans
from datetime import datetime

from src import utils
from src.networks import get_network
from src.fisher.fisher_utils import vmf_loss as fisher_NLL  # matrixFisher loss
from src.fisher.fisher_utils import fisher_CE
from src.fisher.fisher_utils import batch_torch_A_to_R  # applicable to matrixFisher and RotationLaplace
from src.fisher.fisher_utils import fisher_entropy  # applicable to matrixFisher and RotationLaplace
from src.laplace.rotation_laplace import NLL_loss as laplace_NLL # RotationLaplace loss

from src.utils import compute_euler_angles_from_rotation_matrices
from src.augments import random_cutout_tensor, random_cutmix_tensor

def get_agent(config):
    return SSLAgent(config)


class SSLAgent:
    def __init__(self, config):
        self.config = config
        self.clock = utils.TrainClock()
        self.net = get_network(config)
        self.ema_net = get_network(config)
        # ema net is updated by ema, not training
        for param in self.ema_net.parameters():
            param.detach_()
        
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        
        nowstr = datetime.now().strftime("%Y%m%d-%H%M%S")
        logs_file_name = os.path.join(self.config.log_dir, "txt_" + nowstr + ".log" )
        self.logs_writer = open(logs_file_name, "w")
        
        if self.config.distribution == "RotationLaplace":  # matrixFisher or RotationLaplace
            torch.backends.cuda.matmul.allow_tf32 = False  # a vital operation for RotationLaplace
            current_path = os.path.dirname(os.path.abspath(__file__))
            if self.config.is_full_range == True:
                grids_path = os.path.join(current_path, 'laplace/eq_grids2.npy')  # 4608 grids (100%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids3.npy')  # 36864 grids (100%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids4.npy')  # 36864 * 8 grids (100%)
            if self.config.is_full_range == False:
                # grids_path = os.path.join(current_path, 'laplace/eq_grids3_front.npy')  # 6656 grids (18%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids3_large.npy')  # 8872 grids (24%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids3_half.npy')  # 18432 grids (50%)
                grids_path = os.path.join(current_path, 'laplace/eq_grids2.npy')  # 4608 grids (100%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids3.npy')  # 36864 grids (100%)
                # grids_path = os.path.join(current_path, 'laplace/eq_grids4.npy')  # 36864 * 8 grids (100%)
            print(f'[Rotation Laplace] Loading SO3 discrete grids {grids_path}')
            self.grids = torch.from_numpy(np.load(grids_path)).cuda()
            

    def forward(self, data, ulb_data, eval_ema=False):
        ### 1. supervised loss
        img = data.get('img').cuda()
        gt = data.get('rot_mat').cuda()  # (b, 3, 3)
        if 'euler_angles' in data:
            gt_euler = data.get('euler_angles').cuda()  # (b, 3) of (pitch, yaw, roll) in degrees
        else:
            gt_euler = None  # some train dataset has no gt_euler labels, such as DAD3DHeads train-set

        # Teacher model or student model
        if eval_ema:
            net = self.ema_net
        else:
            net = self.net
        
        fisher_out = net(img)
        
        if self.config.distribution == "matrixFisher":  # default
            losses, pred_orth = fisher_NLL(fisher_out, gt, overreg=1.025)
        if self.config.distribution == "RotationLaplace":  # newly added
            losses, pred_orth = laplace_NLL("RLaplace", fisher_out, gt, self.grids)
            
        loss = losses.mean()

        err_deg = self.compute_err_deg_from_matrices(pred_orth, gt, gt_euler=gt_euler)

        fisher_dict = dict(
            loss=loss,
            pred=fisher_out,
            pred_orth=pred_orth,
            err_deg=err_deg
        )

        # usd for val_func
        if ulb_data is None:
            return fisher_dict, None

        ### 2. unsupervised loss
        ulb_img_weak = ulb_data.get('img').cuda()
        ulb_img_strong = ulb_data.get('img_strong').cuda()
        if 'rot_mat' in ulb_data:
            ulb_gt = ulb_data.get('rot_mat').cuda()  # ulb_gt is only used for evaluation
        else:
            ulb_gt = None  # some ulb_train dataset has no gt labels, such as AFLWFace and COCOHead
        
        # ema_net
        pred_weak = self.ema_net(ulb_img_weak)  # (b*nm, 9)
        utils.requires_grad(pred_weak, False)
        
        if self.config.rotate_aug:  # for Rotation strong_aug
            ulb_aug_rot_mat = ulb_data.get('aug_rot_mat').cuda()
            pred_weak_mat = pred_weak.clone().view(-1, 3, 3)
            if self.config.train_labeled == "DAD3DHeads":
                pred_weak_adjusted = torch.matmul(ulb_aug_rot_mat, pred_weak_mat).view(-1, 9)
            if self.config.train_labeled == "300WLP":
                rot_180 = torch.FloatTensor(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])).cuda()
                pred_weak_mat_trans = torch.matmul(rot_180, pred_weak_mat.transpose(1, 2))
                pred_weak_mat_trans = torch.matmul(ulb_aug_rot_mat, pred_weak_mat_trans)
                pred_weak_mat = torch.matmul(rot_180, pred_weak_mat_trans).transpose(1, 2)
                pred_weak_adjusted = pred_weak_mat.reshape(-1, 9)
        else:
            pred_weak_adjusted = pred_weak.clone()
            
        # we may need to do CutOut or CutMix augmentation for the im_strong
        if self.config.cutout_aug:  ulb_img_strong = random_cutout_tensor(ulb_img_strong, 3, normal=True)
        if self.config.cutmix_aug:  ulb_img_strong = random_cutmix_tensor(ulb_img_strong, 3, normal=True)
        
        jpg_names = [name for name in os.listdir(self.config.log_dir) if "super_strong_" in name]
        if len(jpg_names) < 10 and (self.config.cutout_aug or self.config.cutmix_aug):
            vis_imgs = ulb_img_strong[:10, ...].clone().cpu().numpy().transpose((0, 2, 3, 1))
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            vis_imgs = np.array(np.clip(255 * (vis_imgs * std + mean), 0, 255), dtype=np.uint8)
            for idx in range(10):
                save_jpg_name = os.path.join(self.config.log_dir, 
                    self.config.train_unlabeled+"_super_strong_"+str(idx).zfill(8)+".jpg")
                cv2.imwrite(save_jpg_name, vis_imgs[idx, :, :, ::-1])
        
        pred_strong = self.net(ulb_img_strong)
        entropy = fisher_entropy(pred_weak)  # this do not need rotate_aug
        
        ''' treat entropies as the normal distribution is not generalizable, thus we give it up [2023.09.17]
        # we may give a dynamic self.config.conf_thres value by claculating mean/std of entropy
        if self.config.dynamic_thres:
            conf_thres = entropy.mean() + self.config.std_ratio * entropy.std()  # e.g., -5.6 + 2 * 0.15 = -5.3
            self.config.conf_thres = conf_thres
        '''
        
        mask_fisher = entropy < self.config.conf_thres  # (b, )

        mask_ratio_fisher = mask_fisher.sum() / len(mask_fisher)
        if mask_ratio_fisher > 0:
            pseudo_labels = batch_torch_A_to_R(pred_weak_adjusted[mask_fisher])
            if self.config.distribution == "matrixFisher":  # default representation
                if self.config.type_unsuper == 'ce':  # the default loss
                    unsuper_losses = fisher_CE(pred_weak_adjusted[mask_fisher], pred_strong[mask_fisher])
                if self.config.type_unsuper == 'nll':
                    unsuper_losses, _ = fisher_NLL(pred_strong[mask_fisher], pseudo_labels, overreg=1.025)
            if self.config.distribution == "RotationLaplace":  # newly added representation
                if self.config.type_unsuper == 'ce':  # the default loss
                    unsuper_losses = fisher_CE(pred_weak_adjusted[mask_fisher], pred_strong[mask_fisher])
                if self.config.type_unsuper == 'nll':
                    unsuper_losses, _ = laplace_NLL("RLaplace", pred_strong[mask_fisher], pseudo_labels, self.grids)
            unsuper_loss = unsuper_losses.mean()
        else:
            unsuper_loss = torch.tensor([0.], device='cuda').float()

        # We want unsupervised loss to be 1/(mu*B) Sum_{mu*B*mask} l, now unsuper_loss is 1/(mu*B*mask) Sum_{mu*B*mask} l, so multiply mask
        unsuper_loss = unsuper_loss * mask_ratio_fisher

        # errors
        if ulb_gt is not None:
            err_weakAll_gt = self.compute_err_deg_from_matrices(batch_torch_A_to_R(pred_weak_adjusted), ulb_gt)
            err_weakPseudo_gt = self.compute_err_deg_from_matrices(
                batch_torch_A_to_R(pred_weak_adjusted[mask_fisher]), ulb_gt[mask_fisher])
        else:
            err_weakAll_gt = torch.tensor([0.], device='cuda').float()
            err_weakPseudo_gt = torch.tensor([0.], device='cuda').float()
        err_strongSuper_pseudo = self.compute_err_deg_from_matrices(
            batch_torch_A_to_R(pred_strong[mask_fisher]),
            batch_torch_A_to_R(pred_weak_adjusted[mask_fisher])
        )

        fisher_dict_unsuper = dict(
            unsuper_loss=unsuper_loss,
            entropy=entropy,
            mask_ratio=mask_ratio_fisher,
            err_weakAll_gt=err_weakAll_gt,
            err_weakPseudo_gt=err_weakPseudo_gt,
            err_strongSuper_pseudo=err_strongSuper_pseudo,
        )

        return fisher_dict, fisher_dict_unsuper

    def train_func(self, data, ulb_data):
        """one step of training"""
        self.net.train()
        self.ema_net.train()

        stage2_iter = self.clock.iteration - self.config.stage1_iteration
        self.update_ema_variables(self.config.is_ema, self.config.ema_decay, stage2_iter)

        fisher_dict, fisher_dict_unsuper = self.forward(data, ulb_data)

        SSL_lambda = self.config.SSL_lambda

        loss_all = fisher_dict['loss'] + SSL_lambda * fisher_dict_unsuper['unsuper_loss']

        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()


        out_dict = dict(
            SSL_lambda=SSL_lambda,
            loss_all=loss_all
        )

        return fisher_dict, fisher_dict_unsuper, out_dict

    def val_func(self, data, eval_ema=False):
        """one step of validation"""
        self.net.eval()
        self.ema_net.eval()

        with torch.no_grad():
            fisher_dict, fisher_dict_unsuper = self.forward(data, None, eval_ema=eval_ema)
            
            # mask
            entropy = fisher_entropy(fisher_dict['pred'])

            fisher_mask = entropy < self.config.conf_thres  # (b, )
            fisher_mask_ratio = (fisher_mask.sum() / len(fisher_mask)).item()

            if fisher_mask_ratio > 0:
                if 'euler_angles' in data:
                    gt_euler = data.get('euler_angles').cuda()[fisher_mask]  # (b, 3) of (pitch, yaw, roll) in deg
                else:
                    gt_euler = None  # some val dataset has no gt_euler labels, such as DAD3DHeads val-set
                # error for pseudo labels
                fisher_err_pseudo_gt = self.compute_err_deg_from_matrices(
                    fisher_dict['pred_orth'][fisher_mask], 
                    data.get('rot_mat').cuda()[fisher_mask], gt_euler=gt_euler)
            else:
                fisher_err_pseudo_gt = None

            out_dict = dict(
                mask_ratio=fisher_mask_ratio,
                err_pseudo_gt=fisher_err_pseudo_gt,
            )

            return fisher_dict, fisher_dict_unsuper, out_dict


    def train_func_s1(self, data):
        """supervised training"""
        self.net.train()

        fisher_dict, _ = self.forward(data, None)

        loss = fisher_dict['loss']

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return fisher_dict

    def val_func_s1(self, data):
        """supervised validation"""
        self.net.eval()

        with torch.no_grad():
            fisher_dict, _ = self.forward(data, None)
            return fisher_dict


    def update_ema_variables(self, is_ema, alpha, global_step):
        if is_ema:
            # Use the true average until the exponential average is more correct
            alpha = min(1 - 1 / (global_step + 1), alpha)
        else:
            # ema_param = param if is_ema is False
            alpha = 0
        
        """https://github.com/amazon-science/exponential-moving-average-normalization/blob/main/models/fixmatch.py#L38"""
        if self.config.eman:  # EMAN (Exponential Moving Average Normalization)
            state_dict_main = self.net.state_dict()
            state_dict_ema = self.ema_net.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                """https://discuss.pytorch.org/t/what-num-batches-tracked-in-the-new-bn-is-for/27097/4"""
                if 'num_batches_tracked' in k_ema:
                    v_ema.copy_(v_main)
                else:
                    v_ema.copy_(v_ema * alpha + (1. - alpha) * v_main)
        else:
            for ema_param, param in zip(self.ema_net.parameters(), self.net.parameters()):
                ema_param.data.mul_(alpha).add_(param.detach(), alpha=1 - alpha)


    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.config.model_dir, "ckpt_iteration{}.pth".format(self.clock.iteration))
            print("[{}/{}] Saving checkpoint iteration {}...".format(self.config.exp_name, self.config.date, self.clock.iteration))
        else:
            save_path = os.path.join(self.config.model_dir, "{}.pth".format(name))
            print("[{}/{}] Saving checkpoint {}...".format(self.config.exp_name, self.config.date, name))

        # self.net
        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        # self.ema_net
        if isinstance(self.ema_net, nn.DataParallel):
            model_state_dict_ema = self.ema_net.module.cpu().state_dict()
        else:
            model_state_dict_ema = self.ema_net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'model_state_dict_ema': model_state_dict_ema,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

        self.net.cuda()
        self.ema_net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        if os.path.isabs(name):
            load_path = name
        else:
            load_path = os.path.join(self.config.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict_ema' in checkpoint.keys():
                self.ema_net.module.load_state_dict(checkpoint['model_state_dict_ema'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
            if 'model_state_dict_ema' in checkpoint.keys():
                self.ema_net.load_state_dict(checkpoint['model_state_dict_ema'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], )
        self.clock.restore_checkpoint(checkpoint['clock'])


    def compute_dynamic_entropy_threshold(self, ulb_train_bar):
        
        # if self.clock.iteration == self.config.stage1_iteration:
            # self.net.eval()
            # predict_net = self.net  # first stage using the student_net
        # else:
            # self.ema_net.eval()
            # predict_net = self.ema_net  # after fisrt stage using the teacher_net
        
        self.ema_net.eval()
        predict_net = self.ema_net  # always using ema_net is better
        
        if self.config.save_feat:  # backbone MobileNet or ResNet-18 or ResNet-50 or RepNetVGG
            
            # hook function: https://www.cnblogs.com/Fish0403/p/17141048.html 
            fmap_block = dict()  # load feature maps
            def forward_hook(module, input, output):
                fmap_block['output'] = output

            if self.config.network == 'mobilenet': select_layer_name = 'classifier.4'  # (bs, 64)
            if self.config.network == 'resnet18': select_layer_name = 'fc.4'  # (bs, 64)
            if self.config.network == 'resnet50': select_layer_name = 'fc.4'  # (bs, 128)
            if self.config.network == 'repvgg': select_layer_name = 'linear_reg.4'  # (bs, 128)
            for (name, module) in predict_net.named_modules():
                if name == select_layer_name:
                    module.register_forward_hook(hook=forward_hook)

        ulb_sample_feat = {}
        all_entropy_list = []
        with torch.no_grad():
            for i, ulb_data in enumerate(ulb_train_bar):
                ulb_img_weak = ulb_data.get('img').cuda()
                pred_weak = predict_net(ulb_img_weak)  # (bs, 9)
                entropy = fisher_entropy(pred_weak)  # (bs, 1)
                entropy_arr = entropy.detach().cpu().numpy()
                all_entropy_list.append(entropy_arr)
                
                if self.config.save_feat:
                    ulb_feat_arr = fmap_block['output'].detach().cpu().numpy()  # (bs, 64 or 128)
                    ulb_feat_list = ulb_feat_arr.tolist()  # for JSON serializable
                    entropy_list = entropy_arr.tolist()  # for JSON serializable
                    ulb_sample_idxs = ulb_data.get('idx')
                    for cur_ind, ulb_idx in enumerate(ulb_sample_idxs):
                        feat = [ulb_feat_list[cur_ind], entropy_list[cur_ind]]
                        ulb_sample_feat[str(ulb_idx)] = feat
                    
            entropy_all = np.concatenate(all_entropy_list, 0)
        
        entropy_all.sort()
        index = int(len(entropy_all) * self.config.left_ratio)
        entropy_thre = entropy_all[index]
        print("The best dynamic entropy threshold is:", entropy_thre)
        
        if self.config.save_feat:  # merge all json files
            save_feat_path = os.path.join(self.config.log_dir, 
                "ulb_feats_iter%d_thre%s.json"%(self.clock.iteration, str(entropy_thre)))
            with open(save_feat_path, "w") as json_file:
                json.dump(ulb_sample_feat, json_file)
                
        self.config.conf_thres = entropy_thre  # update the conf_thres
        return entropy_thre
   
        
    @staticmethod
    def compute_err_deg_from_quats(pred, gt):
        err_rad = trans.so3_relative_angle(trans.quaternion_to_matrix(pred), trans.quaternion_to_matrix(gt))
        err_deg = torch.rad2deg(err_rad)
        return err_deg

    # @staticmethod
    # def compute_err_deg_from_matrices(pred, gt):
        # err_rad = trans.so3_relative_angle(pred, gt)
        # err_deg = torch.rad2deg(err_rad)
        # return err_deg
    
    '''
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html
    so3_relative_angle() --> rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1R2^T)-1))`
    
    It is also the geodesic distance between both rotation matrices, which is the loss of 6DRepNet
    
    And it is also the mentioned "the angle in axis-angle representation of R1R2^T" in DAD3DHeads paper
    https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    
    3D rotation group, often denoted SO(3), is the group of all rotations about the origin of 3D Euclidean space
    https://en.wikipedia.org/wiki/3D_rotation_group
    
    https://dfki-ric.github.io/pytransform3d/rotations.html
    '''
    
    @staticmethod
    def compute_err_deg_from_matrices(pred, gt, gt_euler=None):
        if gt_euler is None:  # especially for DAD3DHeads without gt_euler labels
            err_rad = trans.so3_relative_angle(pred, gt)
            err_deg = torch.rad2deg(err_rad)
        else:
            pd_euler = compute_euler_angles_from_rotation_matrices(pred, full_range=False)*180/np.pi
            err_deg = torch.mean(torch.abs(pd_euler - gt_euler), dim=-1)  # (bs, 3) of (pitch, yaw, roll)
        return err_deg

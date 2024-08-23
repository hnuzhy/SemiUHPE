import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from os.path import dirname, abspath, join

from src.config import get_config
from src.agent import get_agent
from src.utils import cycle, dict_get

from src.datasets.dataset_300WLP import get_dataloader_300WLP
from src.datasets.dataset_AFLW2000 import get_dataloader_AFLW2000
from src.datasets.dataset_DAD3DHeads import get_dataloader_DAD3DHeads
from src.datasets.dataset_COCOHead import get_dataloader_COCOHead
from src.datasets.dataset_WildHead import get_dataloader_WildHead

def main():
    # create experiment config containing all hyperparameters
    config = get_config('train')

    # GPU usage
    if config.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_ids)

    if config.distribution == "RotationLaplace":  # RotationLaplace (better)
        torch.backends.cuda.matmul.allow_tf32 = False  # a vital operation for RotationLaplace

    '''
    [SSL datasets without domian gaps settings]
    300WLP:     only for training with or without labels, 122450 images with half flip
    AFLW2000:   only for testing with labels, 2000 images
    DAD3DHeads: only for training and testing with labels, having 37840/4312 images for train/val set
    COCOHead:   only for training without labels, having about 223049/9351 heads for train/val set (74128 left)
    
    For ablation studies:
    * 300WLP_AFLW2000 (front range): x% labeled 300WLP + (100-x)% unlabeled 300WLP + testing on AFLW2000
    For further performance improving or in-the-wild HPE
    * 300WLP_COCOHead (front range): all labeled 300WLP + unlabeled COCOHead + testing on AFLW2000
    * DAD3DHeads_COCOHead (full range): labeled DAD3DHeads-train + unlabeled COCOHead + testing on DAD3DHeads-val
    * DAD3DHeads_CrowdHuman (full range): labeled DAD3DHeads-train + unlabeled CrowdHuman + testing on DAD3DHeads-val
    '''


    # create dataloader
    if "300WLP_AFLW2000" in config.exp_name:
        assert config.is_full_range == False, "We only support front range for 300WLP!!!"
        test_loader = get_dataloader_AFLW2000('test', config)
        train_loader = get_dataloader_300WLP('train', config)
        if config.stage1_iteration < config.max_iteration:
            ulb_train_loader = get_dataloader_300WLP('ulb_train', config)
        
    if "300WLP_COCOHead" in config.exp_name:
        assert config.is_full_range == False, "We only support front range for 300WLP!!!"
        test_loader = get_dataloader_AFLW2000('test', config)
        train_loader = get_dataloader_300WLP('train_all', config)
        ulb_train_loader = get_dataloader_COCOHead('ulb_train', config)
        
    if "DAD3DHeads_COCOHead" in config.exp_name:
        assert config.is_full_range == True, "We only support full range for DAD3DHeads!!!"
        test_loader = get_dataloader_DAD3DHeads('val', config)
        train_loader = get_dataloader_DAD3DHeads('train', config)
        ulb_train_loader = get_dataloader_COCOHead('ulb_train', config)
    
    if "300WLP_WildHead" in config.exp_name:
        assert config.is_full_range == False, "We only support front range for 300WLP!!!"
        test_loader = get_dataloader_AFLW2000('test', config)
        train_loader = get_dataloader_300WLP('train_all', config)
        ulb_train_loader = get_dataloader_WildHead('ulb_train', config)
    
    if "DAD3DHeads_WildHead" in config.exp_name:
        assert config.is_full_range == True, "We only support full range for DAD3DHeads!!!"
        test_loader = get_dataloader_DAD3DHeads('val', config)
        train_loader = get_dataloader_DAD3DHeads('train', config)
        ulb_train_loader = get_dataloader_WildHead('ulb_train', config)
    
    if config.stage1_iteration < config.max_iteration:
        iter_ulb_train_loader = cycle(ulb_train_loader)

 
    # create network and training agent
    agent = get_agent(config)

    if config.cont:
        # recover training
        agent.load_ckpt(config.ckpt)
        agent.clock.tock()

        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = config.lr

    # start training
    clock = agent.clock
    best_mean_error = 360

    while True:
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step

            # change lr for all datasets in stage2
            if clock.iteration == config.stage1_iteration:
                stage1_clock = agent.clock.make_checkpoint()
                agent.load_ckpt('best')
                agent.clock.restore_checkpoint(stage1_clock)
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= 0.1

            if clock.iteration < config.stage1_iteration:
                # supervised
                s1 = True
                fisher_dict = agent.train_func_s1(data)
                loss = fisher_dict['loss']
            elif config.stage1_iteration == config.max_iteration:
                break
            else:
                # ssl
                s1 = False
                ulb_data = next(iter_ulb_train_loader)
                fisher_dict, fisher_dict_unsuper, out_dict = agent.train_func(data, ulb_data)
                loss = out_dict['loss_all']

            if agent.clock.iteration % config.log_frequency == 0:
                line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                line_str += " train/lr:{:.6f}".format(agent.optimizer.param_groups[0]['lr'])
                line_str += " train/loss:{:.6f}".format(fisher_dict['loss'])
                line_str += " train/err_mean:{:.6f}\n".format(fisher_dict['err_deg'].mean().item())
                agent.logs_writer.write(line_str)
                agent.logs_writer.flush()
                
                if not s1:
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " train_SSL/unsuper_loss:{:.6f}".format(
                        dict_get(fisher_dict_unsuper, 'unsuper_loss', -1).item())
                    line_str += " train_SSL/entropy_mean:{:.6f} train_SSL/entropy_std:{:.6f}".format(
                        dict_get(fisher_dict_unsuper, 'entropy', -1).mean().item(),
                        dict_get(fisher_dict_unsuper, 'entropy', -1).std().item())
                    line_str += " train_SSL/mask_ratio:{:.6f}\n".format(
                        dict_get(fisher_dict_unsuper, 'mask_ratio', -1).item())
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()
                
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " train_SSL/err_weakAll_gt:{:.6f} train_SSL/err_weakPseudo_gt:{:.6f}".format(
                        dict_get(fisher_dict_unsuper, 'err_weakAll_gt', -1).mean().item(),
                        dict_get(fisher_dict_unsuper, 'err_weakPseudo_gt', -1).mean().item())
                    line_str += " train_SSL/err_strongSuper_pseudo:{:.6f}\n".format(
                        dict_get(fisher_dict_unsuper, 'err_strongSuper_pseudo', -1).mean().item())
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()

            pbar.set_description("EPOCH[{}][{}]".format(clock.epoch, clock.minibatch))
            pbar.set_postfix({'loss': loss.item()})

            clock.tick()

            # evaluation
            if clock.iteration % config.val_frequency == 0:
                fisher_test_loss = []
                fisher_test_err_deg = []
                fisher_test_mask_ratio = []
                fisher_test_err_pseudo_gt = []

                testbar = tqdm(test_loader)
                for i, data in enumerate(testbar):
                    if s1:
                        fisher_dict = agent.val_func_s1(data)
                    else:
                        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data)

                        fisher_test_mask_ratio.append(out_dict['mask_ratio'])
                        if out_dict['err_pseudo_gt'] is not None:
                            fisher_test_err_pseudo_gt.append(out_dict['err_pseudo_gt'].detach().cpu().numpy())

                    fisher_test_loss.append(fisher_dict['loss'].item())
                    fisher_test_err_deg.append(fisher_dict['err_deg'].detach().cpu().numpy())

                fisher_test_err_deg = np.concatenate(fisher_test_err_deg, 0)
                line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                line_str += " test/loss:{:.6f} test/err_mean:{:.6f}\n".format(
                    np.mean(fisher_test_loss), np.mean(fisher_test_err_deg))
                agent.logs_writer.write(line_str)
                agent.logs_writer.flush()
                if not s1:
                    fisher_test_err_pseudo_gt = [-1] if len(fisher_test_err_pseudo_gt) == 0 else \
                        np.concatenate(fisher_test_err_pseudo_gt, 0)
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " test/mask_ratio:{:.6f} test/err_pseudo_gt:{:.6f}\n".format(
                        np.mean(fisher_test_mask_ratio), np.mean(fisher_test_err_pseudo_gt))
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()
                    
                # save the best checkpoint
                if np.mean(fisher_test_err_deg) < best_mean_error:
                    best_mean_error = np.mean(fisher_test_err_deg)
                    agent.save_ckpt('best')

                if not s1:
                    # For SSL, evaluate again by ema_model
                    fisher_test_loss = []
                    fisher_test_err_deg = []
                    fisher_test_mask_ratio = []
                    fisher_test_err_pseudo_gt = []

                    testbar = tqdm(test_loader)
                    for i, data in enumerate(testbar):
                        fisher_dict, fisher_dict_unsuper, out_dict = agent.val_func(data, eval_ema=True)

                        fisher_test_mask_ratio.append(out_dict['mask_ratio'])
                        if out_dict['err_pseudo_gt'] is not None:
                            fisher_test_err_pseudo_gt.append(out_dict['err_pseudo_gt'].detach().cpu().numpy())

                        fisher_test_loss.append(fisher_dict['loss'].item())
                        fisher_test_err_deg.append(fisher_dict['err_deg'].detach().cpu().numpy())

                    fisher_test_err_deg = np.concatenate(fisher_test_err_deg, 0)
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " test_ema/loss:{:.6f} test_ema/err_mean:{:.6f}\n".format(
                        np.mean(fisher_test_loss), np.mean(fisher_test_err_deg))
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()
                    
                    fisher_test_err_pseudo_gt = [-1] if len(fisher_test_err_pseudo_gt) == 0 else \
                        np.concatenate(fisher_test_err_pseudo_gt, 0)
                        
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " test_ema/mask_ratio:{:.6f} test_ema/err_pseudo_gt:{:.6f}\n".format(
                        np.mean(fisher_test_mask_ratio), np.mean(fisher_test_err_pseudo_gt))
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()
                    
            # save checkpoint
            if clock.iteration % config.save_frequency == 0:
                # for SSL only
                if config.dynamic_thres and clock.iteration >= config.stage1_iteration:
                    ulb_train_bar = tqdm(ulb_train_loader)
                    config.conf_thres = agent.compute_dynamic_entropy_threshold(ulb_train_bar)
                    line_str = datetime.now().strftime("%Y%m%d-%H%M%S")
                    line_str += " {:03d} {:09d}".format(clock.epoch, clock.iteration)
                    line_str += " settings/conf_thres:{:.6f}\n".format(config.conf_thres)
                    agent.logs_writer.write(line_str)
                    agent.logs_writer.flush()
                
                # agent.save_ckpt()  # do not save other ckpt weights

        clock.tock()

        if clock.iteration >= config.max_iteration:
            break
            
    agent.logs_writer.close()
        
if __name__ == '__main__':
    main()

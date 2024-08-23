# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # def __init__(self, model, autobalance=False, num_offsets=1):
    def __init__(self, model, autobalance=False, num_offsets=1, num_states=0):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance

        if self.autobalance:
            self.loss_coeffs = model.module.loss_coeffs if is_parallel(model) else model.loss_coeffs[-1]

        self.num_offsets = num_offsets  # the default is 3 for head part center
        self.num_states = num_states  # the default is 0 for BPJDet, is 8 for BPJDetPlus
        
        self.na = det.na
        self.nc = det.nc
        self.nl = det.nl
        self.anchors = det.anchors

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lbpl = torch.zeros(1, device=device)  # Regression loss of body part bbox
        lcts = torch.zeros(1, device=device)  # ConTact State loss of hand

        # tcls, tbox, tbps, indices, anchors = self.build_targets(p, targets)  # targets
        tcls, tbox, tbps, tctss, indices, anchors = self.build_targets(p, targets)  # targets
        
        
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # range [0, 4] * anchor
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # print(i, "Regression Bbox:", "\n", pbox.shape, pbox, "\n", tbox[i].shape, tbox[i])
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # print(i, lbox)
                
                # MSE loss of body/part object's center points
                if self.num_offsets:
                    tbp = tbps[i]
                    vis = tbp[..., 2] > 0
                    tbp_vis = tbp[vis]
                    if len(tbp_vis):
                        if self.num_states:
                            pbp = ps[:, 5 + self.nc:-self.num_states].reshape((-1, self.num_offsets // 2, 2))
                        else:
                            pbp = ps[:, 5 + self.nc:].reshape((-1, self.num_offsets // 2, 2))
                        pbp = (pbp.sigmoid() * 4. - 2.) * anchors[i][:, None, :]  # range [-2, 2] * anchor
                        pbp_vis = pbp[vis]
                        # print(i, "MSE Loss:", "\n", pbp_vis.shape, pbp_vis, "\n", tbp_vis.shape, tbp_vis)
                        l2 = torch.linalg.norm(pbp_vis - tbp_vis[..., :2], dim=-1)
                        lbpl += torch.mean(l2)
                        # print(i, lbpl)                        
                
                # BCE loss for hand contact state estimation, same as the Classification
                if self.num_states:
                    tcts = tctss[i]
                    pcts = ps[:, -self.num_states:]
                    for si in range(self.num_states):  # si --> state index [NC, SC, PC, OC] + [NC, SC, PC, OC]
                        vis = tcts[..., si] < 2  # 0, 1 and 2 is for No, Yes and Not-sure
                        tcts_vis = tcts[..., si][vis]
                        if len(tcts_vis):
                            pcts_vis = pcts[..., si][vis]
                            lcts += self.BCEcls(pcts_vis, tcts_vis.float())  # BCE (no smooth_BCE)
                            # pcts_vis_new = torch.unsqueeze(pcts_vis, dim=-1)
                            # t = torch.full_like(pcts_vis_new, self.cn, device=device)  # targets
                            # t[range(len(tcts_vis)), tcts_vis] = self.cp
                            # lcts += self.BCEcls(pcts_vis_new, t.float())  # BCE (smooth_BCE)(for mulit-classes)
                            
                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:5 + self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5 + self.nc], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lbpl *= self.hyp['part_w']
        lcts *= self.hyp['state_w']

        if self.autobalance:
            # loss = (lbox + lobj + lcls) / (torch.exp(2 * self.loss_coeffs[0])) + self.loss_coeffs[0]
            loss = (lbox + lobj + lcls + lcts) / (torch.exp(2 * self.loss_coeffs[0])) + self.loss_coeffs[0]
            loss += lbpl / (torch.exp(2 * self.loss_coeffs[1])) + self.loss_coeffs[1]
        else:
            # loss = lbox + lobj + lcls + lbpl
            loss = lbox + lobj + lcls + lbpl + lcts

        bs = tobj.shape[0]  # batch size
        
        if self.num_states:
            return loss * bs, torch.cat((lbox, lobj, lcls, lbpl, lcts)).detach()
        else:
            return loss * bs, torch.cat((lbox, lobj, lcls, lbpl)).detach()
    
    ''' Understand YOLO-style anchors 
    https://blog.csdn.net/flyfish1986/article/details/117594265 
    https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html
    '''
    def build_targets(self, p, targets):  
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h, x_part,y_part)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        # tcls, tbox, tbps, indices, anch = [], [], [], [], []
        # gain = torch.ones(7 + self.num_offsets * 3 // 2, device=targets.device)  # normalized to gridspace gain
        tcls, tbox, tbps, tctss, indices, anch = [], [], [], [], [], []
        gain = torch.ones(7 + self.num_offsets * 3 // 2 + self.num_states, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            xy_gain = torch.tensor(p[i].shape)[[3, 2]]
            gain[2:4] = xy_gain
            gain[4:6] = xy_gain
            for j in range(self.num_offsets // 2):
                part_idx = 6 + j * 3
                gain[part_idx:part_idx + 2] = xy_gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b = t[:, 0].long()  # image
            c = t[:, 1].long()  # class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            if self.num_offsets:
                # part_xy = t[:, 6:-1].reshape(-1, self.num_offsets // 2, 3)
                part_xy = t[:, 6:-(1+self.num_states)].reshape(-1, self.num_offsets // 2, 3)
                part_xy[..., :2] -= gij[:, None, :]  # grid part box relative to grid box anchor
                tbps.append(part_xy)
            
            if self.num_states:
                s = t[:, -(1+self.num_states):-1].long()  # GT of hand contact states
                tctss.append(s)

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        # return tcls, tbox, tbps, indices, anch
        return tcls, tbox, tbps, tctss, indices, anch

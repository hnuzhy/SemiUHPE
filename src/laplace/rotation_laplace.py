import os
import torch
import numpy as np
from torch.nn import functional as F

from src.fisher.fisher_utils import batch_torch_A_to_R 

# We found SVD on cuda may have speed / stability (CUDA error encountered) issues, 
# which is related to CUDA version, PyTorch version, etc., and decided to use SVD on cpu
CPUSVD = True
EPS = 1e-8

torch.backends.cuda.matmul.allow_tf32 = False

def delta_R(N):
    """
    volume for SO(3) is 1 (based on the normalization condition of Haar measure)
    volume for S^3 is 2*pi^2
    so delta_R is 1 / N, delta_q is 2*pi^2 / N
    """
    return 1 / N


def NLL_loss(fn_type, pred, gt, grids):
    """
    @param pred: A from network (b, 3, 3)
    @param gt: gt matrices, (b, 3, 3)
    """
    pred = pred.reshape(-1, 3, 3)
    logp = log_pdf(fn_type, pred, gt, grids)
    losses = -logp

    pred_orth, _ = analytical_mode(pred, fn_type)
    return losses, pred_orth
    
    # pred_orth = batch_torch_A_to_R(pred)
    # return losses, pred_orth



def logF_const(power_fn, A, grids):
    # grids = grids.to(A.device)
    N = grids.shape[0]

    # change dimensionality for broadcasting
    grids1 = grids[None]  # (1, N, 3, 3)
    A1 = A[:, None]  # (b, 1, 3, 3)

    power = power_fn(A1, grids1)    # (b, N)
    # to avoid numerical explosion
    # logF = c + log(Sum{ exp(power-c) } * dR)
    c = power.max(dim=-1)[0]  # (b, )
    exps = torch.exp(power - c[:, None])    # (b, N)
    logF = c + torch.log(exps.sum(1) * delta_R(N))
    return logF


def logF_const_laplace(power_fn, A, grids):
    # grids = grids.to(A.device)
    N = grids.shape[0]

    # change dimensionality for broadcasting
    grids1 = grids[None]  # (1, N, 3, 3)
    A1 = A[:, None]  # (b, 1, 3, 3)

    power = power_fn(A1, grids1)    # (b, N)
    # to avoid numerical explosion
    # logF = c + log(Sum{ exp(power-c) } * dR)
    c = power.max(dim=-1)[0]  # (b, )
    exps = torch.exp(power - c[:, None])    # (b, N)
    logF = c + torch.log((exps / (-power)).sum(1) * delta_R(N))
    return logF



def log_pdf(fn_type, A, x, grids, broadcast=False):
    fn_dict = dict(
        RFisher=power_fn_fisher,
        RLaplace=power_fn_sqrtL2_proper,
    )
    power_fn = fn_dict[fn_type]

    if 'RLaplace' in fn_type:
        logF = logF_const_laplace(power_fn, A, grids)
    else:
        logF = logF_const(power_fn, A, grids)

    if x.shape[0] == grids.shape[0] or broadcast:
        # change dimensionality for broadcasting
        x = x[None]  # (1, N, 3, 3)
        A = A[:, None]  # (b, 1, 3, 3)
        logF = logF[:, None]    # (b, 1)
    if 'RLaplace' in fn_type:
        power = power_fn(A, x)
        pdf = -logF + power - torch.log(-power)
    else:
        pdf = -logF + power_fn(A, x)

    return pdf      # (b, ) if not broadcast; (b, N) if broadcast


def analytical_mode(pred, fn_type):
    device = pred.device
    if CPUSVD:
        pred = pred.cpu()

    U, S, VT = torch.linalg.svd(pred)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, VT))
    diag = torch.stack((torch.ones_like(s3sign), torch.ones_like(s3sign), s3sign), -1)
    diag = torch.diag_embed(diag)

    pred_orth = U @ diag @ VT
    pred_orth = pred_orth.to(device)
    return pred_orth, s3sign


def power_fn_fisher(A, input):
    """
    power = tr(A^T x)
    To verify the discrete calculation. Should be consistent with fisher_utils.fisher_log_pdf()

    if x:
        @param A: (b, 3, 3)
        @param input: (b, 3, 3)
        @return logp: (b, )
    if grids:
        @param A: (b, 1, 3, 3)
        @param input: (1, N, 3, 3)
        @return logp: (b, N)
    """

    mul = torch.matmul(torch.transpose(A, -1, -2), input)
    assert mul.shape[-2:] == (3, 3)
    power = mul[..., 0, 0] + mul[..., 1, 1] + mul[..., 2, 2]

    return power


def power_fn_sqrtL2_proper(A, input):
    """
    power = -sqrt{ s1+s2+s3 - tr(A^T x) }

    if x:
        @param A: (b, 3, 3)
        @param input: (b, 3, 3)
        @return logp: (b, )
    if grids:
        @param A: (b, 1, 3, 3)
        @param input: (1, N, 3, 3)
        @return logp: (b, N)
    """
    mul = torch.matmul(torch.transpose(A, -1, -2), input)
    assert mul.shape[-2:] == (3, 3)
    tr = mul[..., 0, 0] + mul[..., 1, 1] + mul[..., 2, 2]

    device = A.device
    if CPUSVD:
        A = A.cpu()

    S = torch.linalg.svdvals(A)
    S = torch.cat((S[..., :-1], S[..., -1:] * torch.sign(torch.det(A))[..., None]), -1)

    s_sum = S.sum(-1)
    s_sum = s_sum.to(device)

    sqrt_min = (s_sum - tr).min()
    # if not sqrt_min > -0.01:
        # print(f'power_fn_sqrtL2: sqrt(negative numbers), min:{(s_sum - tr).min()}, num: {((s_sum - tr) < 0).sum()}')

    power = -torch.sqrt(torch.clamp_min(s_sum - tr, EPS))

    return power



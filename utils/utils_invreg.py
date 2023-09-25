import torch
from torch import autograd
from losses import scl_loss, scl_loss_mid, scl_logits
import dist_all_gather


def cal_entropy(prob, dim=1):
    return -(prob * prob.log()).sum(dim=dim)


def soft_scl_logits(logits, logits_mask, mask, weights, mode='v1', nonorm=False):
    if mode == 'v1':
        assert logits.size(0) == logits.size(1)
        logits *= weights
        cont_loss_env = scl_logits(logits, logits_mask, mask)
    elif mode == 'v2':
        bs = logits.size(0)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # [bs, bs]

        pos_mask = mask.long()
        neg_mask = logits_mask * (~pos_mask.bool()).long()
        n_pos = pos_mask.sum(1)
        n_neg = neg_mask.sum(1)

        weight_pos = weights * pos_mask
        weight_neg = weights * neg_mask

        weight_neg_norm = weight_neg / weight_neg.sum(1, keepdim=True) * n_neg.unsqueeze(1)
        weights_new = weight_neg_norm + weight_pos.bool().float()

        exp_logits = torch.exp(logits) * weights_new * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [bs,bs]

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        # loss
        loss = - mean_log_prob_pos  # [bs,]

        if nonorm:
            cont_loss_env = (loss * weight_pos.sum(1) / n_pos).sum() / bs
        else:
            cont_loss_env = (loss * weight_pos.sum(1) / n_pos).sum() / (weight_pos.sum(1) / n_pos).sum()
    return cont_loss_env


def env_loss_ce_ddp(logits, labels, wsize, cfg, updated_split_all, current_ep=0):
    # https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
    """
    compute gradients based IRM penalty with DDP fc setting
    :param logits: local logits [bs, C]
    :param labels: local labels [bs]
    :param wsize: world size
    :param cfg: config
    :param updated_split_all: list of all partition matrix
    :return:
    avg IRM penalty of all partitions
    """
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    env_penalty = []
    assert isinstance(updated_split_all, list), 'retain all previous partitions'
    for updated_split_each in updated_split_all:
        per_penalty = []
        env_num = updated_split_each.shape[-1]
        split_current = updated_split_each[labels.flatten()]  # 4bs
        group_assign = split_current.argmax(dim=1)
        for env in range(env_num):
            # split groups
            select_idx = torch.where(group_assign == env)[0]

            # filter subsets
            sub_logits = logits[select_idx]
            sub_label = labels[select_idx]

            # compute penalty
            scale_dummy = torch.tensor(1.).cuda().requires_grad_()
            loss_env = ce_loss(sub_logits * scale_dummy, sub_label)
            grad = autograd.grad(loss_env, [scale_dummy], create_graph=True)[0]
            grad_avg = dist_all_gather.all_reduce(grad) / wsize
            per_penalty.append(torch.sum(grad_avg ** 2))

        env_penalty.append(torch.stack(per_penalty).mean())

    loss_penalty = torch.stack(env_penalty).mean()

    return loss_penalty * cfg.invreg['penalty_weight']


def penalty(logits, logits_mask, mask, irm_temp):
    device = logits.device
    scale = torch.ones((1, logits.size(-1))).to(device).requires_grad_()
    logits = logits / irm_temp * scale
    loss = scl_logits(logits, logits_mask, mask)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def soft_penalty(logits, logits_mask, mask, loss_weight, loss_mode, nonorm, irm_temp):
    device = logits.device
    scale = torch.ones((1, logits.size(-1))).to(device).requires_grad_()
    logits = logits / irm_temp * scale
    loss = soft_scl_logits(logits, logits_mask, mask, loss_weight,
                           mode=loss_mode, nonorm=nonorm)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def assign_features(feature, local_labels, updated_split, env_idx):
    split_current = updated_split[local_labels]
    group_assign = split_current.argmax(dim=1)
    select_idx = torch.where(group_assign == env_idx)[0]
    return feature[select_idx], local_labels[select_idx]


def assign_loss(loss, local_labels, updated_split, env_idx):
    split_current = updated_split[local_labels]
    group_assign = split_current.argmax(dim=1)
    select_idx = torch.where(group_assign == env_idx)[0]
    return loss[select_idx]

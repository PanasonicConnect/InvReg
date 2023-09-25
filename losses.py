import torch
import math
import torch.nn.functional as F


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        # for noise sample suppression
        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:  # Arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        elif self.m3 > 0:  # CosFace
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            pass

        return logits


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


def contrastive_loss(feature, label, temperature=0.3):
    device = feature.device
    bs = label.shape[0]
    lab_mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    feat = F.normalize(feature, dim=1)
    similarity_matrix = torch.matmul(feat, feat.T)
    mask = torch.eye(bs, dtype=torch.bool).to(device)
    labels = lab_mask[~mask].view(bs, -1)
    similarity_matrix = similarity_matrix[~mask].view(bs, -1) / temperature

    # select and combine multiple positives
    pos = torch.zeros_like(similarity_matrix).to(device)
    pos[labels.bool()] = similarity_matrix[labels.bool()].exp()

    # select only the negatives the negatives
    neg = torch.zeros_like(similarity_matrix).to(device)
    neg[~labels.bool()] = similarity_matrix[~labels.bool()].exp()

    # loss computation
    denominator = (pos + neg).sum(dim=-1).reshape(-1, 1)
    res = pos / denominator
    loss = torch.zeros_like(res).to(device)
    loss[labels.bool()] = -torch.log(res[labels.bool()])
    loss = loss.sum(dim=-1) / labels.sum(dim=-1)

    return loss.mean()


def scl_loss(feature, label, temperature=0.3):
    # implementation based on https://github.com/HobbitLong/SupContrast
    base_temperature = temperature
    device = feature.device
    bs = label.shape[0]
    feature = F.normalize(feature, dim=1)

    # create mask
    mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(bs).view(-1, 1).to(device), 0)  # non-self mask
    mask = mask * logits_mask  # pos mask

    valid_ind = mask.sum(-1) > 0

    anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), temperature)[valid_ind]

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask[valid_ind] +1e-8  # both pos and neg logits
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask[valid_ind] * log_prob).sum(1) / mask[valid_ind].sum(1)

    loss = - (temperature / base_temperature) * mean_log_prob_pos

    return loss.mean()


def scl_loss_mid(feature, label, temperature=0.3):
    # logits: similarity matrix
    # logits_mask: non-self mask
    # mask: same id non-self mask
    # index sequence: [[0~bs], [0~bs]...]

    device = feature.device
    bs = label.shape[0]
    feature = F.normalize(feature, dim=1)

    # create mask
    mask = (label.unsqueeze(0) == label.unsqueeze(1)).float().to(device)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(bs).view(-1, 1).to(device), 0)
    mask = mask * logits_mask

    logits = torch.div(torch.matmul(feature, feature.T), temperature)

    # compute the index
    index_sequence = torch.arange(bs).to(device)
    index_sequence = index_sequence.unsqueeze(0).expand(bs, bs)

    valid_ind = mask.sum(-1)>0

    return logits[valid_ind], logits_mask[valid_ind], mask[valid_ind], index_sequence[valid_ind]


def scl_logits(logits, logits_mask, mask):
    assert min(mask.sum(-1))>0
    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # both pos and neg logits
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - mean_log_prob_pos

    return loss.mean()

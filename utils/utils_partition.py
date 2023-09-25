import logging
import os
import collections
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data

from losses import scl_loss_mid, scl_loss
from utils.utils_invreg import cal_entropy, soft_scl_logits, soft_penalty


def load_past_partition(cfg, epoch):
    updated_split_all = []
    total = cfg.invreg['stage']
    past = total[:total.index(epoch) + 1]
    for i in past:
        past_dir = os.path.join(cfg.output, 'saved_feat', f'epoch_{i}', 'final_partition.npy')
        partition = torch.from_numpy(np.load(past_dir))
        updated_split_all.append(partition)

    return updated_split_all


class update_split_dataset(data.Dataset):
    def __init__(self, feature_all, label_all):
        """Initialize and preprocess the dataset."""
        self.feature = torch.from_numpy(feature_all)
        self.label = torch.from_numpy(label_all)

    def __getitem__(self, index):
        """Return one image and its corresponding label."""
        feat = self.feature[index]
        lab = self.label[index]
        return feat, lab

    def __len__(self):
        """Return the number of images."""
        return self.feature.size(0)


def auto_split_offline(mlp_net, trainloader, part_module, cfg, summary_writer, local_rank):
    # initialize the parameters
    temperature = cfg.invreg['temperature']
    irm_temp = cfg.invreg['irm_temp']
    loss_mode = cfg.invreg['loss_mode']
    irm_mode = cfg.invreg['irm_mode']
    irm_weight = cfg.invreg['irm_weight']
    constrain = cfg.invreg['constrain']
    cons_relax = cfg.invreg['cons_relax']
    nonorm = cfg.invreg['nonorm']
    num_env = cfg.invreg['env_num']  # [N, 2]

    # irm mode: v1 is original irm; v2 is variance
    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0

    # optimizer and schedule
    part_module.train()
    pre_optimizer = torch.optim.Adam(list(part_module.parameters()), lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [3, 6, 9, 12], gamma=0.2, last_epoch=-1)

    global_step = 0
    total_epoch = 15
    for epoch in range(total_epoch):
        trainloader.sampler.set_epoch(epoch)

        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [], [], [], [], 0

        for ori_feat, label in trainloader:
            global_step += 1
            bs = ori_feat.size(0)
            training_num += bs
            loss_cont_list, loss_penalty_list = [], []

            re_shuffle = torch.randperm(bs)
            ori_feat = ori_feat[re_shuffle].cuda()
            label = label[re_shuffle].cuda()

            with torch.no_grad():
                feat = mlp_net(ori_feat)  # mapped features
            sample_split = part_module(label)

            if irm_mode == 'v1':  # gradient
                for env_idx in range(num_env):
                    logits, logits_mask, mask, index_sequence = scl_loss_mid(feat, label, temperature=1.0)
                    loss_weight = sample_split[:, env_idx][index_sequence]  # [bs, bs-1]
                    cont_loss_env = soft_scl_logits(logits / temperature, logits_mask, mask, loss_weight,
                                                    mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                    # irm_loss based on weighted CL loss
                    penalty_grad = soft_penalty(logits, logits_mask, mask, loss_weight, loss_mode, nonorm, irm_temp)
                    loss_penalty_list.append(penalty_grad)

                cont_loss_epoch = torch.stack(loss_cont_list).mean()  # contrastive loss
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()  # gradient of the CL loss
                risk_final = - (cont_loss_epoch + irm_weight * inv_loss_epoch)

            elif irm_mode == 'v2':  # variance
                for env_idx in range(num_env):
                    logits, logits_mask, mask, index_sequence = scl_loss_mid(feat, label, temperature=1.0)
                    loss_weight = sample_split[:, env_idx][index_sequence]  # [bs, bs-1]
                    cont_loss_env = soft_scl_logits(logits / temperature, logits_mask, mask, loss_weight,
                                                    mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                cont_loss_epoch = torch.stack(loss_cont_list).mean()  # contrastive loss
                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))  # variance of the CL loss
                risk_final = - (cont_loss_epoch + irm_weight * inv_loss_epoch)

            if constrain:  # constrain for balanced partition
                if nonorm:
                    constrain_loss = 0.2 * (- cal_entropy(sample_split.mean(0), dim=0) +
                                            cal_entropy(sample_split, dim=1).mean())
                else:
                    if cons_relax:
                        constrain_loss = torch.relu(0.6365 - cal_entropy(sample_split.mean(0), dim=0))
                    else:
                        constrain_loss = - cal_entropy(sample_split.mean(0),
                                                       dim=0)
                risk_final += constrain_loss

            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()
            soft_split_print = part_module.module.partition_matrix.detach().clone()

            if local_rank == 0:
                risk_all_list.append(risk_final.item())
                risk_cont_all_list.append(-cont_loss_epoch.item())
                risk_penalty_all_list.append(-inv_loss_epoch.item())
                risk_constrain_all_list.append(constrain_loss.item())

                avg_risk = sum(risk_all_list) / len(risk_all_list)  # total loss
                avg_cont_risk = sum(risk_cont_all_list) / len(risk_cont_all_list)  # CL env
                avg_irm_risk = sum(risk_penalty_all_list) / len(risk_penalty_all_list)  # IRM penalty
                avg_cst_risk = sum(risk_constrain_all_list) / len(risk_constrain_all_list)  # balance penalty

                lr = pre_optimizer.param_groups[0]['lr']
                summary_writer.add_scalar(tag='IRM_loss', scalar_value=risk_final.data,
                                          global_step=global_step)  # total loss
                summary_writer.add_scalar(tag='CL_loss', scalar_value=-cont_loss_epoch.item(),
                                          global_step=global_step)  # CL loss of each env
                summary_writer.add_scalar(tag='Penalty_loss',
                                          scalar_value=-inv_loss_epoch.item(),
                                          global_step=global_step)  # IRM penalty (grad/var)
                summary_writer.add_scalar(tag='Constrain_loss',
                                          scalar_value=constrain_loss.item(),
                                          global_step=global_step)  # balance partition
                np.save(os.path.join(cfg.invreg['output_partition'], f'partition_{epoch}.npy'),
                        soft_split_print.cpu().numpy())

                if global_step % 200 == 0:
                    logging.info(
                        '\rUpdating Env [%d/%d]'
                        'Total_loss: %.2f  '
                        'CL_loss: %.2f  '
                        'Penalty_loss: %.2f  '
                        'Constrain_loss: %.2f  '
                        'Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                        % (epoch, total_epoch,
                           avg_risk,
                           avg_cont_risk,
                           avg_irm_risk,
                           avg_cst_risk,
                           lr, irm_mode, F.softmax(soft_split_print, dim=-1)))

        pre_scheduler.step()

    soft_split_final = part_module.module.partition_matrix.detach().clone()
    if local_rank == 0:
        logging.info(
            'Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Lr: %.4f  Inv_Mode: %s'
            % (epoch, total_epoch, training_num, len(trainloader.dataset),
               avg_risk,
               avg_cont_risk,
               avg_irm_risk,
               avg_cst_risk,
               lr, irm_mode))
        final_split_softmax = F.softmax(soft_split_final, dim=-1)
        group_assign = final_split_softmax.argmax(dim=1)
        dstr = collections.Counter(group_assign.cpu().numpy().tolist())
        dstr = {key: dstr[key] for key in sorted(dstr)}
        dstr = list(dstr.values())
        logging.info('Distribution:' + ' / '.join([str(d) for d in dstr]))
        del pre_optimizer, final_split_softmax, part_module, soft_split_print
        logging.info('The partition learning is completed, saving best partition matrix...')
        np.save(os.path.join(cfg.invreg['output_partition'], 'final_partition.npy'), soft_split_final.cpu().numpy())

    return soft_split_final


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, head='mlp', dim_in=512, feat_dim=128):
        super(MLP, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        x_norm = F.normalize(x.float(), dim=1)
        mlp_x = self.head(x_norm)
        return mlp_x


class Partition(nn.Module):
    def __init__(self, n_cls, n_env):
        super(Partition, self).__init__()
        self.partition_matrix = nn.Parameter(torch.randn((n_cls, n_env)))
        self.n_env = n_env

    def forward(self, label):
        sample_split = F.softmax(self.partition_matrix[label], dim=-1)
        return sample_split


def train_mlp(mlp_net, train_loader, cfg, summary_writer, local_rank):
    mlp_net.train()
    mlp_optimizer = torch.optim.Adam(list(mlp_net.parameters()), lr=0.2, weight_decay=0.)
    mlp_scheduler = MultiStepLR(mlp_optimizer, [2, 4], gamma=0.1, last_epoch=-1)

    global_step = 0
    total_epoch = 5
    for epoch in tqdm(range(total_epoch)):
        train_loader.sampler.set_epoch(epoch)
        for ori_feat, label in train_loader:
            bs = label.size(0)
            re_shuffle = torch.randperm(bs)
            ori_feat = ori_feat[re_shuffle].cuda()
            label = label[re_shuffle].cuda()
            feat = mlp_net(ori_feat)  # mapped features
            scl_mlp = scl_loss(feat, label, cfg.invreg['temperature'])

            mlp_optimizer.zero_grad()
            scl_mlp.backward()
            mlp_optimizer.step()
            lr = mlp_optimizer.param_groups[0]['lr']
            mlp_scheduler.step(epoch=epoch)
            global_step += 1
            if local_rank == 0:
                summary_writer.add_scalar(tag='scl_loss', scalar_value=scl_mlp.item(),
                                          global_step=global_step)
                summary_writer.add_scalar(tag='scl_lr', scalar_value=lr,
                                          global_step=global_step)

    return mlp_net


def update_partition(cfg, save_dir, n_cls, emb, lab, summary_writer,
                     device, local_rank, wsize):
    cfg.invreg['output_partition'] = save_dir

    ### Load the data
    from utils.utils_distributed_sampler import DistributedSampler_CL
    dataset = update_split_dataset(emb, lab.astype(int))
    with torch.cuda.device(device):
        train_sampler = DistributedSampler_CL(
            dataset, num_replicas=wsize, rank=local_rank, seed=0)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2048, pin_memory=False,
            num_workers=4, drop_last=True, sampler=train_sampler)

    ### Learn feature net based on SCL
    mlp_net = MLP(head='mlp', dim_in=512, feat_dim=128).cuda()
    mlp_net.apply(weights_init)
    mlp_net = torch.nn.parallel.DistributedDataParallel(
        module=mlp_net, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    mlp_net = train_mlp(mlp_net, train_loader, cfg, summary_writer, local_rank)
    mlp_net.eval()

    ### Learn partition matrix
    part_module = Partition(n_cls, cfg.invreg['env_num']).cuda()
    part_module = torch.nn.parallel.DistributedDataParallel(
        module=part_module, device_ids=[local_rank])
    part_module.train().cuda()
    updated_split = auto_split_offline(mlp_net, train_loader, part_module, cfg, summary_writer, local_rank)
    del dataset, train_loader
    del mlp_net

    return updated_split

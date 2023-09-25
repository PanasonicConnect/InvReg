import os

import numpy as np
import torch
from torch import distributed

from dataset import get_dataloader_partition


def extract_feat_per_gpu(backbone, cfg, args, save_dir):
    ### extract features
    from tqdm import tqdm
    partition_loader = get_dataloader_partition(cfg.rec, args.local_rank, cfg.batch_size, 2048, 2)
    backbone.eval()
    distributed.barrier()
    with torch.no_grad():
        feat_lst = []
        label_lst = []
        for idx, (img, local_labels) in enumerate(tqdm(partition_loader)):
            local_embeddings = backbone(img)
            feat_lst.append(local_embeddings.cpu().numpy())
            label_lst.append(local_labels.cpu().numpy())
    distributed.barrier()
    feature = np.concatenate(feat_lst, axis=0)
    label = np.concatenate(label_lst, axis=0)
    del feat_lst, label_lst
    del local_labels, local_embeddings
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'embeddings_{}.npy'.format(args.local_rank)), feature)
    np.save(os.path.join(save_dir, 'labels_{}.npy'.format(args.local_rank)), label)
    distributed.barrier()

    del partition_loader
    del feature, label
    backbone.train()


def concat_feat(n_img, wsize, save_dir):
    ### re-arrange features
    n_total = int(np.ceil(n_img / wsize) * wsize)
    emb_size = 512

    ## Re-arrange feature and label
    new_emb = np.zeros((n_total, emb_size))
    new_lab = np.zeros((n_total,))

    for i in range(wsize):
        new_emb[i::wsize, :] = np.load(os.path.join(save_dir, "embeddings_{}.npy".format(i)))
        new_lab[i::wsize] = np.load(os.path.join(save_dir, "labels_{}.npy".format(i)))

    emb = new_emb[:n_img, :]
    lab = new_lab[:n_img]
    del new_emb, new_lab
    for i in range(wsize):
        os.remove(os.path.join(save_dir, "embeddings_{}.npy".format(i)))
        os.remove(os.path.join(save_dir, "labels_{}.npy".format(i)))

    np.save(os.path.join(save_dir, 'feature.npy'), emb)
    np.save(os.path.join(save_dir, 'label.npy'), lab)
    return emb, lab

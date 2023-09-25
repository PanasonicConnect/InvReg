from easydict import EasyDict as edict

config = edict()
### --------- Default setting ---------
config.margin_list = (1.0, 0.4, 0.0)  # /, arcface, cosface
config.scale = 64
config.dropout = 0.4
config.network = "r50"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.verbose = 2000

### --------- Resume setting ---------
config.output = './trained_models/invreg_r50_emore_cifp'
config.resume = False
config.pretrained = ""
config.pretrained_ep = 0

### --------- LR setting ---------
config.lr = 0.1
config.batch_size = 128
config.scheduler = 'step'
config.step = [110000, 190000, 220000]  # [9.6, 16.7, 19.3] epoch
config.num_epoch = 21
config.warmup_epoch = 0

### --------- CIFP setting ---------
config.cifp = {}
config.cifp['use_cifp'] = True
config.cifp['far'] = None
if config.margin_list[-1]==0.0:
    config.cifp['m'] = config.margin_list[1]
    config.cifp['mode'] = 'arcface'
if config.margin_list[1]==0.0:
    config.cifp['m'] = config.margin_list[-1]
    config.cifp['mode'] = 'cosface'
config.cifp['s'] = config.scale

### --------- Inv-Reg setting ---------
config.invreg = {}
config.invreg['env_num_lst'] = [2,2,2]
config.invreg['stage'] = [1,11,17]

# Partition learning parameter
config.invreg['bs'] = 5120
config.invreg['temperature'] = 0.3
config.invreg['irm_temp'] = 0.5
config.invreg['loss_mode'] = 'v2'
config.invreg['irm_mode'] = 'v2'
config.invreg['irm_weight'] = 0.2
config.invreg['penalty_weight'] = 0.2
config.invreg['nonorm'] = False
config.invreg['offline'] = True
config.invreg['random_init'] = True
config.invreg['constrain'] = True
config.invreg['cons_relax'] = False

# Feature learning parameter
config.invreg['irm_train'] = 'grad'  # var /grad
config.invreg['loss_weight_irm'] = 0.05
config.invreg['loss_weight_irm_anneal'] = True

### --------- Trainset and testset setting ---------
config.datainfo = ""
config.val_rec = "face_data/validation/"
config.val_targets = ['rfw_caucasian','rfw_african','rfw_asian','rfw_indian']

config.dataname = "emore"
config.rec = "face_data/faces_emore"
config.val_targets = []
config.num_classes = 85742
config.num_image = 5822653


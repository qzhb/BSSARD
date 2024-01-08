import os
import argparse
import socket
from datetime import datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from configs.args_config import get_argsparser
from configs.model_config import get_model

from utils.data_util import load_video_features, save_json, load_json
from utils.data_gen import gen_or_load_dataset
from utils.data_loader_t7 import get_train_loader, get_test_loader
from utils.runner_utils_t7 import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, get_last_checkpoint


# get args of user
parser = get_argsparser()
configs = parser.parse_args()

# set random seed
set_th_config(configs.seed)


# prepare or load dataset
dataset = gen_or_load_dataset(configs)

configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']

# get train and test loader
train_loader = get_train_loader(dataset=dataset['train_set'], configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], configs)
test_iid_loader = None if dataset['test_iid_set'] is None else get_test_loader(dataset=dataset['test_iid_set'], configs=configs)
test_ood_loader = None if dataset['test_ood_set'] is None else get_test_loader(dataset=dataset['test_ood_set'], configs=configs)

configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
model, optimizer, scheduler, model_adapter = get_model(configs, device, dataset, train_loader, writer=None)

checkpoint_dirs = [
    "results/charades_org_cd/base_none_base_3/Feb24_10-58-19/model"
]

model_state_dict = model.state_dict()
for checkpoint_dir in checkpoint_dirs:
    filename = get_last_checkpoint(checkpoint_dir, suffix='t7')
    print("load checkpoint: " + filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)

    model.eval()
    r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU, score_str = model_adapter.test_one_epoch_template(test_iid_loader, "test_iid")
    print('【Test】Epoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f \ndr1i3: %.2f | dr1i5: %.2f | dr1i7: %.2f '
          '\nmIoU: %.2f' % (1, (1) * num_train_batches, r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU), flush=True)

    r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU, score_str = model_adapter.test_one_epoch_template(test_ood_loader, "test_ood")
    print('【Test】Epoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f \ndr1i3: %.2f | dr1i5: %.2f | dr1i7: %.2f '
          '\nmIoU: %.2f' % (1, (1) * num_train_batches, r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU), flush=True)
    
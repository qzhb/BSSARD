import os
import socket
from datetime import datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm

from configs.args_config import get_argsparser
from configs.model_config import get_model

from utils.data_util import load_video_features, save_json, load_json, load_pickle, load_i3d_video_features
from utils.data_gen import gen_or_load_dataset
from utils.data_loader_t7 import get_train_loader, get_test_loader
from utils.runner_utils_t7 import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, \
    get_last_checkpoint
from utils.utils import model_info
# get args of user
parser = get_argsparser()
configs = parser.parse_args()

# set random seed
set_th_config(configs.seed)

# prepare or load dataset
dataset = gen_or_load_dataset(configs)

configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']
# print("word_size " + str(dataset['n_words']))

# get train and test loader
train_loader = get_train_loader(dataset=dataset['train_set'], configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset['val_set'], configs)
test_iid_loader = get_test_loader(dataset=dataset['test_iid_set'], configs=configs)
test_ood_loader = get_test_loader(dataset=dataset['test_ood_set'], configs=configs)

configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)


# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# device = 'cpu'
# create model dir
if configs.verbose != 0:
    timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
    home_dir = os.path.join(configs.model_dir, '_'.join([configs.task, configs.vf, configs.qf]), configs.model_name, timestamp)
    model_dir = os.path.join(home_dir, "model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    writer = SummaryWriter(log_dir=os.path.join(home_dir, "tensorboard", "runs", socket.gethostname()),
                           comment='-params')
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
else:
    writer = None
    score_writer = None

model, optimizer, scheduler, model_adapter = get_model(configs, device, dataset, train_loader, writer)
model_info(model)
# model_info(model_adapter.generator)
# start training
best_r1i7 = -1.0
best_r1i5 = -1.0
best_mIoU = -1.0
best_dr1i7 = -1.0
best_dr1i5 = -1.0

iid_best_r1i7 = -1.0
iid_best_r1i5 = -1.0
iid_best_mIoU = -1.0
iid_best_dr1i7 = -1.0
iid_best_dr1i5 = -1.0
print('start training...', flush=True)
for epoch in range(configs.epochs):
    # 训练一轮
    model.train()
    model_adapter.train_one_epoch_template()

    # 测试一轮
    model.eval()
    r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU, score_str = model_adapter.test_one_epoch_template(test_iid_loader, "test_iid")
    print(
        '\n【test_iid】Epoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f \ndr1i3: %.2f | dr1i5: %.2f | dr1i7: %.2f \nmIoU: %.2f' % (
            epoch + 1, (epoch + 1) * num_train_batches, r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU), flush=True)
    if configs.verbose != 0:
        score_writer.write(score_str + f"(test_iid)\n")
        score_writer.flush()
        writer.add_scalar(f'test_iid/r1i3', r1i3, epoch + 1)
        writer.add_scalar(f'test_iid/r1i5', r1i5, epoch + 1)
        writer.add_scalar(f'test_iid/r1i7', r1i7, epoch + 1)
        writer.add_scalar(f'test_iid/mIoU', mIoU, epoch + 1)

    iid_best_r1i5 = max(iid_best_r1i5, r1i5)
    iid_best_r1i7 = max(iid_best_r1i7, r1i7)
    iid_best_dr1i5 = max(iid_best_dr1i5, dr1i5)
    iid_best_dr1i7 = max(iid_best_dr1i7, dr1i7)
    iid_best_mIoU = max(iid_best_mIoU, mIoU)

    r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU, score_str = model_adapter.test_one_epoch_template(test_ood_loader, "test_ood")
    print(
        '\n【test_ood】Epoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f \ndr1i3: %.2f | dr1i5: %.2f | dr1i7: %.2f \nmIoU: %.2f' % (
            epoch + 1, (epoch + 1) * num_train_batches, r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mIoU), flush=True)
    if configs.verbose != 0:
        score_writer.write(score_str + f"(test_ood)\n")
        score_writer.flush()
        writer.add_scalar(f'test_ood/r1i3', r1i3, epoch + 1)
        writer.add_scalar(f'test_ood/r1i5', r1i5, epoch + 1)
        writer.add_scalar(f'test_ood/r1i7', r1i7, epoch + 1)
        writer.add_scalar(f'test_ood/mIoU', mIoU, epoch + 1)
    best_r1i5 = max(best_r1i5, r1i5)
    best_dr1i5 = max(best_dr1i5, dr1i5)
    best_dr1i7 = max(best_dr1i7, dr1i7)
    best_mIoU = max(best_mIoU, mIoU)
    model.train()
    if r1i7 >= best_r1i7 and configs.verbose != 0:
        best_r1i7 = r1i7
        save_path = os.path.join(model_dir, '{}_{}.t7'.format(configs.model_name, (epoch + 1) * num_train_batches))
        torch.save(model.state_dict(), save_path)
        filter_checkpoints(model_dir, suffix='t7', max_to_keep=1)
        # write file
        score_writer.write(
            f"save best_r1i7:{best_r1i7}, best_r1i5:{best_r1i5} best_dr1i7:{best_dr1i7}, best_dr1i5:{best_dr1i5} best_mIoU:{best_mIoU}\n"
            f"\tiid_best_r1i7:{iid_best_r1i7}, iid_best_r1i5:{iid_best_r1i5} iid_best_dr1i7:{iid_best_dr1i7}, iid_best_dr1i5:{iid_best_dr1i5} iid_best_mIoU:{iid_best_mIoU}\n")
        score_writer.flush()

if configs.save_num_counts:
    save_path = os.path.join(model_dir, 'num_counts.json')
    save_json(model_adapter.num_counts, save_path)

if configs.verbose !=0:
    score_writer.close()

    writer.close()

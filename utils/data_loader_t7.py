import os

import numpy as np
import torch
import torch.utils.data
from utils.data_util import pad_seq, pad_char_seq, pad_video_seq
from torch.nn import functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, vf):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.vf = vf

    def __getitem__(self, index):
        record = self.dataset[index]
        try:
            video_feature = self.video_features(record['vid'])
        except Exception as e:
            print(e)
            return self.__getitem__(index+1)
        s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
        word_ids, char_ids = record['w_ids'], record['c_ids']
        # is_bias = record['is_bias']
        is_bias = False
        query = " ".join(record['words'])
        return record, video_feature, word_ids, char_ids, s_ind, e_ind, is_bias, query

    def video_features(self, vid):
        file_path = os.path.join(os.path.join('data', 'features', 'charades', self.vf, vid+".npy"))
        if self.dataset[0]['vid'].startswith('v_'):
            file_path = os.path.join('../debias-vslnet-for-anet/data', 'features', 'activitynet', self.vf, vid+".npy")
            if self.vf == "i3d":
                file_path = os.path.join('../debias-vslnet-for-anet/data', 'features', 'activitynet', self.vf, vid[2:]+".npy")
            feature = np.load(file_path)
            feature = feature.reshape(feature.shape[0], feature.shape[-1])
        else:
            feature = np.load(file_path)
        return self.visual_feature_sampling(feature, max_num_clips=128)


    def visual_feature_sampling(self, visual_feature, max_num_clips):
        num_clips = visual_feature.shape[0]
        if num_clips <= max_num_clips:
            return visual_feature
        idxs = np.arange(0, max_num_clips + 1, 1.0) / max_num_clips * num_clips
        idxs = np.round(idxs).astype(np.int32)
        idxs[idxs > num_clips - 1] = num_clips - 1
        new_visual_feature = []
        for i in range(max_num_clips):
            s_idx, e_idx = idxs[i], idxs[i + 1]
            if s_idx < e_idx:
                new_visual_feature.append(np.mean(visual_feature[s_idx:e_idx], axis=0))
            else:
                new_visual_feature.append(visual_feature[s_idx])
        new_visual_feature = np.asarray(new_visual_feature)
        return new_visual_feature


    def __len__(self):
        return len(self.dataset)


def train_collate_fn(data):
    records, video_features, word_ids, char_ids, s_inds, e_inds, is_biases, querys = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    h_labels = np.zeros(shape=[batch_size, max_len], dtype=np.int32)
    extend = 0.1
    for idx in range(batch_size):
        st, et = s_inds[idx], e_inds[idx]
        cur_max_len = vfeat_lens[idx]
        extend_len = round(extend * float(et - st + 1))
        if extend_len > 0:
            st_ = max(0, st - extend_len)
            et_ = min(et + extend_len, cur_max_len - 1)
            h_labels[idx][st_:(et_ + 1)] = 1
        else:
            h_labels[idx][st:(et + 1)] = 1
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    is_biases = torch.tensor(is_biases, dtype=torch.int64)
    s_labels, e_labels = F.one_hot(s_labels, max_len), F.one_hot(e_labels, max_len)

    h_labels = torch.tensor(h_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels, is_biases, querys

def test_collate_fn(data):
    records, video_features, word_ids, char_ids, *_ = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)  # (batch_size, w_seq_len)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)  # (batch_size, w_seq_len, c_seq_len)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)  # (batch_size, v_seq_len, v_dim)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)  # (batch_size, )
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    return records, vfeats, vfeat_lens, word_ids, char_ids


def get_train_loader(dataset, configs):
    train_set = Dataset(dataset=dataset, vf=configs.vf)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn)
    return train_loader


def get_test_loader(dataset, configs):
    test_set = Dataset(dataset=dataset, vf=configs.vf)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=test_collate_fn)
    return test_loader

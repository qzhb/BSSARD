import torch
import numpy as np

class AdvSampleLoader3:
    def __init__(self, device):
        self.device = device
        self.start_probs = {}
        for length in range(1, 128+1):
            start_prob = []
            for j in range(length):
                start_prob.extend([j] * (length - j))
            start_prob = np.array(start_prob)
            self.start_probs[length] = start_prob
    def get_adv_sample(self, batch_size, vfeat_lens, video_mask):
        video_length = int(vfeat_lens.max().cpu())

        za = torch.randn(batch_size, 128, 1, 1, 1).to(self.device)
        zm = torch.randn(batch_size, 10, 4, 1, 1).to(self.device)
        zp = torch.zeros(batch_size, 128, 3).to(self.device)
        fake_h_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_s_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_e_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_video_mask = torch.ones(batch_size, video_length).to(self.device)

        zp[:, video_length:, 0] = 1
        # TODO:实现duration具有最小值，不能让长度为0
        start = torch.tensor(np.random.choice(self.start_probs[video_length]))
        duration = torch.randint(int(video_length - start), size=(1,))
        zp[:, :start, 1] = 1
        zp[:, start + duration:video_length, 1] = 1
        zp[:, start:start + duration, 2] = 1
        fake_h_labels[:, start:start + duration] = 1
        fake_s_labels[:, start] = 1
        fake_e_labels[:, start + duration - 1] = 1
        return za, zm, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels

class AdvSampleLoader3ForQ:
    def __init__(self, device):
        self.device = device
        self.start_probs = {}
        for length in range(1, 128+1):
            start_prob = []
            for j in range(length):
                start_prob.extend([j] * (length - j))
            start_prob = np.array(start_prob)
            self.start_probs[length] = start_prob
    def get_adv_sample(self, batch_size, vfeat_lens, video_mask):
        video_length = int(vfeat_lens.max().cpu())

        zc = torch.randn(batch_size, 128).to(self.device)
        zp = torch.zeros(batch_size, 128).to(self.device)
        fake_h_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_s_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_e_labels = torch.zeros(batch_size, video_length).to(self.device)
        fake_video_mask = torch.ones(batch_size, video_length).to(self.device)

        start = torch.tensor(np.random.choice(self.start_probs[video_length]))
        duration = torch.randint(int(video_length - start), size=(1,))
        zp[:, start:start + duration] = 1
        fake_h_labels[:, start:start + duration] = 1
        fake_s_labels[:, start] = 1
        fake_e_labels[:, start + duration - 1] = 1
        return zc, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels

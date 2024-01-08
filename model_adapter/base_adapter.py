import torch
from torch import nn

from model_adapter.model_adapter import ModelAdapter
from utils.runner_utils_t7 import convert_length_to_mask


class BaseModelAdapter(ModelAdapter):

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        _, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels, is_biases, _ = data

        vfeats, vfeat_lens = vfeats.to(self.device), vfeat_lens.to(self.device)
        word_ids, char_ids = word_ids.to(self.device), char_ids.to(self.device)
        s_labels, e_labels, h_labels, is_biases = s_labels.to(self.device), e_labels.to(self.device), h_labels.to(self.device), is_biases.to(self.device)
        # generate mask
        query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(self.device)
        video_mask = convert_length_to_mask(vfeat_lens).to(self.device)
        # compute logits
        h_score, start_logits, end_logits, bias_logits = self.model(word_ids, char_ids, vfeats, video_mask, query_mask)
        # compute loss
        # print(start_logits.argmax(1))
        highlight_loss = self.model.compute_highlight_loss(h_score, h_labels, video_mask)
        loc_loss = self.model.compute_loss(start_logits, end_logits, s_labels, e_labels)
        # dism_loss = self.temporal_order_discrimination_loss(bias_logits, is_biases)
        # loss = loc_loss + self.highlight_lambda * highlight_loss + dism_loss
        loss = loc_loss + self.highlight_lambda * highlight_loss
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.scheduler.step()

        return highlight_loss, loc_loss, highlight_loss, loc_loss, highlight_loss, loc_loss, \
        loc_loss, highlight_loss, loc_loss, highlight_loss, loc_loss,

    def eval_one_step(self, data):
        records, vfeats, vfeat_lens, word_ids, char_ids = data
        # prepare features
        vfeats, vfeat_lens = vfeats.to(self.device), vfeat_lens.to(self.device)
        word_ids, char_ids = word_ids.to(self.device), char_ids.to(self.device)
        # generate mask
        query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(self.device)
        video_mask = convert_length_to_mask(vfeat_lens).to(self.device)
        # compute logits
        h_score, start_logits, end_logits, _ = self.model(word_ids, char_ids, vfeats, video_mask, query_mask, None)
        return h_score, start_logits, end_logits

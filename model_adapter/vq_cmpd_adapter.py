import random

import torch
from torch import nn

from model_adapter.model_adapter import ModelAdapter
from utils.runner_utils_t7 import convert_length_to_mask


class VQCmpdAdapter(ModelAdapter):
    def __init__(self, model, generator, g_optimizer, g_scheduler, train_loader, optimizer, scheduler, advSampleLoader
                 , batch_size, writer, total_epoch, highlight_lambda, clip_norm, device, verbose=False, m=1,
                 loss_type="33",
                 train_strategy="alter", compound_gen=False):

        super().__init__(model, generator, g_optimizer, g_scheduler, train_loader, optimizer, scheduler
                         , batch_size, writer, total_epoch, highlight_lambda, clip_norm, device, verbose, m, loss_type,
                         train_strategy, compound_gen)

        self.index2type = {
            0: 'v',
            1: 'q',
        }
        assert self.index2type.__len__() == len(self.generator)

    def train_one_step(self, data):
        assert self.compound_gen is True

        # 0. Prepare
        _, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels, zeros, querys = data
        self.batch_size = vfeats.shape[0]
        vfeats, vfeat_lens = vfeats.to(self.device), vfeat_lens.to(self.device)
        word_ids, char_ids = word_ids.to(self.device), char_ids.to(self.device)
        s_labels, e_labels, h_labels, zeros = s_labels.to(self.device), e_labels.to(self.device), h_labels.to(
            self.device), zeros.to(self.device)
        # generate mask
        query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(self.device)
        video_mask = convert_length_to_mask(vfeat_lens).to(self.device)
        ones = 1 - zeros

        gen_index = 0
        # 1. Train Dism and TSGV
        # get adversarial samples
        fake_video, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels = self.get_video_adv_sample_cmpd(
            batch_size=self.batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask, querys=querys,
            type=self.index2type[gen_index], qfeat_len=word_ids.shape[1])
        # compute logits
        h_score, start_logits, end_logits, bias_logits = self.model(word_ids, char_ids, vfeats, video_mask,
                                                                    query_mask, None)
        # compute fake logits
        fake_h_score, fake_start_logits, fake_end_logits, fake_bias_logits = self.model(
            word_ids, char_ids, vfeats, fake_video_mask, query_mask, fake_video[:, :vfeats.shape[1], :])
        self.optimizer.zero_grad()
        # compute loss
        highlight_loss = self.model.compute_highlight_loss(h_score, h_labels, video_mask)
        loc_loss = self.model.compute_loss(start_logits, end_logits, s_labels, e_labels)
        dism_loss = self.temporal_order_discrimination_loss(bias_logits, zeros)

        loc_loss2 = self.model.compute_loss(fake_start_logits, fake_end_logits, s_labels, e_labels)
        highlight_loss2 = self.model.compute_highlight_loss(fake_h_score, h_labels, fake_video_mask)
        dism_loss2 = self.temporal_order_discrimination_loss(fake_bias_logits, ones)

        h_match_loss = self.KL_divergence_highlight(fake_h_score, h_score, fake_video_mask)
        loc_match_loss = self.KL_divergence_loc(fake_start_logits, start_logits, fake_video_mask) + \
                         self.KL_divergence_loc(fake_end_logits, end_logits, fake_video_mask)

        loss = loc_loss + self.highlight_lambda * highlight_loss \
               + loc_loss2 + highlight_loss2
        loss = loss + loc_match_loss + dism_loss + dism_loss2

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        # self.scheduler.step()

        # 2. Train Gen
        # get adversarial samples
        fake_video, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels = self.get_video_adv_sample_cmpd(
            batch_size=self.batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask, querys=querys,
            type=self.index2type[gen_index], qfeat_len=word_ids.shape[1])
        # compute logits
        fake_h_score, fake_start_logits, fake_end_logits, fake_bias_logits = self.model(
            word_ids, char_ids, vfeats, fake_video_mask, query_mask, fake_video[:, :vfeats.shape[1], :])
        self.g_optimizer[gen_index].zero_grad()
        # compute loss
        fake_highlight_loss = self.model.compute_highlight_loss(fake_h_score, fake_h_labels,
                                                                fake_video_mask)
        fake_loc_loss = self.model.compute_loss(fake_start_logits, fake_end_logits, fake_s_labels,
                                                fake_e_labels)
        fake_dism_loss = self.temporal_order_discrimination_loss(fake_bias_logits, zeros)
        loss = fake_loc_loss + self.highlight_lambda * fake_highlight_loss
        loss = loss + fake_dism_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator[gen_index].parameters(), self.clip_norm)
        self.g_optimizer[gen_index].step()
        self.g_scheduler[gen_index].step()

        gen_index = 1
        # 1. Train Dism and TSGV
        # get adversarial samples
        fake_query, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels = self.get_video_adv_sample_cmpd(
            batch_size=self.batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask, querys=querys,
            type=self.index2type[gen_index], qfeat_len=word_ids.shape[1])
        # compute logits
        h_score, start_logits, end_logits, bias_logits = self.model(word_ids, char_ids, vfeats, video_mask,
                                                                    query_mask, None, None)
        # compute fake logits
        fake_h_score, fake_start_logits, fake_end_logits, fake_bias_logits = self.model(
            word_ids, char_ids, vfeats, fake_video_mask, query_mask, None, fake_query[:, :word_ids.shape[1], :])
        self.optimizer.zero_grad()
        # compute loss
        highlight_loss = self.model.compute_highlight_loss(h_score, h_labels, video_mask)
        loc_loss = self.model.compute_loss(start_logits, end_logits, s_labels, e_labels)
        dism_loss = self.temporal_order_discrimination_loss(bias_logits, zeros)

        loc_loss2 = self.model.compute_loss(fake_start_logits, fake_end_logits, s_labels, e_labels)
        highlight_loss2 = self.model.compute_highlight_loss(fake_h_score, h_labels, fake_video_mask)
        dism_loss2 = self.temporal_order_discrimination_loss(fake_bias_logits, ones)

        h_match_loss = self.KL_divergence_highlight(fake_h_score, h_score, fake_video_mask)
        loc_match_loss = self.KL_divergence_loc(fake_start_logits, start_logits, fake_video_mask) + \
                         self.KL_divergence_loc(fake_end_logits, end_logits, fake_video_mask)

        loss = loc_loss + self.highlight_lambda * highlight_loss \
               + loc_loss2 + highlight_loss2
        loss = loss + loc_match_loss + dism_loss + dism_loss2

        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        self.scheduler.step()

        # 2. Train Gen
        fake_query, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels = self.get_video_adv_sample_cmpd(
            batch_size=self.batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask, querys=querys,
            type=self.index2type[gen_index], qfeat_len=word_ids.shape[1])
        # compute logits
        fake_h_score, fake_start_logits, fake_end_logits, fake_bias_logits = self.model(
            word_ids, char_ids, vfeats, fake_video_mask, query_mask, None, fake_query[:, :word_ids.shape[1], :])
        self.g_optimizer[gen_index].zero_grad()
        # compute loss
        fake_highlight_loss = self.model.compute_highlight_loss(fake_h_score, fake_h_labels,
                                                                fake_video_mask)
        fake_loc_loss = self.model.compute_loss(fake_start_logits, fake_end_logits, fake_s_labels,
                                                fake_e_labels)
        fake_dism_loss = self.temporal_order_discrimination_loss(fake_bias_logits, zeros)
        # print(fake_loc_loss, fake_highlight_loss, fake_dism_loss)
        loss = fake_loc_loss + self.highlight_lambda * fake_highlight_loss
        if self.loss_type.startswith("33"):
            loss = loss + fake_dism_loss
        #loss = fake_dism_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.generator[gen_index].parameters(), self.clip_norm)
        self.g_optimizer[gen_index].step()
        self.g_scheduler[gen_index].step()

        return highlight_loss, loc_loss, dism_loss, highlight_loss2, loc_loss2, dism_loss2, \
                h_match_loss, loc_match_loss, fake_highlight_loss, fake_loc_loss, fake_dism_loss

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

import time

import numpy as np
import torch
from tqdm import tqdm

from adv_sample_loader.AdvSampleLoader import AdvSampleLoader3ForQ, AdvSampleLoader3
from utils.data_util import index_to_time
from utils.runner_utils_t7 import calculate_iou, calculate_iou_accuracy, calculate_diou_accuracy, convert_length_to_mask
from utils.utils import AverageMeter
import json
import torch.nn.functional as F

class ModelAdapter:

    def __init__(self, model, generator, g_optimizer, g_scheduler,train_loader, optimizer, scheduler,
                 batch_size, writer, total_epoch, highlight_lambda, clip_norm, device, verbose=False, m=1, loss_type="33",
                 train_strategy="alter", compound_gen=False):
        self.model = model
        self.compound_gen = compound_gen

        self.generator = generator
        self.g_optimizer = g_optimizer
        self.g_scheduler = g_scheduler
        if type(self.generator) is list:
            for g in self.generator:
                tp = str(type(g)).split('.')[-2].split('_')[0]
                if tp == 'z2v':
                    self.v_generator = g
                elif tp == 'z2q':
                    self.q_generator = g
        self.q_advSampleLoader = AdvSampleLoader3ForQ(device)
        self.v_advSampleLoader = AdvSampleLoader3(device)

        self.train_loader = train_loader
        self.num_train_batches = len(train_loader)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.highlight_lambda = highlight_lambda
        self.clip_norm = clip_norm
        self.batch_size = batch_size
        self.writer = writer
        self.total_epoch = total_epoch
        self.device = device
        self.verbose = verbose
        self.epoch = 0
        self.global_step = 0
        self.m = m
        self.loss_type = loss_type
        self.train_strategy = train_strategy
        self.criterion_domain = torch.nn.CrossEntropyLoss().to(self.device)


    def temporal_order_discrimination_loss(self, prob, gt):
        loss_single = self.criterion_domain(prob, gt)
        return loss_single

    def KL_divergence_highlight(self, prob1, prob2, video_mask, epsilon=1e-4):
        KL = prob1 * torch.log((prob1 + epsilon) / (prob2 + epsilon)) +\
             (1 - prob1) * torch.log(((1 - prob1) + epsilon) / ((1 - prob2) + epsilon))
        loss = torch.mean(KL)
        return loss

    def KL_divergence_loc(self, prob1, prob2, video_mask):
        prob1 = F.log_softmax(prob1, dim=-1)
        prob2 = F.softmax(prob2, dim=-1)
        KL = F.kl_div(prob1, prob2, reduction='sum')
        loss = KL / prob1.shape[0]
        return loss

    def masked_softmax(self, vec, mask, dim=1, epsilon=1e-4):
        exps = torch.exp(vec)
        masked_exps = exps * mask.float()
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return (masked_exps / masked_sums)

    def get_video_adv_sample(self, batch_size, vfeat_lens, video_mask, querys):

        za, zm, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels \
            = self.v_advSampleLoader.get_adv_sample(batch_size=batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask)
        fake_video = self.generator(za, zm, zp)
        return fake_video, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels

    def get_video_adv_sample_cmpd(self, batch_size, vfeat_lens, video_mask, querys, type, qfeat_len):
        if type == 'v':
            za, zm, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels \
                = self.v_advSampleLoader.get_adv_sample(batch_size=batch_size, vfeat_lens=vfeat_lens,
                                                      video_mask=video_mask)
            fake_video = self.v_generator(za, zm, zp)
            return fake_video, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels
        elif type == 'q':
            zc, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels \
                = self.q_advSampleLoader.get_adv_sample(batch_size=batch_size, vfeat_lens=vfeat_lens,
                                                        video_mask=video_mask)
            fake_query = self.q_generator(zc, zp)
            return fake_query, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels
        else:
            raise Exception("错误的type")

    def get_query_adv_sample(self, batch_size, vfeat_lens, qfeat_len, video_mask):
        zc, zp, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels \
            = self.q_advSampleLoader.get_adv_sample(batch_size=batch_size, vfeat_lens=vfeat_lens, video_mask=video_mask)
        fake_query = self.generator(zc, zp)
        return fake_query, fake_video_mask, fake_s_labels, fake_e_labels, fake_h_labels

    def train_one_epoch_template(self):
        self.epoch += 1
        if self.verbose != 0:
            # 记录每个epoch的平均时间和 loss
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end_time = time.time()
        for data in tqdm(self.train_loader, total=self.num_train_batches, desc='Epoch %3d / %3d' % (self.epoch, self.total_epoch)):
            if self.verbose != 0:
                data_time.update(time.time() - end_time)
            self.global_step += 1

            # 调用不同模型的训练策略
            highlight_loss, loc_loss, dism_loss, highlight_loss2, loc_loss2, dism_loss2,\
            h_match_loss, loc_match_loss, fake_highlight_loss, fake_loc_loss, fake_dism_loss = self.train_one_step(data)

            if self.verbose != 0:
                # 更新统计信息
                total_loss = highlight_loss + loc_loss
                losses.update(total_loss.item(), self.batch_size)
                batch_time.update(time.time() - end_time)
                # tensorboard记录loss值
                self.writer.add_scalar('tsgv_loss1/highlight_loss', highlight_loss, self.global_step)
                self.writer.add_scalar('tsgv_loss1/loc_loss', loc_loss, self.global_step)
                self.writer.add_scalar('tsgv_loss1/dism_loss', dism_loss, self.global_step)
                self.writer.add_scalar('tsgv_loss2/highlight_loss2', highlight_loss2, self.global_step)
                self.writer.add_scalar('tsgv_loss2/loc_loss2', loc_loss2, self.global_step)
                self.writer.add_scalar('tsgv_loss2/dism_loss2', dism_loss2, self.global_step)
                self.writer.add_scalar('tsgv_loss3/h_match_loss', h_match_loss, self.global_step)
                self.writer.add_scalar('tsgv_loss3/loc_match_loss', loc_match_loss, self.global_step)
                # tensorboard记录fake_loss值
                self.writer.add_scalar('gen_loss/fake_highlight_loss', fake_highlight_loss, self.global_step)
                self.writer.add_scalar('gen_loss/fake_loc_loss', fake_loc_loss, self.global_step)
                self.writer.add_scalar('gen_loss/fake_dism_loss', fake_dism_loss, self.global_step)
        if self.verbose != 0:
            self.writer.add_scalar('train_loss1/loss_epoch', losses.avg, self.epoch)
            self.writer.add_scalar('train_info/batch_time', batch_time.avg, self.epoch)
            self.writer.add_scalar('train_info/data_time', data_time.avg, self.epoch)
            self.writer.add_scalar('train_info/group1_lr', self.optimizer.param_groups[0]['lr'], self.epoch)
            self.writer.add_scalar('train_info/group2_lr', self.optimizer.param_groups[1]['lr'], self.epoch)

        return self.global_step

    def test_one_epoch_template(self, test_loader, scope):
        ious = []
        deltas = []
        with torch.no_grad():
            for idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='evaluate {}'.format(scope)):
                records, vfeats, vfeat_lens, word_ids, char_ids = data
                h_score, start_logits, end_logits = self.eval_one_step(data)

                start_indices, end_indices = self.model.extract_index(start_logits, end_logits)
                start_indices = start_indices.cpu().numpy()
                end_indices = end_indices.cpu().numpy()
                vfeat_lens = vfeat_lens.cpu().numpy()
                for record, start_index, end_index, vfeat_len in zip(records, start_indices, end_indices, vfeat_lens):
                    start_time, end_time = index_to_time(start_index, end_index, vfeat_len, record["duration"])
                    iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
                    delta_s = 1 - (abs(start_time - record["s_time"]) / record['duration'])
                    delta_e = 1 - (abs(end_time - record["e_time"]) / record['duration'])
                    deltas.append((delta_s, delta_e))
                    ious.append(iou)
        r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        dr1i3 = calculate_diou_accuracy(ious, threshold=0.3, deltas=deltas)
        dr1i5 = calculate_diou_accuracy(ious, threshold=0.5, deltas=deltas)
        dr1i7 = calculate_diou_accuracy(ious, threshold=0.7, deltas=deltas)
        mi = np.mean(ious) * 100.0
        # write the scores
        score_str = "Epoch {}, Step {}:\t".format(self.epoch, self.global_step)
        score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
        score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
        score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
        score_str += "dRank@1, IoU=0.3: {:.2f}\t".format(dr1i3)
        score_str += "dRank@1, IoU=0.5: {:.2f}\t".format(dr1i5)
        score_str += "dRank@1, IoU=0.7: {:.2f}\t".format(dr1i7)
        score_str += "mean IoU: {:.2f}\t".format(mi)
        return r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mi, score_str

    def test_one_epoch_template_to_visual(self, train_loader, scope):
        ious = []
        deltas = []
        results = {}
        with torch.no_grad():
            for idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='evaluate {}'.format(scope)):
                records, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, h_labels, zeros, querys = data
                # records, vfeats, vfeat_lens, word_ids, char_ids = data
                h_score, start_logits, end_logits = self.eval_one_step(data[:5])
                start_indices, end_indices = self.model.extract_index(start_logits, end_logits)
                start_indices = start_indices.cpu().numpy()
                end_indices = end_indices.cpu().numpy()
                vfeat_lens = vfeat_lens.cpu().numpy()
                for record, start_index, end_index, vfeat_len in zip(records, start_indices, end_indices, vfeat_lens):
                    start_time, end_time = index_to_time(start_index, end_index, vfeat_len, record["duration"])
                    iou = calculate_iou(i0=[start_time, end_time], i1=[record["s_time"], record["e_time"]])
                    delta_s = 1 - (abs(start_time - record["s_time"]) / record['duration'])
                    delta_e = 1 - (abs(end_time - record["e_time"]) / record['duration'])
                    deltas.append((delta_s, delta_e))
                    ious.append(iou)

                    result = {"iou": str(iou), "pred": (str(start_time), str(end_time)), "record": record}
                    results[record['vid']] = result
                    ious.append(iou)

                start_logits = torch.softmax(start_logits, 1)
                end_logits = torch.softmax(end_logits, 1)
                for record, start_logit, end_logit, vfeat_len in zip(records, start_logits, end_logits, vfeat_lens):
                    bias_s_out = start_logit[record["s_ind"]].cpu()
                    bias_e_out = end_logit[record["e_ind"]].cpu()
                    results[record['vid']]['bias_s_out'] = str(float(bias_s_out))
                    results[record['vid']]['bias_e_out'] = str(float(bias_e_out))
        # with open(f'./visualization/{self.model.__class__.__name__}_Output.json', 'w+') as file:
        #     file.write(json.dumps(results))
        r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
        r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
        r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
        dr1i3 = calculate_diou_accuracy(ious, threshold=0.3, deltas=deltas)
        dr1i5 = calculate_diou_accuracy(ious, threshold=0.5, deltas=deltas)
        dr1i7 = calculate_diou_accuracy(ious, threshold=0.7, deltas=deltas)
        mi = np.mean(ious) * 100.0
        # write the scores
        score_str = "Epoch {}, Step {}:\t".format(self.epoch, self.global_step)
        score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
        score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
        score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
        score_str += "dRank@1, IoU=0.3: {:.2f}\t".format(dr1i3)
        score_str += "dRank@1, IoU=0.5: {:.2f}\t".format(dr1i5)
        score_str += "dRank@1, IoU=0.7: {:.2f}\t".format(dr1i7)
        score_str += "mean IoU: {:.2f}\t".format(mi)
        return r1i3, r1i5, r1i7, dr1i3, dr1i5, dr1i7, mi, score_str


    def train_one_step(self, data):
        raise Exception("please implement this method in the subclass")

    def eval_one_step(self, data):
        raise Exception("please implement this method in the subclass")


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F
from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        pad_mask = pad_mask.unsqueeze(dim=2)
        pad_mask = pad_mask.to(torch.bool)
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # TODO: 对所有token求一个均值，使kl-loss的量级与CE loss相同
    p_loss_manual_calc = p_loss.sum(dim=-1).mean()
    q_loss_manual_calc = q_loss.sum(dim=-1).mean()

    loss = (p_loss_manual_calc + q_loss_manual_calc) / 2
    return loss


class Model(PushToHubFriendlyModel):
    def __init__(self, args, training_args):
        super().__init__()
        self.args = args
        self.training_args = training_args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert.location, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.bert.location,
        )
        self.config = self.pretrain_model.config

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs):
        # TODO: Commented by Zhiyu: Change here for computing the combined loss for R-Drop Consistency Training
        reg_alpha = self.training_args.consistency_alpha
        if self.training is True:
            if self.training_args.use_extra_question is False and self.training_args.do_consistency_training is False and self.training_args.use_shuffled_table is False:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                labels = inputs['labels']
                loss = self.pretrain_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    labels=labels,
                ).loss

            elif self.training_args.use_extra_question is True and self.training_args.do_consistency_training is False:
                input_ids_1_2 = inputs['input_ids']
                attention_mask_1_2 = inputs['attention_mask']
                labels = inputs['labels']
                loss_1 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 0:1, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                ).loss

                loss_2 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 1:, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 1:, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                ).loss

                loss = (loss_1 + loss_2) / 2

            elif self.training_args.use_extra_question is True and self.training_args.do_consistency_training is True:
                input_ids_1_2 = inputs['input_ids']
                attention_mask_1_2 = inputs['attention_mask']
                labels = inputs['labels']
                output_1 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 0:1, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )

                output_2 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 1:, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 1:, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )

                # TODO: Commented by Zhiyu: Implement the computation of the combined loss for R-Drop Consistency Training here

                logits_1 = output_1.logits
                logits_2 = output_2.logits
                ce_loss_1 = output_1.loss
                ce_loss_2 = output_2.loss

                # cross entropy loss for classifier
                ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)

                #########################################################################################
                # TODO: Implement pad_mask to filter out unwanted loss calculation in padding positions #
                #########################################################################################
                targets = labels
                pad_mask = torch.zeros(targets.size())
                pad_mask[targets == -100] = 1
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pad_mask = pad_mask.to(device)

                kl_loss = compute_kl_loss(logits_1, logits_2, pad_mask=pad_mask)

                # carefully choose hyper-parameters
                loss = ce_loss + reg_alpha * kl_loss

            elif self.training_args.use_shuffled_table is True and self.training_args.do_consistency_training is False:
                input_ids_1_2 = inputs['input_ids']
                attention_mask_1_2 = inputs['attention_mask']
                labels = inputs['labels']
                loss_1 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 0:1, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                ).loss

                loss_2 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 1:, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 1:, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                ).loss

                loss = (loss_1 + loss_2) / 2

            elif self.training_args.use_shuffled_table is True and self.training_args.do_consistency_training is True:
                input_ids_1_2 = inputs['input_ids']
                attention_mask_1_2 = inputs['attention_mask']
                labels = inputs['labels']
                output_ori_1 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 0:1, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )
                output_ori_2 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 0:1, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )

                output_st_1 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 1:, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 1:, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )
                output_st_2 = self.pretrain_model(
                    input_ids=input_ids_1_2[:, 1:, :].squeeze(dim=1),
                    attention_mask=attention_mask_1_2[:, 1:, :].squeeze(dim=1),
                    use_cache=False,
                    labels=labels,
                )

                logits_ori_1 = output_ori_1.logits
                logits_ori_2 = output_ori_2.logits
                logits_st_1 = output_st_1.logits
                logits_st_2 = output_st_2.logits
                ce_loss_ori = output_ori_1.loss
                ce_loss_st = output_st_1.loss

                # cross entropy loss for classifier
                ce_loss = 0.5 * ce_loss_ori + 0.5 * ce_loss_st

                targets = labels
                pad_mask = torch.zeros(targets.size())
                pad_mask[targets == -100] = 1
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pad_mask = pad_mask.to(device)

                kl_loss_ori_st = compute_kl_loss(logits_ori_1, logits_st_1, pad_mask=pad_mask)
                kl_loss_ori = compute_kl_loss(logits_ori_1, logits_ori_2, pad_mask=pad_mask)
                kl_loss_st = compute_kl_loss(logits_st_1, logits_st_2, pad_mask=pad_mask)

                # carefully choose hyper-parameters
                loss = ce_loss + reg_alpha * kl_loss_ori_st + 2. * kl_loss_ori + 2. * kl_loss_st

            else:  # use_extra_question=False, do_consistency_training=True
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                labels = inputs['labels']
                output_1 = self.pretrain_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    labels=labels,
                )

                output_2 = self.pretrain_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    labels=labels,
                )

                logits_1 = output_1.logits
                logits_2 = output_2.logits
                ce_loss_1 = output_1.loss
                ce_loss_2 = output_2.loss

                # cross entropy loss for classifier
                ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)

                targets = labels
                pad_mask = torch.zeros(targets.size())
                pad_mask[targets == -100] = 1
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                pad_mask = pad_mask.to(device)

                kl_loss = compute_kl_loss(logits_1, logits_2, pad_mask=pad_mask)

                # carefully choose hyper-parameters
                loss = ce_loss + reg_alpha * kl_loss

        else:
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            loss = self.pretrain_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                labels=labels,
            ).loss

        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids

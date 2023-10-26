#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import torch
from torch.nn import functional as F, CrossEntropyLoss

from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration


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
        # self.lm_head_q = torch.nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head_r = torch.nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        # self.lm_head_q = copy.deepcopy(self.pretrain_model.lm_head)
        self.lm_head_r = copy.deepcopy(self.pretrain_model.lm_head)

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs):
        reg_alpha = self.training_args.consistency_alpha
        rationale_beta = self.training_args.rationale_beta
        if self.training is True:
            rationale = inputs['rationale']
            labels = inputs['labels']

            decoder_output_1 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 0:1, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 0:1, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=labels
            )

            '''decoder_output_2 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 1:, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 1:, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=labels
            )'''

            decoder_output_3 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 0:1, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 0:1, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=rationale
            )

            loss_t1 = decoder_output_1["loss"]
            # loss_t2 = decoder_output_2["loss"]
            t1_logits = decoder_output_1["logits"]
            # t2_logits = decoder_output_2["logits"]

            if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            # t1_logits = decoder_output_1['decoder_hidden_states'][-1] * (self.pretrain_model.model_dim ** -0.5)
            # t2_logits = decoder_output_2['decoder_hidden_states'][-1] * (self.pretrain_model.model_dim ** -0.5)
                rationale_logits = decoder_output_3['decoder_hidden_states'][-1] * (self.pretrain_model.model_dim ** -0.5)

            else:
            # t1_logits = decoder_output_1['decoder_hidden_states'][-1]
            # t2_logits = decoder_output_2['decoder_hidden_states'][-1]
                rationale_logits = decoder_output_3['decoder_hidden_states'][-1]

            # t1_logits = self.lm_head_q(t1_logits)
            # t2_logits = self.lm_head_q(t2_logits)
            rationale_logits = self.lm_head_r(rationale_logits)

            # loss = None
            # if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            # labels = labels.to(t1_logits.device)
            # loss_t1 = loss_fct(t1_logits.view(-1, t1_logits.size(-1)), labels.view(-1))
            # loss_t2 = loss_fct(t2_logits.view(-1, t2_logits.size(-1)), labels.view(-1))

            # else:
            # raise AssertionError("No labels are found")

            if rationale is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable PP
                rationale = rationale.to(rationale_logits.device)
                loss_rationale = loss_fct(rationale_logits.view(-1, rationale_logits.size(-1)), rationale.view(-1))
            else:
                raise AssertionError("No rationale are found")

            # TODO: Commented by Zhiyu: Implement the computation of the combined loss for R-Drop Consistency Training here
            # cross entropy loss for classifier

            # ce_loss = 0.5 * (loss_t1 + loss_t2)
            ce_loss = loss_t1

            #########################################################################################
            # TODO: Implement pad_mask to filter out unwanted loss calculation in padding positions #
            #########################################################################################
            '''pad_mask = torch.zeros(labels.size())
            pad_mask[labels == -100] = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pad_mask = pad_mask.to(device)

            kl_loss = compute_kl_loss(t1_logits, t2_logits, pad_mask=pad_mask)'''

            # carefully choose hyper-parameters
            # loss = ce_loss + reg_alpha * kl_loss + rationale_beta * loss_rationale
            loss = ce_loss + rationale_beta * loss_rationale

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

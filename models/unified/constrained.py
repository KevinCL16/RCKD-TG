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


def init_froze_model(pretrain_model, tokenizer, special_tokens):
    global pretrain_model_froze
    pretrain_model_froze = copy.deepcopy(pretrain_model)
    tokenizer.add_tokens([v for k, v in special_tokens])
    pretrain_model_froze.resize_token_embeddings(len(tokenizer))


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
        init_froze_model(self.pretrain_model, self.tokenizer, args.special_tokens)
        self.config = self.pretrain_model.config
        self.lm_head_r = torch.nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        self.lm_head_r = copy.deepcopy(self.pretrain_model.lm_head)

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs):
        reg_alpha = self.training_args.consistency_alpha
        rationale_beta = self.training_args.rationale_beta
        if self.training is True:
            labels = inputs['labels']
            relevant_cells = inputs['relevant_cells']
            pretrain_model_froze.eval()
            pretrain_model_froze.to('cuda')
            with torch.no_grad():
                decoder_output_base = pretrain_model_froze(
                    input_ids=inputs['input_ids'][:, 0:1, :].squeeze(dim=1),
                    attention_mask=inputs['attention_mask'][:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    output_hidden_states=True,
                    labels=labels
                )
                logits_base = decoder_output_base["logits"]

            self.pretrain_model.eval()
            with torch.no_grad():
                decoder_output_1 = self.pretrain_model(
                    input_ids=inputs['input_ids'][:, 0:1, :].squeeze(dim=1),
                    attention_mask=inputs['attention_mask'][:, 0:1, :].squeeze(dim=1),
                    use_cache=False,
                    output_hidden_states=True,
                    labels=labels
                )
                logits_d1 = decoder_output_1["logits"]

            self.pretrain_model.train()
            decoder_output_2 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 1:2, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 1:2, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=relevant_cells
            )

            loss_d2 = decoder_output_2["loss"]

            # TODO: cross entropy loss for classifier
            # ce_loss = 0.5 * (loss_t1 + loss_t2)
            ce_loss = loss_d2

            #########################################################################################
            # TODO: Implement pad_mask to filter out unwanted loss calculation in padding positions #
            #########################################################################################
            pad_mask = torch.zeros(labels.size())
            pad_mask[labels == -100] = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pad_mask = pad_mask.to(device)

            kl_loss = compute_kl_loss(logits_base, logits_d1, pad_mask=pad_mask)

            # TODO: carefully choose hyper-parameters
            # loss = ce_loss + reg_alpha * kl_loss + rationale_beta * loss_rationale
            loss = ce_loss + 1. * kl_loss

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

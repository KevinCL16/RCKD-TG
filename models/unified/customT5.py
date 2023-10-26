#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import torch
from torch.nn import functional as F, CrossEntropyLoss

from .base import PushToHubFriendlyModel, MyT5ForConditionalGeneration
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
        self.pretrain_model = MyT5ForConditionalGeneration.from_pretrained(
            args.bert.location,
        )
        self.config = self.pretrain_model.config
        # self.pretrain_model.lm_head_r = copy.deepcopy(self.pretrain_model.lm_head)
        self.pretrain_model.lm_head_hc = copy.deepcopy(self.pretrain_model.lm_head)
        self.pretrain_model.decoder_2 = copy.deepcopy(self.pretrain_model.decoder)

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, **inputs):
        reg_alpha = self.training_args.consistency_alpha
        rationale_beta = self.training_args.rationale_beta
        if self.training is True:
            relevant_cells = inputs['relevant_cells']
            labels = inputs['labels']

            '''decoder_output_3 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 1:, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 1:, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=rationale,
                use_lm_head='r'
            )'''

            decoder_output_1 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 0:1, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 0:1, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                answer_labels=labels,
                relevant_cell_labels=relevant_cells
            )

            '''decoder_output_2 = self.pretrain_model(
                input_ids=inputs['input_ids'][:, 1:, :].squeeze(dim=1),
                attention_mask=inputs['attention_mask'][:, 1:, :].squeeze(dim=1),
                use_cache=False,
                output_hidden_states=True,
                labels=labels,
                use_lm_head='q'
            )'''

            loss_d1 = decoder_output_1["loss"]
            # loss_t2 = decoder_output_2["loss"]
            # t1_logits = decoder_output_1["logits"]
            # t2_logits = decoder_output_2["logits"]
            # loss_d3 = decoder_output_3["loss"]

            # if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                # rationale_logits = decoder_output_3['decoder_hidden_states'][-1] * (self.pretrain_model.model_dim ** -0.5)

            # else:
                # rationale_logits = decoder_output_3['decoder_hidden_states'][-1]

            # rationale_logits = self.lm_head_r(rationale_logits)

            '''if rationale is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # move labels to correct device to enable PP
                rationale = rationale.to(rationale_logits.device)
                loss_rationale = loss_fct(rationale_logits.view(-1, rationale_logits.size(-1)), rationale.view(-1))
            else:
                raise AssertionError("No rationale are found")'''

            # TODO: cross entropy loss for classifier
            # ce_loss = 0.5 * (loss_t1 + loss_t2)
            ce_loss = loss_d1

            #########################################################################################
            # TODO: Implement pad_mask to filter out unwanted loss calculation in padding positions #
            #########################################################################################
            '''pad_mask = torch.zeros(labels.size())
            pad_mask[labels == -100] = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pad_mask = pad_mask.to(device)

            kl_loss = compute_kl_loss(t1_logits, t2_logits, pad_mask=pad_mask)'''

            # TODO: carefully choose hyper-parameters
            # loss = ce_loss + reg_alpha * kl_loss + rationale_beta * loss_rationale
            loss = ce_loss

        else:
            input_ids = inputs['input_ids'].to("cuda")
            attention_mask = inputs['attention_mask'].to("cuda")
            labels = inputs['labels'].to("cuda")
            loss = self.pretrain_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                answer_labels=labels,
                use_lm_head='test'
            ).loss

        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            use_lm_head='test',
            **kwargs,
        )

        '''input_ids = torch.cat((input_ids, generated_ids), dim=1)
        ones = torch.ones(generated_ids.size()).to("cuda")
        attention_mask = torch.cat((attention_mask, ones), dim=1)

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            use_lm_head='test',
            **kwargs,
        )'''

        return generated_ids

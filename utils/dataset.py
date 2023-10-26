import os
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][
                                                                                  index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    if self.args.model.name == 'unified.flan' or self.args.model.name == 'unified.constrained':
                        '''seq_in = "Answer the question given the table below. {} ; Table: {}".\
                                                    format(raw_item["text_in"], raw_item["struct_in"])
                        seq_in_1_t2 = "Answer the question given the perturbed table below. {} ; Perturbed table: {}".\
                                                    format(raw_item["text_in"], raw_item["struct_in_shuffled"])
                        seq_in_for_rationale = "Give the rationale before answering the question given the table below." \
                                                                       " {} ; Table: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        rationale = "Rationale: {}.".format(raw_item["rationale"])'''

                        seq_in = "[Answer] {} ; Table: {}; {}".format(raw_item["text_in"], raw_item["struct_in"], raw_item["relevant_cells"])
                        seq_in_for_hc = "[Relevant Cells] {} ; Table: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        # seq_in_for_rationale = "[Rationale] {} ; Table: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        # rationale = "{}.".format(raw_item["rationale"])
                        relevant_cells = "{}.".format((raw_item["relevant_cells"]))

                    elif self.args.model.name == 'unified.customT5':
                        seq_in = "Output relevant cell information and answer the question: {} ; Table: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        seq_in_for_hc = "[Relevant Cells] {} ; Table: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        relevant_cells = "Relevant cell information for answering the question: {}.".format((raw_item["relevant_cells"]))

                    elif self.args.model.name == 'unified.rationale_mtl':
                        seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        seq_in_1_t2 = "{} ; structured knowledge: {}".format(raw_item["text_in"],
                                                                             raw_item["struct_in_shuffled"])
                        if raw_item["rationale"] is not None:
                            rationale = "solving the question step by step: {}".format(raw_item["rationale"])
                        else:
                            rationale = None

                    else:
                        if self.training_args.use_shuffled_table is False and self.training_args.use_extra_question is False:
                            seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                        elif self.training_args.use_shuffled_table is False and self.training_args.use_extra_question is True:
                            seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                            seq_in_q2 = "{} ; structured knowledge: {}".format(raw_item["question_two"],
                                                                               raw_item["struct_in"])
                        elif self.training_args.use_shuffled_table is True and self.training_args.use_extra_question is False:
                            seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                            seq_in_t2 = "{} ; structured knowledge: {}".format(raw_item["text_in"],
                                                                               raw_item["struct_in_shuffled"])
                        else:
                            raise AssertionError(
                                "only one customized data format can be used. Combination of both not supported yet")

                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        # TODO: Commented by Zhiyu: Tokenize extra data here.
        ####################################################
        # TODO: Rationale MTL                              #
        ####################################################
        if self.args.model.name == 'unified.rationale_mtl':
            tokenized_q1_t2 = self.tokenizer(
                seq_in_1_t2,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length
            )
            if rationale is not None:
                tokenized_rationale = self.tokenizer(
                    rationale,
                    padding="max_length",
                    truncation=True,
                    max_length=self.training_args.generation_max_length
                )
            else:
                tokenized_rationale = self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    max_length=self.training_args.generation_max_length
                )

        ####################################################
        # TODO: FLAN                                       #
        ####################################################
        if self.args.model.name == 'unified.flan' or self.args.model.name == 'unified.constrained':
            tokenized_q1_t1_for_hc = self.tokenizer(
                seq_in_for_hc,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length
            )

            tokenized_relevant_cells = self.tokenizer(
                relevant_cells,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.generation_max_length
            )

        ####################################################
        # TODO: CustomT5                                   #
        ####################################################
        if self.args.model.name == 'unified.customT5':
            tokenized_q1_t1_for_hc = self.tokenizer(
                seq_in_for_hc,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length
            )
            tokenized_relevant_cells = self.tokenizer(
                relevant_cells,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.generation_max_length
            )

        ####################################################
        # TODO: Extra Question                             #
        ####################################################
        if self.training_args.use_extra_question is True:
            tokenized_question_two_and_schemas = self.tokenizer(
                seq_in_q2,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
            )

        ####################################################
        # TODO: Shuffled Table                             #
        ####################################################
        if self.training_args.use_shuffled_table is True:
            tokenized_question_and_schemas_shuffled = self.tokenizer(
                seq_in_t2,
                padding="max_length",
                truncation=True,
                max_length=self.training_args.input_max_length,
            )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        if self.args.model.name == 'unified.rationale_mtl' or self.args.model.name == 'unified.flan' or\
                self.args.model.name == 'unified.customT5' or self.args.model.name == 'unified.constrained':
            # Here -100 will let the model not to compute the loss of the padding tokens.
            # tokenized_rationale_input_ids = torch.LongTensor(tokenized_rationale.data["input_ids"])
            # tokenized_rationale_input_ids[tokenized_rationale_input_ids == self.tokenizer.pad_token_id] = -100

            tokenized_relevant_cells_input_ids = torch.LongTensor(tokenized_relevant_cells.data["input_ids"])
            tokenized_relevant_cells_input_ids[tokenized_relevant_cells_input_ids == self.tokenizer.pad_token_id] = -100

        # TODO: Commented by Zhiyu: this following dict, should correspond to the arguments in
        #  forward() function in finetune.py
        if self.args.model.name == 'unified.rationale_mtl':
            item = {
                'input_ids': [torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                              torch.LongTensor(tokenized_q1_t2.data["input_ids"])],
                'attention_mask': [torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                                   torch.LongTensor(tokenized_q1_t2.data["attention_mask"])],
                'labels': tokenized_inferred_input_ids,
                'rationale': tokenized_rationale_input_ids
            }

        elif self.args.model.name == 'unified.flan' or self.args.model.name == 'unified.constrained':
            item = {
                'input_ids': [torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                              torch.LongTensor(tokenized_q1_t1_for_hc.data["input_ids"])],
                'attention_mask': [torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                                   torch.LongTensor(tokenized_q1_t1_for_hc.data["attention_mask"])],
                'labels': tokenized_inferred_input_ids,
                'relevant_cells': tokenized_relevant_cells_input_ids
            }
        elif self.args.model.name == 'unified.customT5':
            item = {
                'input_ids': [torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                              torch.LongTensor(tokenized_q1_t1_for_hc.data["input_ids"])],
                'attention_mask': [torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                                   torch.LongTensor(tokenized_q1_t1_for_hc.data["attention_mask"])],
                'labels': tokenized_inferred_input_ids,
                'relevant_cells': tokenized_relevant_cells_input_ids
            }

        else:
            if self.training_args.use_shuffled_table is False and self.training_args.use_extra_question is False:
                item = {
                    'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                    'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                    'labels': tokenized_inferred_input_ids,
                }

            elif self.training_args.use_shuffled_table is False and self.training_args.use_extra_question is True:
                item = {
                    'input_ids': [torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                                  torch.LongTensor(tokenized_question_two_and_schemas.data["input_ids"])],
                    'attention_mask': [torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                                       torch.LongTensor(tokenized_question_two_and_schemas.data["attention_mask"])],
                    'labels': tokenized_inferred_input_ids,
                }
            elif self.training_args.use_shuffled_table is True and self.training_args.use_extra_question is False:
                item = {
                    'input_ids': [torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
                                  torch.LongTensor(tokenized_question_and_schemas_shuffled.data["input_ids"])],
                    'attention_mask': [torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
                                       torch.LongTensor(
                                           tokenized_question_and_schemas_shuffled.data["attention_mask"])],
                    'labels': tokenized_inferred_input_ids,
                }
            else:
                raise AssertionError(
                    "only one customized data format can be used. Combination of both not supported yet")

        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)


class TokenizedEvalDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][
                                                                                  index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    if raw_item["relevant_cells"] is None:
                        # seq_in  = "text_in ; structured knowledge: struct_in"
                        seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                    else:
                        seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description:
            seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["relevant_cells"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dev notes:
# * The correct length for a given level must be supplied to the forward function.
# * Each level must be forwarded separately.
# * Currently, each level has the same number of filters (but different filters--i.e., different weights,
#   including for the associated embeddings). The input Transformer is expected to be shared.

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["vocab_size"]
        self.class_size = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.padding_idx = kwargs["padding_idx"]
        self.bert_emb_size = kwargs["bert_emb_size"]
        #self.ec_size = kwargs["ec_size"]

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # Note that we use 'ec' (short for error correction) for the final level_id, since (and to avoid confusion that)
        # this level uses ONLY THE TITLES data from level_id 3 (i.e., at the final level, we re-cast as standard
        # classification instead of retrieval involving the difference vector). Typically 'ec' would be trained by
        # freezing BERT.
        for level_id in [1, 2, 3, "ec"]:
            embedding = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM, padding_idx=self.padding_idx)
            setattr(self, f'level{level_id}_embedding', embedding)
            if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
                self.WV_MATRIX = kwargs["WV_MATRIX"]
                self.get_embedding(level_id).weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                if self.MODEL == "static":
                    self.get_embedding(level_id).weight.requires_grad = False

            for i in range(len(self.FILTERS)):  # currently all levels use the same number of filters (and same kernel sizes)
                # always a single channel (first arg)
                conv = nn.Conv1d(1, self.FILTER_NUM[i], (self.WORD_DIM+self.bert_emb_size) * self.FILTERS[i], stride=self.WORD_DIM+self.bert_emb_size)
                setattr(self, f'level{level_id}_conv_{i}', conv)

            # currently only the final level has an error classifier, which is sufficient for the current tasks,
            # but could be expanded in the future
            if level_id == "ec":
                self.fc = nn.Linear(sum(self.FILTER_NUM), self.class_size)

    def get_embedding(self, level_id):
        return getattr(self, f'level{level_id}_embedding')

    def get_conv(self, level_id, i):
        return getattr(self, f'level{level_id}_conv_{i}')

    def forward(self, inp, bert_output=None, level_id=-1, total_length=0, forward_type_description="sentence_representation", main_device=None, split_point=None):

        if bert_output is not None:
            x = self.get_embedding(level_id)(inp)
            x = torch.cat([x, bert_output], 2).view(-1, 1, (self.WORD_DIM+self.bert_emb_size) * total_length)
        else:
            x = self.get_embedding(level_id)(inp).view(-1, 1, self.WORD_DIM * total_length)

        if forward_type_description == "sentence_representation":
            if level_id != "ec":
                # Note: No relu, no dropout
                conv_results = [
                    F.max_pool1d(
                        self.get_conv(level_id, i)(x), total_length - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])
                    for i in range(len(self.FILTERS))]
                x = torch.cat(conv_results, 1)
                return x
            elif level_id == "ec":
                # Note: relu and dropout as input to the fc
                conv_results = [
                    F.max_pool1d(
                        F.relu(
                            self.get_conv(level_id, i)(x)
                        ), total_length - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i])
                    for i in range(len(self.FILTERS))]

                x = torch.cat(conv_results, 1)
                x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
                x = self.fc(x)
                return x
            else:
                assert False, f"level_id {level_id} not implemented"
        elif forward_type_description == "blade":
            # For analyzing binary classification/error correction, or training supervised token-level labeling
            assert level_id == "ec"
            assert False, "ERROR: BLADE is not implemented here; Use the BLADE repo for the time being."

            # max_pool_outputs = []
            # max_pool_outputs_indices = []
            # for i in range(len(self.FILTERS)):
            #     max_pool, max_pool_indices = F.max_pool1d(F.relu(self.get_conv(i)(x)), self.total_length - self.FILTERS[i] + 1, return_indices=True)
            #     max_pool_outputs.append(max_pool.view(-1, self.FILTER_NUM[i]))
            #     max_pool_outputs_indices.append(max_pool_indices)
            #
            # concat_maxpool = torch.cat(max_pool_outputs, 1)
            # concat_maxpool = F.dropout(concat_maxpool, p=self.DROPOUT_PROB, training=self.training)
            #
            # token_contributions_tensor = torch.zeros(concat_maxpool.shape[0], self.total_length).to(main_device)
            # negative_contributions = concat_maxpool * self.fc.weight[0]
            # positive_contributions = concat_maxpool * self.fc.weight[1]
            # # We can potentially remove this loop (in a manner analogous to the multiBLADE implementation), but the binary case is already quite fast,
            # # so we keep this expanded to aid in understanding. See the multiBLADE implementation (which can also
            # # be used for the binary case) for a more efficient variant. (Efficiency only becomes particularly
            # # relevant for the supervised cases with multiple labels.)
            # for maxpool_id in range(len(self.FILTERS)):
            #     for index_into_maxpool in range(self.FILTER_NUM[maxpool_id]):
            #         contribution_id = sum(self.FILTER_NUM[0:maxpool_id]) + index_into_maxpool
            #         contributions = positive_contributions[:, contribution_id] - negative_contributions[:,
            #                                                                      contribution_id]
            #         indexes_into_padded_sentence = max_pool_outputs_indices[maxpool_id][:, index_into_maxpool, 0]
            #         for filter_index_offset in range(self.FILTERS[maxpool_id]):
            #             index_into_padded_sentence = indexes_into_padded_sentence + filter_index_offset
            #             token_contributions_tensor.scatter_add_(1, index_into_padded_sentence.view(-1, 1),
            #                                                     contributions.view(-1, 1))
            #
            # token_contributions_tensor.add_(self.fc.bias[1] - self.fc.bias[0])
            #
            # return token_contributions_tensor, self.fc(concat_maxpool)
        elif forward_type_description == "alignment_visualization":  # alignment across sequences (without ec layer)
            # We can remove the loops (as in the multiBLADE implementation) for improved efficiency, but here, we
            # keep them to make it easier to follow (and since the extra speedup isn't necessary here).
            assert level_id != "ec", f"ERROR: alignment_visualization is only intended for levels 1-3. " \
                                     f"Use [multi]BLADE for ec."

            # in this case, we save the token-level 'contributions' (note the semantics are different than in the
            # BLADE case), as well as the alignments
            max_pool_outputs = []
            max_pool_outputs_indices = []
            for i in range(len(self.FILTERS)):
                # Note: No relu (contrast with BLADE with ec)
                max_pool, max_pool_indices = F.max_pool1d(self.get_conv(level_id, i)(x), total_length - self.FILTERS[i] + 1, return_indices=True)
                max_pool_outputs.append(max_pool.view(-1, self.FILTER_NUM[i]))
                max_pool_outputs_indices.append(max_pool_indices)

            concat_maxpool = torch.cat(max_pool_outputs, 1)
            # we assume the claims are stacked above the titles, separated at split_point:
            token_contributions_tensor = torch.zeros(concat_maxpool.shape[0], total_length).to(main_device)
            filter_diff = torch.pow(concat_maxpool[0:split_point] - concat_maxpool[split_point:], 2)

            for maxpool_id in range(len(self.FILTERS)):
                for index_into_maxpool in range(self.FILTER_NUM[maxpool_id]):
                    contribution_id = sum(self.FILTER_NUM[0:maxpool_id]) + index_into_maxpool
                    # note that contributions are now concat_maxpool.shape[0]/2
                    # The following is the filter weight scaled by the bi-sequence difference. A smaller difference
                    # corresponds to greater contribution weight. I.e., this captures the notion of highlighting max
                    # activated tokens that also have small differences. Note that this is primarily for debugging;
                    # whereas the actual aligned distance weights are faithful to the underlying model, in that the
                    # summation of those weights is what is used to make the predictions at every level.
                    # (I.e., at an abstract level, the token-level [projected] scores are not directly interpretable
                    # in the sense of BLADE, but the bi-sequence filter differences are.)
                    contributions_claims = concat_maxpool[0:split_point, contribution_id] * torch.exp(-1*filter_diff[:, contribution_id])
                    contributions_titles = concat_maxpool[split_point:, contribution_id] * torch.exp(-1*filter_diff[:, contribution_id])

                    indexes_into_padded_sentence_claims = max_pool_outputs_indices[maxpool_id][0:split_point, index_into_maxpool, 0]
                    indexes_into_padded_sentence_titles = max_pool_outputs_indices[maxpool_id][split_point:, index_into_maxpool, 0]
                    # Note: Here, we always use filters of length 1. I kept this for generality with the multi-BLADE
                    # only CNN model code base.
                    for filter_index_offset in range(self.FILTERS[maxpool_id]):
                        index_into_padded_sentence_claims = indexes_into_padded_sentence_claims + filter_index_offset
                        token_contributions_tensor[0:split_point].scatter_add_(1, index_into_padded_sentence_claims.view(-1, 1),
                                                                contributions_claims.view(-1, 1))
                        index_into_padded_sentence_titles = indexes_into_padded_sentence_titles + filter_index_offset
                        token_contributions_tensor[split_point:].scatter_add_(1, index_into_padded_sentence_titles.view(-1, 1),
                                                                contributions_titles.view(-1, 1))

            return token_contributions_tensor, concat_maxpool, max_pool_outputs_indices, filter_diff

        elif forward_type_description == "multiblade":
            # For analyzing or training  multi label/class classification/error correction.
            # Here, a second fc must be initialized.
            # This can also be used in binary cases where the global norm behavior is desired (i.e., associating
            # one token with the global prediction, as in the original paper).
            # Note that this currently returns a re-sort of the labels (the trailing dimension of
            # token_contributions_tensor), even if the sample size equals the total number of labels
            # (e.g., use scatter_add_(1, x, y) operation with top_labels_i as the index x into the contributions y).
            # The sort must be accounted for at inference. (The top-k sorting was to accommodate
            # sampling the very large number of samples in early epochs in MIMIC.)
            # Other than the sort, the key thing to understand is that now token_contributions_tensor
            # has a token-level score FOR EACH LABEL. Conceptually, this is otherwise similar to BLADE above, but the
            # actual implementation avoids an explicit for loop over the labels (of which there may be several
            # thousands in practice); hence, the transformations with the repeat_interleave operation.
            # TODO for refactor:
            #  1. re-sort before return.
            #  2. standardize the property names from the BLADE repo
            #  3. standardize masking, which doesn't matter with 'sentence_representation' but does with min-max
            #  losses (i.e., be explicit about the no-mask option, which would be a defacto relu over the
            #  contributions in min-max).
            #  4. Add the global re-norm here to the forward rather than having it in train()/test().
            assert level_id == "ec"
            assert False, "ERROR: multiBLADE is not implemented here; Use the BLADE repo for the time being."

            # max_pool_outputs = []
            # max_pool_outputs_indices = []
            # for i in range(len(self.FILTERS)):
            #     max_pool, max_pool_indices = F.max_pool1d(F.relu(self.get_conv(i)(x)), self.total_length - self.FILTERS[i] + 1, return_indices=True)
            #     max_pool_outputs.append(max_pool.view(-1, self.FILTER_NUM[i]))
            #     max_pool_outputs_indices.append(max_pool_indices)
            #
            # concat_maxpool = torch.cat(max_pool_outputs, 1)
            # concat_maxpool = F.dropout(concat_maxpool, p=self.DROPOUT_PROB, training=self.training)
            # second_fc_out = self.second_fc(concat_maxpool)
            #
            # global_label_predictions = second_fc_out[:, :self.real_class_size] - second_fc_out[:, self.real_class_size:]
            # # Of CRITICAL importance, is to remember that labels are now ordered by top_labels_i (*not* necessarily in sorted order by index)
            # top_labels_scores, top_labels_i = torch.topk(global_label_predictions, label_sample_size, dim=1, largest=True, sorted=True)
            # token_contributions_tensor = torch.zeros(concat_maxpool.shape[0], self.total_length, label_sample_size).to(main_device)  # now we have a 3rd dim to handle contributions by class
            #
            # # dim=2 is necessary for the gather
            # top_labels_i_repeated = torch.repeat_interleave(top_labels_i.unsqueeze(dim=2), concat_maxpool.shape[1], dim=2)
            # # repeat weights by batch size (concat_maxpool.shape[0])
            # # Note that the following transforms are necessary, since weights are constant across sentences, BUT the top
            # # labels for any given sentence differ, so we need to grab different weights for each sentence
            # pos_weights_by_batch = torch.repeat_interleave(self.fc.weight[:self.real_class_size].unsqueeze(dim=0), concat_maxpool.shape[0], dim=0)
            # neg_weights_by_batch = torch.repeat_interleave(self.fc.weight[self.real_class_size:].unsqueeze(dim=0), concat_maxpool.shape[0], dim=0)
            # # only select the values for the labels of interest (topk)
            # fc_pos_weights_subset = torch.gather(pos_weights_by_batch, 1, top_labels_i_repeated)
            # fc_neg_weights_subset = torch.gather(neg_weights_by_batch, 1, top_labels_i_repeated)
            #
            # positive_contributions = concat_maxpool.unsqueeze(dim=1) * fc_pos_weights_subset
            # negative_contributions = concat_maxpool.unsqueeze(dim=1) * fc_neg_weights_subset
            #
            # for maxpool_id in range(len(self.FILTERS)):  # 0,1,2
            #     filter_start_index = sum(self.FILTER_NUM[0:maxpool_id])
            #     filter_end_index = filter_start_index + self.FILTER_NUM[maxpool_id]
            #     contributions = positive_contributions[:, :, filter_start_index:filter_end_index] - negative_contributions[:, :, filter_start_index:filter_end_index]
            #
            #     # these are indexes into the filters (and by extension, token indexes) for every filter (BY BATCH)
            #     all_indexes_into_padded_sentence = max_pool_outputs_indices[maxpool_id][:,:,0]
            #     # max_pool_outputs_indices[maxpool_id].shape is batch by filter num (of this filter) by 1
            #
            #     for filter_index_offset in range(self.FILTERS[maxpool_id]):
            #         adjusted_all_indexes_into_padded_sentence = all_indexes_into_padded_sentence + filter_index_offset
            #         # need to replicate this to be batch BY LABELS
            #         adjusted_all_indexes_into_padded_sentence = torch.repeat_interleave(adjusted_all_indexes_into_padded_sentence.unsqueeze(dim=2),
            #                                                              label_sample_size, dim=2)
            #         token_contributions_tensor.scatter_add_(1, adjusted_all_indexes_into_padded_sentence, contributions.transpose(1,2))
            #
            # # analogous to the transform for weights
            # pos_bias_by_batch = torch.repeat_interleave(self.fc.bias[:self.real_class_size].unsqueeze(dim=0), concat_maxpool.shape[0], dim=0)
            # neg_bias_by_batch = torch.repeat_interleave(self.fc.bias[self.real_class_size:].unsqueeze(dim=0), concat_maxpool.shape[0], dim=0)
            # # only select the values for the labels of interest (topk)
            # # here, since the bias is just a single value, we use top_labels_i instead of top_labels_i_repeated
            # fc_pos_bias_subset = torch.gather(pos_bias_by_batch, 1, top_labels_i)
            # fc_neg_bias_subset = torch.gather(neg_bias_by_batch, 1, top_labels_i)
            # # now we have to repeat by self.total_length before adding
            # fc_pos_bias_subset = torch.repeat_interleave(fc_pos_bias_subset.unsqueeze(dim=1), self.total_length, dim=1)
            # fc_neg_bias_subset = torch.repeat_interleave(fc_neg_bias_subset.unsqueeze(dim=1), self.total_length, dim=1)
            # token_contributions_tensor.add_(fc_pos_bias_subset - fc_neg_bias_subset)
            # # it is important to remember that token_contributions_tensor dim=2 is now indexed by top_labels_i (and is not guaranteed to be in the original label order)
            # return global_label_predictions, token_contributions_tensor, top_labels_scores, top_labels_i
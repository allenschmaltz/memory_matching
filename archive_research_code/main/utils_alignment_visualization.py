# alignment viz: currently only for 2-class and of level 1

from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
import utils_search

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy

import math

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam

from collections import defaultdict

import torch.nn.functional as F

import subprocess
import time


def visualization_main(data, params, np_random_state, bert_model, tokenizer, bert_device, model, predicted_output):

    viz_output = retrieve_and_save_viz_cross_sequences(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                                   model, predicted_output)
    return viz_output


def retrieve_and_save_viz_cross_sequences(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                               model, predicted_output):

    exemplars_levels_to_consider = [1, 2, 3]
    # important, note we only consider 2 class for the symmetric experiments:
    decision_labels_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID]

    start_time = time.time()
    model.eval()
    bert_model.eval()

    alignment_scale_factor = 100  # this is for display purposes: alignment_scale_factor*filter difference
    batch_size = params["BATCH_SIZE"]

    decision_labels = data[f"{'test'}_decision_labels"]
    # each level should have the same number of claims:
    total_number_of_claims = len(data[f"level{1}_idx_{'test'}_x"])
    for level_id in exemplars_levels_to_consider:
        assert len(data[f"level{level_id}_idx_{'test'}_x"]) == total_number_of_claims
        assert len(predicted_output[f"level{level_id}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"]) == \
               total_number_of_claims
        assert len(decision_labels) == total_number_of_claims

    # here we keep it simple, since retrieval is given in the symmetric data and predictions are only based on level 3
    # decisions
    level3_augmented_is_correct_predictions = []
    level3_augmented_predictions = []

    # initialize data structures:
    mini_batches = {}
    for level_id in exemplars_levels_to_consider:
        mini_batches[f"level{level_id}_augmented_idx_x"], \
        mini_batches[f"level{level_id}_augmented_bert_idx_sentences"], \
        mini_batches[f"level{level_id}_augmented_bert_input_masks"], \
        mini_batches[f"level{level_id}_augmented_idx_titles"], \
        mini_batches[f"level{level_id}_augmented_bert_idx_titles"], \
        mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"] = [], [], [], [], [], []

        mini_batches[f"level{level_id}_augmented_x_tokenized"] = []
        # currently, the following are only generated for level 1; if we want to output levels 2 or 3, we need to
        # reconstruct the alignment, as in the ec code
        mini_batches[f"level{level_id}_augmented_tokenized_titles"] = []
        mini_batches[f"level{level_id}_augmented_original_titles"] = []
    # These cross sequences (claim+title) always match the number of claims.
    cross_sequence_id = 0
    cross_sequence_id_to_claim_id = {}
    unique_titles_to_decision_labels = data[f"predicted_level{3}_{'test'}_unique_titles_to_decision_labels"]
    for claim_index in range(total_number_of_claims):
        level3_title_id_and_dist_and_ec_relative_to_level = \
            predicted_output[f"level{3}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]

        assert len(level3_title_id_and_dist_and_ec_relative_to_level) == 2, f"ERROR: Only 2 labels expected for level 3 " \
                                                                     f"in the symmetric data"
        # only take the top of the beam for the query at level 3
        level3_title_id_and_dist_and_ec_relative_to_level = level3_title_id_and_dist_and_ec_relative_to_level[0:1]

        assert len(predicted_output[f"level{1}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]) == 1, \
            f"ERROR: Only 1 title expected for levels " \
            f"1 and 2 in the symmetric data (i.e., " \
            f"evidence is given and it consists of only" \
            f"a single sentence)."
        assert len(predicted_output[f"level{2}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]) == 1, \
            f"ERROR: Only 1 title expected for levels " \
            f"1 and 2 in the symmetric data (i.e., " \
            f"evidence is given and it consists of only" \
            f"a single sentence)."
        for level3_predicted_i in range(len(level3_title_id_and_dist_and_ec_relative_to_level)):
            level3_predicted_title_id = level3_title_id_and_dist_and_ec_relative_to_level[level3_predicted_i][0]
            predicted_decision_id = unique_titles_to_decision_labels[level3_predicted_title_id]
            true_decision_id = decision_labels[claim_index]
            assert predicted_decision_id in decision_labels_to_consider
            level3_augmented_is_correct_predictions.append(int(predicted_decision_id == true_decision_id))
            level3_augmented_predictions.append(predicted_decision_id)

            assert cross_sequence_id not in cross_sequence_id_to_claim_id
            cross_sequence_id_to_claim_id[cross_sequence_id] = claim_index
            cross_sequence_id += 1
            for level_id in exemplars_levels_to_consider:
                if level_id == 3:
                    predicted_title_id = level3_predicted_title_id
                else:
                    # for levels 1 and 2, we just take the top of the beam, since retrieval is given in this case
                    predicted_title_id = \
                    predicted_output[f"level{level_id}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][
                        claim_index][0][0]

                # add 'left' sequences
                mini_batches[f"level{level_id}_augmented_idx_x"].append(
                    data[f"level{level_id}_idx_{'test'}_x"][claim_index]
                )
                mini_batches[f"level{level_id}_augmented_bert_idx_sentences"].append(
                    data[f"level{level_id}_{'test'}_bert_idx_sentences"][claim_index]
                )
                mini_batches[f"level{level_id}_augmented_bert_input_masks"].append(
                    data[f"level{level_id}_{'test'}_bert_input_masks"][claim_index]
                )

                # NOTE: currently this tracks the claims, without respect to length cutoffs (which are level dependent)
                # or padding, which must be accounted for when re-assigning maxpool indexes
                mini_batches[f"level{level_id}_augmented_x_tokenized"].append(
                    data[f"{'test'}_x"][claim_index]
                )

                # add 'right' sequences
                if level_id == 1:
                    mini_batches[f"level{level_id}_augmented_idx_titles"].append(
                        data[f"{'test'}_idx_unique_titles"][predicted_title_id]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_idx_titles"].append(
                        data[f"{'test'}_bert_idx_unique_titles"][predicted_title_id]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"].append(
                        data[f"{'test'}_bert_input_masks_unique_titles"][predicted_title_id]
                    )
                    # NOTE: currently only constructed for level 1
                    mini_batches[f"level{level_id}_augmented_tokenized_titles"].append(
                        data[f"{'test'}_unique_titles"][predicted_title_id]
                    )
                    mini_batches[f"level{level_id}_augmented_original_titles"].append(
                        ' '.join(data[f"{'test'}_unique_original_titles"][predicted_title_id])
                    )

                else:
                    mini_batches[f"level{level_id}_augmented_idx_titles"].append(
                        data[f"predicted_level{level_id}_{'test'}_idx_unique_titles"][predicted_title_id]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_idx_titles"].append(
                        data[f"predicted_level{level_id}_{'test'}_bert_idx_unique_titles"][predicted_title_id]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"].append(
                        data[f"predicted_level{level_id}_{'test'}_bert_input_masks_unique_titles"][predicted_title_id]
                    )
    assert cross_sequence_id == total_number_of_claims, f"{cross_sequence_id}, {total_number_of_claims}"

    total_sequences = len(mini_batches[f"level{1}_augmented_idx_x"])
    for level_id in exemplars_levels_to_consider:
        assert len(mini_batches[f"level{level_id}_augmented_idx_x"]) == total_sequences

    with torch.no_grad():
        running_sentence_index = 0
        batch_num = 0
        for i in range(0, total_sequences, batch_size):
            batch_num += 1
            batch_range = min(batch_size, total_sequences - i)

            # at the moment, we only consider level 1
            #for level_id in exemplars_levels_to_consider:
            for level_id in [1]:
                batch_x = mini_batches[f"level{level_id}_augmented_idx_x"][i:i + batch_range]
                number_of_claims = len(batch_x)
                batch_x.extend(mini_batches[f"level{level_id}_augmented_idx_titles"][i:i + batch_range])

                bert_output = run_main.get_bert_representations(
                    mini_batches[f"level{level_id}_augmented_bert_idx_sentences"][i:i + batch_range] +
                    mini_batches[f"level{level_id}_augmented_bert_idx_titles"][i:i + batch_range],
                    mini_batches[f"level{level_id}_augmented_bert_input_masks"][i:i + batch_range] +
                    mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"][i:i + batch_range],
                    bert_model, bert_device, params["bert_layers"], len(batch_x[0]))
                bert_output = torch.FloatTensor(bert_output).to(params["main_device"])

                batch_x = torch.LongTensor(batch_x).to(params["main_device"])

                total_length = params["max_length"] * level_id + 2 * constants.PADDING_SIZE

                token_contributions_tensor, concat_maxpool, max_pool_outputs_indices, filter_diff = \
                    model(batch_x, bert_output, level_id=level_id, total_length=total_length,
                          forward_type_description="alignment_visualization", main_device=params["main_device"],
                          split_point=number_of_claims)
                if i == 0:
                    print(f"token_contributions_tensor.shape: {token_contributions_tensor.shape}")
                    print(f"len(max_pool_outputs_indices): {len(max_pool_outputs_indices)}")
                    print(f"max_pool_outputs_indices[0].shape: {max_pool_outputs_indices[0].shape}")
                    print(f"filter_diff.shape: {filter_diff.shape}")
                token_contributions_tensor_claims = token_contributions_tensor[0:number_of_claims]
                token_contributions_tensor_titles = token_contributions_tensor[number_of_claims:]

                batch_x_tokenized = mini_batches[f"level{level_id}_augmented_x_tokenized"][i:i + batch_range]
                batch_title_tokenized = mini_batches[f"level{level_id}_augmented_tokenized_titles"][i:i + batch_range]
                batch_original_titles = mini_batches[f"level{level_id}_augmented_original_titles"][i:i + batch_range]
                #batch_true_title_tokenized = title_tokenized[i:i + batch_range]
                batch_is_correct = level3_augmented_is_correct_predictions[i:i + batch_range]

                for sentence_i in range(number_of_claims):
                    # build alignments
                    claim_pivot_alignments = {}
                    # build scaled alignments (filter * exp(-alpha*filter_diff))
                    claim_pivot_scaled_alignments = {}
                    for token_i, _ in enumerate(batch_x_tokenized[sentence_i]):
                        claim_pivot_alignments[token_i] = defaultdict(float)
                        claim_pivot_scaled_alignments[token_i] = defaultdict(list)
                    for index_into_maxpool in range(model.FILTER_NUM[0]):
                        claim_index = max_pool_outputs_indices[0][sentence_i, index_into_maxpool ,0].item() # 1000
                        title_index = max_pool_outputs_indices[0][number_of_claims+sentence_i, index_into_maxpool, 0].item()
                        alignment_weight = filter_diff[sentence_i, index_into_maxpool].item()  # 1000
                        # only consider alignments to non-padding indexes up to tokenized length
                        tokenized_claim_index = claim_index - constants.PADDING_SIZE
                        tokenized_title_index = title_index - constants.PADDING_SIZE
                        if (tokenized_claim_index >= 0 and tokenized_claim_index < len(batch_x_tokenized[sentence_i])) and \
                            (tokenized_title_index >= 0 and tokenized_title_index < len(batch_title_tokenized[sentence_i])):

                            claim_token = batch_x_tokenized[sentence_i][tokenized_claim_index]
                            title_token = batch_title_tokenized[sentence_i][tokenized_title_index]
                            assert tokenized_claim_index in claim_pivot_alignments
                            one_claim_token_alignments = claim_pivot_alignments[tokenized_claim_index]
                            one_claim_token_alignments[(tokenized_claim_index, tokenized_title_index, claim_token, title_token)] += alignment_weight
                            claim_pivot_alignments[tokenized_claim_index] = one_claim_token_alignments

                            # scaled alignments: we scale the filter weights by the difference; note that the
                            # claim and title weights are not necessarily the same, so we save both
                            claim_filter_weight_scaled = concat_maxpool[sentence_i, index_into_maxpool].item() * np.exp(-1 * alignment_weight)
                            title_filter_weight_scaled = concat_maxpool[number_of_claims+sentence_i, index_into_maxpool].item() * np.exp(-1 * alignment_weight)
                            one_claim_token_scaled_alignments = claim_pivot_scaled_alignments[tokenized_claim_index]
                            running_scaled_alignments = one_claim_token_scaled_alignments[(tokenized_claim_index, tokenized_title_index, claim_token, title_token)]
                            if len(running_scaled_alignments) == 0:
                                running_scaled_alignments = [claim_filter_weight_scaled, title_filter_weight_scaled]
                            else:
                                running_scaled_alignments[0] += claim_filter_weight_scaled
                                running_scaled_alignments[1] += title_filter_weight_scaled
                            one_claim_token_scaled_alignments[(tokenized_claim_index, tokenized_title_index, claim_token, title_token)] = running_scaled_alignments
                            claim_pivot_scaled_alignments[tokenized_claim_index] = one_claim_token_scaled_alignments

                    annotated_x = []
                    annotated_x_indexes = []
                    annotated_title = []
                    #max_pool_outputs_indices[0]  # only use uniCNN here
                    for token_i, token in enumerate(batch_x_tokenized[sentence_i]):
                        # note that some tokenized sentences may exceed the max length; here, we just truncate (as with training)
                        if constants.PADDING_SIZE + token_i < token_contributions_tensor_claims.shape[1]:
                            annotated_x.append(
                                f"{token} ({round(token_contributions_tensor_claims[sentence_i, constants.PADDING_SIZE + token_i].item(), 1)})")
                    for token_i, token in enumerate(batch_title_tokenized[sentence_i]):
                        # note that some tokenized sentences may exceed the max length; here, we just truncate (as with training)
                        if constants.PADDING_SIZE + token_i < token_contributions_tensor_titles.shape[1]:
                            annotated_title.append(
                                f"{token} ({round(token_contributions_tensor_titles[sentence_i, constants.PADDING_SIZE + token_i].item(), 1)})")

                    print(f"Sent: {running_sentence_index}; Level 3 prediction is true: {batch_is_correct[sentence_i]==1}; Level 3 prediction: {level3_augmented_predictions[running_sentence_index]}")
                    original_claim = ' '.join(data[f"{'test'}_original_sentences"][running_sentence_index])
                    print(f"\t{original_claim}")
                    print(f"\t{batch_original_titles[sentence_i]}")
                    print(f"----")
                    print(f"\t{' '.join(annotated_x)}")
                    print(f"----")
                    print(f"\t{' '.join(annotated_title)}")
                    print(f"----")
                    #print(claim_pivot_alignments)
                    for token_i, token in enumerate(batch_x_tokenized[sentence_i]):
                        one_claim_token_alignments = claim_pivot_alignments[token_i]
                        one_claim_token_scaled_alignments = claim_pivot_scaled_alignments[token_i]
                        assert len(one_claim_token_alignments) == len(one_claim_token_scaled_alignments)
                        alignment = [f"{token_i}, {token} ||"]
                        for one_alignment in one_claim_token_alignments:
                            tokenized_claim_index, tokenized_title_index, claim_token, title_token = one_alignment
                            assert token_i == tokenized_claim_index and token == claim_token, f"{token_i}, {token}: {one_alignment}"
                            alignment_weight = one_claim_token_alignments[one_alignment]
                            assert one_alignment in one_claim_token_scaled_alignments
                            scaled_claim_title_alignments = one_claim_token_scaled_alignments[one_alignment]
                            alignment.append(f"{title_token}[{tokenized_title_index}]: ({round(alignment_scale_factor*alignment_weight, 1)}, {round(scaled_claim_title_alignments[0], 1)}, {round(scaled_claim_title_alignments[1], 1)});")
                        print(f"{' '.join(alignment)}")
                    print(f"----Max alignments (claim, title)")
                    for token_i, token in enumerate(batch_x_tokenized[sentence_i]):
                        one_claim_token_alignments = claim_pivot_alignments[token_i]
                        one_claim_token_scaled_alignments = claim_pivot_scaled_alignments[token_i]
                        assert len(one_claim_token_alignments) == len(one_claim_token_scaled_alignments)
                        alignment = f"{token_i}, {token} ||"
                        max_claim_string = ""
                        max_claim_weight = -1.0
                        max_title_string = ""
                        max_title_weight = -1.0
                        for one_alignment in one_claim_token_alignments:
                            tokenized_claim_index, tokenized_title_index, claim_token, title_token = one_alignment
                            assert token_i == tokenized_claim_index and token == claim_token, f"{token_i}, {token}: {one_alignment}"
                            #alignment_weight = one_claim_token_alignments[one_alignment]
                            assert one_alignment in one_claim_token_scaled_alignments
                            scaled_claim_title_alignments = one_claim_token_scaled_alignments[one_alignment]
                            if scaled_claim_title_alignments[0] > max_claim_weight:
                                max_claim_string = f"{title_token}[{tokenized_title_index}]: ({round(scaled_claim_title_alignments[0], 1)})"
                                max_claim_weight = scaled_claim_title_alignments[0]
                            if scaled_claim_title_alignments[1] > max_title_weight:
                                max_title_string = f"{title_token}[{tokenized_title_index}]: ({round(scaled_claim_title_alignments[1], 1)})"
                                max_title_weight = scaled_claim_title_alignments[1]
                            #alignment.append(f"{title_token}[{tokenized_title_index}]: ({round(alignment_scale_factor*alignment_weight, 1)}, {round(scaled_claim_title_alignments[0], 1)}, {round(scaled_claim_title_alignments[1], 1)});")
                        print(f"{alignment} {max_claim_string} {max_title_string}")
                        #print(f"{' '.join(alignment)}")
                    print(f"----Min distance")
                    for token_i, token in enumerate(batch_x_tokenized[sentence_i]):
                        one_claim_token_alignments = claim_pivot_alignments[token_i]
                        alignment = f"{token_i}, {token} ||"
                        min_dist_claim_string = ""
                        min_dist_claim_weight = np.inf
                        for one_alignment in one_claim_token_alignments:
                            tokenized_claim_index, tokenized_title_index, claim_token, title_token = one_alignment
                            assert token_i == tokenized_claim_index and token == claim_token, f"{token_i}, {token}: {one_alignment}"
                            alignment_weight = one_claim_token_alignments[one_alignment]
                            if alignment_weight < min_dist_claim_weight:
                                min_dist_claim_string = f"{title_token}[{tokenized_title_index}]: ({alignment_weight})"
                                min_dist_claim_weight = alignment_weight
                        print(f"{alignment} {min_dist_claim_string}")
                    print("*****************************")
                    running_sentence_index += 1

    print(f"For display purposes, alignment weights have been scaled by a factor of {alignment_scale_factor}.")
    end_time = time.time()
    print(f"retrieve_and_save_viz_cross_sequences time: {(end_time - start_time) / 60} minutes")
    return []

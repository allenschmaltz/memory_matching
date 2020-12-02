from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
import utils_search_eval_levels
#import utils_search

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

# for saving FT BERT
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
import os


def retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title", split_mode="test", level_id=-1):
    start_time = time.time()
    model.eval()
    bert_model.eval()
    assert level_id != -1
    assert level_id != "ec", f"ERROR: Memory retrieval is not needed with 'ec'. Use the standard test() instead."

    total_length = params["max_length"]*level_id + 2*constants.PADDING_SIZE
    print(f"retrieve_and_save_memory(): mode={mode}, split_mode={split_mode}, level_id={level_id}, total_length={total_length}")
    if mode == "title":
        # note level 1 unique titles do not have a prefix
        if level_id == 1:
            x_full, bert_idx_sentences_full, bert_input_masks_full = data[f"{split_mode}_idx_unique_titles"], \
                                                                     data[f"{split_mode}_bert_idx_unique_titles"], \
                                                                     data[f"{split_mode}_bert_input_masks_unique_titles"]
        else:

            x_full, bert_idx_sentences_full, bert_input_masks_full = data[f"predicted_level{level_id}_{split_mode}_idx_unique_titles"], \
                                                                     data[f"predicted_level{level_id}_{split_mode}_bert_idx_unique_titles"], \
                                                                     data[f"predicted_level{level_id}_{split_mode}_bert_input_masks_unique_titles"]
        chunk_size = data[f"level{level_id}_titles_chunk_size"]
        memory_batch_size = data[f"level{level_id}_memory_batch_size"]
        memory_dir = data[f"titles_memory_dir"]
    else:
        assert mode == "retrieve"
        chunk_size = data[f"level{level_id}_retrieval_chunk_size"]
        memory_batch_size = data[f"level{level_id}_retrieval_batch_size"]
        memory_dir = data[f"retrieval_memory_dir"]

        # in fever, these are always the claims; the main content is the same, but the
        # truncation to max_length is different and the prefixes are different
        x_full, bert_idx_sentences_full, bert_input_masks_full = data[f"level{level_id}_idx_{split_mode}_x"], \
                                                                 data[f"level{level_id}_{split_mode}_bert_idx_sentences"], \
                                                                 data[f"level{level_id}_{split_mode}_bert_input_masks"]
    chunk_ids = []
    with torch.no_grad():
        chunk_id = 0
        for chunk_i in range(0, len(x_full), chunk_size):
            chunk_range = min(chunk_size, len(x_full) - chunk_i)
            x = x_full[chunk_i:chunk_i + chunk_range]
            bert_idx_sentences = bert_idx_sentences_full[chunk_i:chunk_i + chunk_range]
            bert_input_masks = bert_input_masks_full[chunk_i:chunk_i + chunk_range]
            print(f"Processing {mode} memory structure chunk_id {chunk_id}: starting at {chunk_i}, with size {len(x)}, out of {len(x_full)}")
            memory_structure = []
            num_batch_instances = math.ceil((len(x) / memory_batch_size))
            batch_num = 0
            for i in range(0, len(x), memory_batch_size):
                # if batch_num % max(1, int(num_batch_instances * 0.25)) == 0:
                #     print(f"\tProcessing {mode} memory, {batch_num/num_batch_instances}")
                batch_num += 1
                batch_range = min(memory_batch_size, len(x) - i)
                batch_x = x[i:i + batch_range]

                # get BERT representations
                bert_output = run_main.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                                bert_input_masks[i:i + batch_range],
                                                                bert_model, bert_device, params["bert_layers"],
                                                                len(batch_x[0]))

                batch_x = torch.LongTensor(batch_x).to(params["main_device"])
                bert_output = torch.FloatTensor(bert_output).to(params["main_device"])

                model_output = model(batch_x, bert_output, level_id=level_id, total_length=total_length,
                        forward_type_description="sentence_representation", main_device=None)
                memory_structure.append(model_output)
                if chunk_i == 0 and i == 0 and params["main_device"].type != "cpu":
                    print(
                        f'i==0 retrieve_and_save_memory torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')

            utils.save_memory_structure_torch(memory_dir, f"level{level_id}_{mode}_{split_mode}", torch.cat(memory_structure, 0), chunk_id)
            chunk_ids.append(chunk_id)
            chunk_id += 1
        # save chunk ids -- These are the keys for recovering the files
        utils.save_memory_structure_torch(memory_dir, f"level{level_id}_chunk_ids_{mode}_{split_mode}", torch.LongTensor(chunk_ids), 0)
    end_time = time.time()
    print(f"retrieve_and_save_memory time: {(end_time - start_time) / 60} minutes")


def get_top_k_nearest_titles_from_memory(pdist, data, model, params, mode="train", level_id=-1):
    # We assume the full memory and retrieval chunks can be placed in memory
    # The cached memory and retrieval ids are kept in sync via the saved chunk_ids

    # This loops through the memory chunks, and loops through retrieval chunks.
    # This is structured in this way since the full memory is, in general, too large to fit in the GPU at one time.
    # The retrieval is also cached to avoid having to recalculate the forward pass when looping over the memory.

    assert level_id != -1
    start_time = time.time()
    model.eval()

    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]
    memory_dir = data["titles_memory_dir"]
    memory_chunk_ids = utils.load_memory_structure_torch(memory_dir, f"level{level_id}_chunk_ids_title_{mode}", 0, -1).numpy()

    retrieval_memory_dir = data["retrieval_memory_dir"]
    retrieval_chunk_ids = utils.load_memory_structure_torch(retrieval_memory_dir, f"level{level_id}_chunk_ids_retrieve_{mode}", 0, -1).numpy()

    # level1 for the covered claims corresponds to the original covered wiki sentences; for levels 2 and 3 these
    # are constructed via search and must match the data[f"predicted_level{level_id}_{split_mode}_idx_unique_titles"],
    # which typically will be unique for each claim (since the claim is part of the text). The only sharing occurs
    # if there are duplicate claims.
    claims_to_covered_titles_ids_tensors = data[f"level{level_id}_{mode}_claims_to_covered_titles_ids_tensors"]

    retrieval_id_to_title_id_and_dist_and_ec = defaultdict(list)  # sorted (closest to farthest) list of [title id, dist]

    with torch.no_grad():
        running_memory_line_index = 0
        for memory_chunk_id in memory_chunk_ids:
            running_retrieval_line_index = 0  # Note that we reload the entire retrieval set for EACH memory chunk
            print(f"TopK: Processing memory chunk id {memory_chunk_id} of {len(memory_chunk_ids)}.")
            title_memory = utils.load_memory_structure_torch(memory_dir, f"level{level_id}_title_{mode}", memory_chunk_id, int(params["GPU"]))
            for retrieval_chunk_id in retrieval_chunk_ids:
                retrieval_memory = utils.load_memory_structure_torch(retrieval_memory_dir, f"level{level_id}_retrieve_{mode}", retrieval_chunk_id, int(params["GPU"]))
                # Now, match each retrieval entry with a title entry (in this chunk)
                for sentence_i in range(retrieval_memory.shape[0]):
                    retrieval_id = running_retrieval_line_index + sentence_i
                    current_title_id_and_dist_and_ec = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
                    if claims_to_covered_titles_ids_tensors[retrieval_id].shape[0] > 0:
                        # offset by current chunk
                        admitted_ids = claims_to_covered_titles_ids_tensors[retrieval_id]
                        # need to restrict the ids to those that appear in this chunk
                        admitted_ids = admitted_ids[admitted_ids.ge(running_memory_line_index) & admitted_ids.lt(running_memory_line_index+title_memory.shape[0])]
                        if admitted_ids.shape[0] > 0:
                            # if any remaining ids, next, need to renormalize the indexes to this chunk:
                            admitted_ids -= running_memory_line_index
                            # and then get the applicable subset
                            title_memory_subset = title_memory[admitted_ids.to(params["main_device"]), :]
                            l2_pairwise_distances = pdist(retrieval_memory[sentence_i, :].expand_as(title_memory_subset), title_memory_subset)
                            smallest_k_distances, smallest_k_distances_idx = torch.topk(l2_pairwise_distances, min(top_k_nearest_memories, title_memory_subset.shape[0]), largest=False, sorted=True)
                            for dist, dist_idx in zip(smallest_k_distances, smallest_k_distances_idx):
                                # dist_idx needs to be recast to the admitted_ids and then translated back to the
                                # original index
                                memory_id = running_memory_line_index + admitted_ids[dist_idx.item()].item()
                                error_correction_score = 0.0  # currently not using
                                current_title_id_and_dist_and_ec.append(
                                    [memory_id, dist.item(), error_correction_score])

                                # # currently, not using level-specific fc layers:
                                # abs_filter_diff = torch.abs(retrieval_memory[sentence_i, :] - title_memory_subset[dist_idx.item(), :])
                                # fc_out = model.layer1_fc(abs_filter_diff)
                                # error_correction_score = fc_out[1] - fc_out[0]
                                # current_title_id_and_dist_and_ec.append(
                                #     [memory_id, dist.item(), error_correction_score.item()])
                    # Sort and keep top_k_nearest_memories
                    retrieval_id_to_title_id_and_dist_and_ec[retrieval_id] = sorted(current_title_id_and_dist_and_ec, key=lambda x: x[1])[0:top_k_nearest_memories]
                # update retrieval line index:
                running_retrieval_line_index += retrieval_memory.shape[0]
            # update memory line index:
            running_memory_line_index += title_memory.shape[0]

    end_time = time.time()
    print(f"get_top_k_nearest_titles_from_memory time: {(end_time - start_time) / 60} minutes")
    return retrieval_id_to_title_id_and_dist_and_ec


def get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist, data, model, params, save_eval_output=False, mode="train", level_id=-1):

    assert level_id in [1, 2, 3]
    start_time = time.time()
    model.eval()

    print(f"Processing {mode} level {level_id}")
    retrieval_id_to_title_id_and_dist_and_ec = get_top_k_nearest_titles_from_memory(pdist, data, model, params,
                                                                                    mode, level_id)

    predicted_output = utils_search_eval_levels.eval_nearest_titles_from_memory_for_level(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data,
                                                  params, save_eval_output=save_eval_output, mode=mode, level_id=level_id)

    # note that the return title_ids are relative to the current level (for use in constructing hard negative examples)
    if data["only_level_1"]:
        print(f"--only_level_1 was provided, so skipping predicted data creation for levels 2 and 3.")
    else:
        if level_id in [1, 2]:
            if level_id == 2 and data["only_levels_1_and_2"]:
                # In this case, we are only training and evaluating levels 1 and 2, so we can skip creating the data for
                # level 3
                print(f"--only_levels_1_and_2 was provided, so skipping predicted data creation for level 3.")
            else:
                # for levels 2 and 3, the 'titles' do not exist so must be constructed from the results of the search of
                # the previous level
                data = update_data_structures_with_predicted_titles_for_next_level(retrieval_id_to_title_id_and_dist_and_ec,
                                                                                   data,
                                                                                   params, mode=mode, level_id=level_id)

    end_time = time.time()
    print(f"get_nearest_titles_from_memory_for_all_levels() time: {(end_time - start_time) / 60} minutes")

    return data, predicted_output


def update_data_structures_with_predicted_titles_for_next_level(retrieval_id_to_title_id_and_dist_and_ec, data, params, mode="train", level_id=-1):
    # Construct the sequences for levels 2 and 3 based on the results of the search
    assert level_id in [1, 2]
    start_time = time.time()
    # constant across levels:
    # unique_original_titles = data[f"{mode}_unique_original_titles"]
    # claims_to_unique_title_ids = data[f"{mode}_claims_to_chosen_title_ids"]
    # claims_to_true_titles_ids = data[f"{mode}_claims_to_true_titles_ids"]  # flat list (not separated by evidence sets)
    # claims_to_true_title_ids_evidence_sets = data[f"{mode}_claims_to_true_title_ids_evidence_sets"]
    # decision_labels = data[f"{mode}_decision_labels"]  # ground-truth decision labels for each claim

    # level-dependent:
    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]

    # CONSTRUCT
    # need to reset (may exist from previous epochs in training) and construct these for levels 2 and 3:
    next_level_id = level_id + 1
    new_max_length = params["max_length"] * next_level_id
    data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"] = []
    data[f"predicted_level{next_level_id}_{mode}_bert_idx_unique_titles"] = []
    data[f"predicted_level{next_level_id}_{mode}_bert_input_masks_unique_titles"] = []
    # We use the following to map to other data structures. Each is a tuple of len <= 2 in this version.
    data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids"] = []
    # tensor off admissible indexes (relative to current level)
    data[f"level{next_level_id}_{mode}_claims_to_covered_titles_ids_tensors"] = []
    if next_level_id == 3:
        data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"] = []
        data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids_marginalized"] = []

    #claims_to_covered_titles_ids_tensors = data[f"level{1}_{mode}_claims_to_covered_titles_ids_tensors"]
    # need to construct this for level 3 in order to relate results from retrieval back to final decision label
    #unique_title_ids_to_decision_labels = data[f"{mode}_unique_title_ids_to_decision_labels"]  # ground-truth decision labels for each unique title

    # for fever, the following provides a mapping from wiki sentences to wiki documents; this is used for eval purposes
    # (e.g., to check whether the ground truth documents are found)
    #unique_title_ids_to_document_ids = data[f"{mode}_unique_title_ids_to_document_ids"]

    assert len(data[f"{mode}_claims_to_chosen_title_ids"]) == len(retrieval_id_to_title_id_and_dist_and_ec), \
        f'{len(data[f"{mode}_claims_to_chosen_title_ids"])}, {len(retrieval_id_to_title_id_and_dist_and_ec)}'
    retrieval_id_check = 0
    for _ in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        assert retrieval_id_check in retrieval_id_to_title_id_and_dist_and_ec
        retrieval_id_check += 1

    for retrieval_id in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        title_id_and_dist_and_ec = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]

        next_level_claims_to_covered_tensor_ids = []
        if level_id == 2:
            selected_original_evidence_ids = []
        for k in range(top_k_nearest_memories):
            if k < len(title_id_and_dist_and_ec):
                title_meta_data = title_id_and_dist_and_ec[k]
                title_idx = title_meta_data[0]
                #title_distance = title_meta_data[1]
                # not currently used: error_correction_score = title_meta_data[2]

                if level_id == 1:
                    next_level_claims_to_covered_tensor_ids, data = \
                        construct_predicted_title_structures(new_max_length, constants.CONSIDER_STRING, mode, next_level_id,
                                                             retrieval_id, title_idx, data,
                                                             next_level_claims_to_covered_tensor_ids, second_title_idx=None, constuct_ground_truth=False)

                elif level_id == 2:
                    # remember, we have 1 level of indirection in order to recover the ORIGINAL title_ids: needs to go
                    # through: data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids"]

                    # For level 3, the evidence of 1 instances look similar to level 2 except the prefix is now the
                    # decision label (and the max length is longer). For evidence of 2, we first need to collect all of
                    # the original title_ids in the beam/top-k.

                    # convert title_idx to original id
                    original_evidence_id_tuple = data[f"predicted_level{level_id}_{mode}_original_evidence_ids"][title_idx]
                    # at level_id 2, original_evidence_id_tuple is always of length 1
                    assert len(original_evidence_id_tuple) == 1
                    # subsequently, we assume these are just int, so take first
                    selected_original_evidence_ids.append(original_evidence_id_tuple[0])
        if level_id == 2 and not data["do_not_marginalize_over_level3_evidence"]:
            # level 2 (for creating level 3) is different, because we marginalize over the evidence.
            # Note that currently we only use maxlength * 3 here, regardless of evidence size
            label_strings_to_consider = [constants.SUPPORTS_STRING, constants.REFUTES_STRING, constants.MOREINFO_STRING]
            label_ids_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID, constants.MOREINFO_ID]
            evidence_ids_to_consider = selected_original_evidence_ids[0:data["level3_top_k_evidence_predictions"]]
            if len(evidence_ids_to_consider) > 0:
                if params["only_2_class"] or data["constrain_to_2_class_at_inference"]:
                    label_strings_to_consider = [constants.SUPPORTS_STRING, constants.REFUTES_STRING]
                    label_ids_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID]
                for label_string, label_id in zip(label_strings_to_consider, label_ids_to_consider):
                    next_level_claims_to_covered_tensor_ids, data = \
                        construct_predicted_title_structures_with_titles_list(new_max_length, label_string, mode,
                                                                          next_level_id, retrieval_id, evidence_ids_to_consider,
                                                                          data, next_level_claims_to_covered_tensor_ids)

                    # The final decision is now part of the input sequence, so we need to keep track of those:
                    data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"].append(label_id)
                    assert len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"]) == \
                           len(data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"])

        if level_id == 2 and data["do_not_marginalize_over_level3_evidence"]:
            # (This is an alternative (and dramatically more expensive) search procedure for finding particular
            # evidence pairs. This is not necessary for FEVER.)
            #
            # level 2 (for creating level 3) is different, because we also need to perform a search over sets with 2
            # pieces of evidence.
            # First, we create 1 piece instances, and then we progressively add 2nd pieces from the beam, where to
            # reduce the space, we do not consider all unique possibilities (i.e., we make order matter).
            label_strings_to_consider = [constants.SUPPORTS_STRING, constants.REFUTES_STRING, constants.MOREINFO_STRING]
            label_ids_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID, constants.MOREINFO_ID]
            remaining_1_evidence_constructions = data["level3_max_1_evidence_constructions"]
            if params["only_2_class"] or data["constrain_to_2_class_at_inference"]:
                label_strings_to_consider = [constants.SUPPORTS_STRING, constants.REFUTES_STRING]
                label_ids_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID]
            for title_idx in selected_original_evidence_ids:  #  test only k=1: [0:1]:
                for label_string, label_id in zip(label_strings_to_consider, label_ids_to_consider):
                    next_level_claims_to_covered_tensor_ids, data = \
                        construct_predicted_title_structures(new_max_length, label_string, mode, next_level_id,
                                                             retrieval_id, title_idx, data,
                                                             next_level_claims_to_covered_tensor_ids, second_title_idx=None, constuct_ground_truth=False)
                    # The final decision is now part of the input sequence, so we need to keep track of those:
                    data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"].append(label_id)
                    assert len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"]) == \
                           len(data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"])
                # we wait to cut-off until all label strings have been seen
                remaining_1_evidence_constructions -= 1
                if remaining_1_evidence_constructions <= 0:
                    break

            remaining_2_evidence_constructions = data["level3_max_2_evidence_constructions"]
            # these counts include all applicable labels
            for beam_index in range(data["level3_top_k_stratifications"]):  # we add a 2nd evidence set to these beam instances
                if remaining_2_evidence_constructions <= 0:
                    break
                if beam_index < len(selected_original_evidence_ids):
                    first_title_idx = selected_original_evidence_ids[beam_index]
                    while True:
                        if remaining_2_evidence_constructions <= 0:
                            break
                        # check if there is a next beam_index from which to get the next evidence piece
                        # in the current setup, this must be *below* the current selection
                        second_beam_index = beam_index + 1
                        if second_beam_index < len(selected_original_evidence_ids):
                            second_title_idx = selected_original_evidence_ids[second_beam_index]
                            for label_string, label_id in zip(label_strings_to_consider, label_ids_to_consider):
                                next_level_claims_to_covered_tensor_ids, data = \
                                    construct_predicted_title_structures(new_max_length, label_string, mode, next_level_id,
                                                                         retrieval_id, first_title_idx, data,
                                                                         next_level_claims_to_covered_tensor_ids,
                                                                         second_title_idx=second_title_idx, constuct_ground_truth=False)
                                data[f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"].append(
                                    label_id)
                                assert len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"]) == \
                                       len(data[
                                            f"predicted_level{next_level_id}_{mode}_unique_titles_to_decision_labels"])
                            remaining_2_evidence_constructions -= 1
                            # we wait to cut-off until all label strings have been seen
                            if remaining_2_evidence_constructions <= 0:
                                break
                        else:
                            break
                else:
                    break

        data[f"level{next_level_id}_{mode}_claims_to_covered_titles_ids_tensors"].append(torch.LongTensor(next_level_claims_to_covered_tensor_ids))

    print(f'Total number of titles generated for level {next_level_id}: {len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"])}')
    assert len(data[f"level{next_level_id}_{mode}_claims_to_covered_titles_ids_tensors"]) == \
           len(data[f"level{1}_idx_{mode}_x"]), f"ERROR: The mapping from claims to new titles is mismatched."
    end_time = time.time()
    print(f"update_data_structures_with_predicted_titles_for_next_level time: {(end_time - start_time) / 60} minutes")
    return data


def get_original_evidence_sequence_without_padding(data, mode, title_idx):
    ##### Get the 1 or two original wikipedia sentences (word embeddings AND WordPiece embeddings):
    ### 1. word embeddings:
    final_token_index = data[f"{mode}_idx_unique_titles_final_index"][title_idx]
    retrieved_idx_unique_title = data[f"{mode}_idx_unique_titles"][title_idx][constants.PADDING_SIZE:final_token_index+1]
    ### 2. WordPiece embeddings:
    intial_cls_index = 0
    final_sep_index = data[f"{mode}_bert_idx_unique_titles_final_sep_index"][title_idx]
    # From title: drop [CLS] and [SEP] and any remaining padding after [SEP]
    retrieved_bert_idx_unique_title = data[f"{mode}_bert_idx_unique_titles"][title_idx][intial_cls_index+1:final_sep_index]
    retrieved_bert_input_masks_unique_title = data[f"{mode}_bert_input_masks_unique_titles"][title_idx][intial_cls_index+1:final_sep_index]
    return retrieved_idx_unique_title, retrieved_bert_idx_unique_title, retrieved_bert_input_masks_unique_title


def construct_predicted_title_structures(new_max_length, label_string, mode, next_level_id, retrieval_id, title_idx,
                                         data, next_level_claims_to_covered_tensor_ids, second_title_idx=None,
                                         constuct_ground_truth=False, constuct_ground_truth_negative=False,
                                         second_negative=False):
    # At level 1, we need to build the 'titles' for level 2. At level 1, the titles were simply the
    # wiki sentences. In level 2, the titles consist of:
    # constants.CONSIDER_STRING + Claim + Evidence, where Evidence is only *1* wiki sentence

    # First, get the evidence ('title') and claim, removing prefix and suffix padding. Note that
    # correct removal of padding is important, otherwise the input to the models (and in particular,
    # BERT) will be incorrect (or at least, unlike what was seen in pre-training and at other levels).
    # Some important points: The final BERT input must be well formed: [CLS] + SENTENCE + [SEP]. It is
    # important that only one set of the special symbols appear in the input. The other tricky points
    # are that truncation must be increased for subsequent layers (given additional info.) and that
    # all intermediate padding must be removed.

    # Note that each "instance" consists of 3 sequences: The word2vec embedding indexes; the WordPiece embeddings; and
    # the masks (for BERT) corresponding to the WordPiece embeddings.
    #new_max_length = params["max_length"] * next_level_id

    ##### Get the 1 or two original wikipedia sentences (word embeddings AND WordPiece embeddings):
    retrieved_idx_unique_title, retrieved_bert_idx_unique_title, retrieved_bert_input_masks_unique_title = \
        get_original_evidence_sequence_without_padding(data, mode, title_idx)
    if second_title_idx is not None:
        second_retrieved_idx_unique_title, second_retrieved_bert_idx_unique_title, \
        second_retrieved_bert_input_masks_unique_title = \
            get_original_evidence_sequence_without_padding(data, mode, second_title_idx)

    ##### Get the claim with level 1 max_length (word embeddings AND WordPiece embeddings)
    ### 1. word embeddings:
    final_token_index = data[f"level{1}_{mode}_idx_x_final_index"][retrieval_id]  # always use level1 claims for concat (due to max_length)
    claim_idx = data[f"level{1}_idx_{mode}_x"][retrieval_id][constants.PADDING_SIZE:final_token_index+1]
    ### 2. WordPiece embeddings:
    # From claim: drop [CLS] and [SEP] and any remaining padding after [SEP]
    intial_cls_index = 0
    final_sep_index = data[f"level{1}_{mode}_bert_idx_final_sep_index"][retrieval_id]
    claim_bert_idx = data[f"level{1}_{mode}_bert_idx_sentences"][retrieval_id][intial_cls_index+1:final_sep_index]
    claim_bert_input_masks = data[f"level{1}_{mode}_bert_input_masks"][retrieval_id][intial_cls_index+1:final_sep_index]

    ##### Now construct the new title
    ### 1. word embeddings:
    new_idx_unique_title = data[f"{label_string}_idx_sentence"] + claim_idx + retrieved_idx_unique_title
    if second_title_idx is not None:
        new_idx_unique_title += second_retrieved_idx_unique_title
    # truncate
    new_idx_unique_title = new_idx_unique_title[0:constants.PADDING_SIZE+new_max_length]
    # add trailing padding (prefix padding already added via data[f"{label_string}_idx_sentence"]):
    new_idx_unique_title.extend(
        [constants.PAD_SYM_ID] * (new_max_length + 2*constants.PADDING_SIZE - len(new_idx_unique_title)))

    ### 2. WordPiece embeddings:
    new_bert_idx_unique_title = data[f"{label_string}_bert_idx_sentence"] + claim_bert_idx + retrieved_bert_idx_unique_title # prefix + claim + evidence
    new_bert_input_masks_unique_title = data[f"{label_string}_bert_input_mask"] + claim_bert_input_masks + retrieved_bert_input_masks_unique_title  # prefix + claim + evidence
    if second_title_idx is not None:
        new_bert_idx_unique_title += second_retrieved_bert_idx_unique_title
        new_bert_input_masks_unique_title += second_retrieved_bert_input_masks_unique_title
    # truncate and re-add [SEP] AND mask
    new_bert_idx_unique_title = new_bert_idx_unique_title[0:new_max_length+1]  # +1 to keep initial [CLS]
    new_bert_idx_unique_title.append(data[f"bert_sep_sym_bert_idx"])  # id for [SEP]
    new_bert_input_masks_unique_title = new_bert_input_masks_unique_title[0:new_max_length+1]  # +1 to keep initial [CLS]
    new_bert_input_masks_unique_title.append(1)  # for the [SEP]

    new_bert_idx_unique_title.extend([0] * ((new_max_length + 2) - len(new_bert_idx_unique_title)))
    new_bert_input_masks_unique_title.extend([0] * ((new_max_length + 2) - len(new_bert_input_masks_unique_title)))

    assert len(new_idx_unique_title) == new_max_length + 2 * constants.PADDING_SIZE
    assert len(new_idx_unique_title) == len(new_bert_idx_unique_title) + 2 * constants.PADDING_SIZE - 2 and len(
        new_bert_idx_unique_title) == len(new_bert_input_masks_unique_title)

    if not constuct_ground_truth and not constuct_ground_truth_negative:
        data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"].append(new_idx_unique_title)
        data[f"predicted_level{next_level_id}_{mode}_bert_idx_unique_titles"].append(new_bert_idx_unique_title)
        data[f"predicted_level{next_level_id}_{mode}_bert_input_masks_unique_titles"].append(new_bert_input_masks_unique_title)
        if second_title_idx is not None:
            data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids"].append((title_idx, second_title_idx))  # 2nd will often be None
        else:
            data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids"].append((title_idx,))
        # now the 'covered titles' are these newly constructed structures; we subsequently index in the
        # same manner as in level1:
        next_level_claims_to_covered_tensor_ids.append(len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"])-1)

        return next_level_claims_to_covered_tensor_ids, data
    elif constuct_ground_truth:
        assert not constuct_ground_truth_negative
        data[f"chosen_level{next_level_id}_{mode}_idx_unique_titles"].append(new_idx_unique_title)
        data[f"chosen_level{next_level_id}_{mode}_bert_idx_unique_titles"].append(new_bert_idx_unique_title)
        data[f"chosen_level{next_level_id}_{mode}_bert_input_masks_unique_titles"].append(
            new_bert_input_masks_unique_title)
        return data
    elif constuct_ground_truth_negative:
        assert not constuct_ground_truth
        if not second_negative:
            data[f"neg_chosen_level{next_level_id}_{mode}_idx_unique_titles"].append(new_idx_unique_title)
            data[f"neg_chosen_level{next_level_id}_{mode}_bert_idx_unique_titles"].append(new_bert_idx_unique_title)
            data[f"neg_chosen_level{next_level_id}_{mode}_bert_input_masks_unique_titles"].append(
                new_bert_input_masks_unique_title)
        else:
            data[f"neg2_chosen_level{next_level_id}_{mode}_idx_unique_titles"].append(new_idx_unique_title)
            data[f"neg2_chosen_level{next_level_id}_{mode}_bert_idx_unique_titles"].append(new_bert_idx_unique_title)
            data[f"neg2_chosen_level{next_level_id}_{mode}_bert_input_masks_unique_titles"].append(
                new_bert_input_masks_unique_title)
        return data


def construct_predicted_title_structures_with_titles_list(new_max_length, label_string, mode, next_level_id, retrieval_id, title_idx_list,
                                         data, next_level_claims_to_covered_tensor_ids):
    assert len(title_idx_list) > 0
    # Construct the titles ('support sequences') from multiple pieces of evidence (i.e., support sequences from earlier
    # levels)

    # First, get the evidence ('title') and claim, removing prefix and suffix padding. Note that
    # correct removal of padding is important, otherwise the input to the models (and in particular,
    # BERT) will be incorrect (or at least, unlike what was seen in pre-training and at other levels).
    # Some important points: The final BERT input must be well formed: [CLS] + SENTENCE + [SEP]. It is
    # important that only one set of the special symbols appear in the input. The other tricky points
    # are that truncation must be increased for subsequent layers (given additional info.) and that
    # all intermediate padding must be removed.

    # Note that each "instance" consists of 3 sequences: The word2vec embedding indexes; the WordPiece embeddings; and
    # the masks (for BERT) corresponding to the WordPiece embeddings.
    #new_max_length = params["max_length"] * next_level_id

    ##### Get the original wikipedia sentences (word embeddings AND WordPiece embeddings):
    retrieved_idx_unique_titles, retrieved_bert_idx_unique_titles, retrieved_bert_input_masks_unique_titles = [], [], []
    for title_idx in title_idx_list:
        retrieved_idx_unique_title, retrieved_bert_idx_unique_title, retrieved_bert_input_masks_unique_title = \
            get_original_evidence_sequence_without_padding(data, mode, title_idx)
        retrieved_idx_unique_titles.append(retrieved_idx_unique_title)
        retrieved_bert_idx_unique_titles.append(retrieved_bert_idx_unique_title)
        retrieved_bert_input_masks_unique_titles.append(retrieved_bert_input_masks_unique_title)

    ##### Get the claim with level 1 max_length (word embeddings AND WordPiece embeddings)
    ### 1. word embeddings:
    final_token_index = data[f"level{1}_{mode}_idx_x_final_index"][retrieval_id]  # always use level1 claims for concat (due to max_length)
    claim_idx = data[f"level{1}_idx_{mode}_x"][retrieval_id][constants.PADDING_SIZE:final_token_index+1]
    ### 2. WordPiece embeddings:
    # From claim: drop [CLS] and [SEP] and any remaining padding after [SEP]
    intial_cls_index = 0
    final_sep_index = data[f"level{1}_{mode}_bert_idx_final_sep_index"][retrieval_id]
    claim_bert_idx = data[f"level{1}_{mode}_bert_idx_sentences"][retrieval_id][intial_cls_index+1:final_sep_index]
    claim_bert_input_masks = data[f"level{1}_{mode}_bert_input_masks"][retrieval_id][intial_cls_index+1:final_sep_index]

    ##### Now construct the new title
    ### 1. word embeddings:
    new_idx_unique_title = data[f"{label_string}_idx_sentence"] + claim_idx
    for i in range(len(title_idx_list)):
        new_idx_unique_title += retrieved_idx_unique_titles[i]
    # truncate
    new_idx_unique_title = new_idx_unique_title[0:constants.PADDING_SIZE+new_max_length]
    # add trailing padding (prefix padding already added via data[f"{label_string}_idx_sentence"]):
    new_idx_unique_title.extend(
        [constants.PAD_SYM_ID] * (new_max_length + 2*constants.PADDING_SIZE - len(new_idx_unique_title)))

    ### 2. WordPiece embeddings:
    new_bert_idx_unique_title = data[f"{label_string}_bert_idx_sentence"] + claim_bert_idx   # prefix + claim + evidence
    new_bert_input_masks_unique_title = data[f"{label_string}_bert_input_mask"] + claim_bert_input_masks  # prefix + claim + evidence
    for i in range(len(title_idx_list)):
        new_bert_idx_unique_title += retrieved_bert_idx_unique_titles[i]
        new_bert_input_masks_unique_title += retrieved_bert_input_masks_unique_titles[i]
    # truncate and re-add [SEP] AND mask
    new_bert_idx_unique_title = new_bert_idx_unique_title[0:new_max_length+1]  # +1 to keep initial [CLS]
    new_bert_idx_unique_title.append(data[f"bert_sep_sym_bert_idx"])  # id for [SEP]
    new_bert_input_masks_unique_title = new_bert_input_masks_unique_title[0:new_max_length+1]  # +1 to keep initial [CLS]
    new_bert_input_masks_unique_title.append(1)  # for the [SEP]

    new_bert_idx_unique_title.extend([0] * ((new_max_length + 2) - len(new_bert_idx_unique_title)))
    new_bert_input_masks_unique_title.extend([0] * ((new_max_length + 2) - len(new_bert_input_masks_unique_title)))

    assert len(new_idx_unique_title) == new_max_length + 2 * constants.PADDING_SIZE
    assert len(new_idx_unique_title) == len(new_bert_idx_unique_title) + 2 * constants.PADDING_SIZE - 2 and len(
        new_bert_idx_unique_title) == len(new_bert_input_masks_unique_title)


    data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"].append(new_idx_unique_title)
    data[f"predicted_level{next_level_id}_{mode}_bert_idx_unique_titles"].append(new_bert_idx_unique_title)
    data[f"predicted_level{next_level_id}_{mode}_bert_input_masks_unique_titles"].append(new_bert_input_masks_unique_title)

    # in this case, we just use the first title as the original evidence, and keep all ids in a separate structure
    # for later analyses
    data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids"].append((title_idx_list[0],))
    data[f"predicted_level{next_level_id}_{mode}_original_evidence_ids_marginalized"].append(tuple(title_idx_list))
    # now the 'covered titles' are these newly constructed structures; we subsequently index in the
    # same manner as in level1:
    next_level_claims_to_covered_tensor_ids.append(len(data[f"predicted_level{next_level_id}_{mode}_idx_unique_titles"])-1)

    return next_level_claims_to_covered_tensor_ids, data

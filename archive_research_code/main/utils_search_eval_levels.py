from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
import utils_search_eval_levels_formats
import utils_search_eval_levels_constrained

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

import copy


def eval_nearest_titles_from_memory_for_level(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data, params, save_eval_output=False, mode="train", level_id=-1):
    assert level_id in [1, 2, 3]
    #  These are just subtly different enough that it is cleaner to just separate out the eval for the final level:
    if level_id in [1, 2]:
        return eval_nearest_titles_from_memory_for_level1_or_level2(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data,
                                                   params, save_eval_output=save_eval_output, mode=mode, level_id=level_id)
    elif level_id == 3:
        if data["save_output_for_ec"]:
            return utils_search_eval_levels_formats.eval_nearest_titles_from_memory_for_level3_ec_format(predicted_output,
                                                              retrieval_id_to_title_id_and_dist_and_ec, data,
                                                              params, save_eval_output=save_eval_output, mode=mode,
                                                              level_id=level_id)
        elif data["eval_constrained"]:
            return utils_search_eval_levels_constrained.eval_nearest_titles_from_memory_for_level3_constrained(predicted_output,
                                                              retrieval_id_to_title_id_and_dist_and_ec, data,
                                                              params, save_eval_output=save_eval_output, mode=mode,
                                                              level_id=level_id)
        else:
            return eval_nearest_titles_from_memory_for_level3(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data,
                                                       params, save_eval_output=save_eval_output, mode=mode, level_id=level_id)


def eval_nearest_titles_from_memory_for_level1_or_level2(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data, params, save_eval_output=False, mode="train", level_id=-1):

    # For level 2, the chosen ground truth is only 1 wiki sentence, but we also take care to not push away any valid second
    # evidence sentences in the chosen set, if applicable.

    # The primary differences with level 1:
    # In level 2, we have to be careful to relate the ids of the memory structure back to the original title ids.
    # In level 2, we avoid pushing away evidence in the chosen set, but in downstream training, we only use the
    # first piece of evidence.

    # retrieval metrics are for the supports/refutes cases
    assert level_id in [1, 2]

    # use * to indicate correct title
    start_time = time.time()

    unique_original_titles = data[f"{mode}_unique_original_titles"]
    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]
    claims_to_chosen_title_ids = data[f"{mode}_claims_to_chosen_title_ids"]
    claims_to_true_titles_ids = data[f"{mode}_claims_to_true_titles_ids"]
    claims_to_true_title_ids_evidence_sets = data[f"{mode}_claims_to_true_title_ids_evidence_sets"]
    decision_labels = data[f"{mode}_decision_labels"]  # ground-truth decision labels for each claim
    #unique_title_ids_to_decision_labels = data[f"{mode}_unique_title_ids_to_decision_labels"]  # ground-truth decision labels for each unique title

    if level_id in [2, 3]:
        predicted_original_evidence_ids = data[f"predicted_level{level_id}_{mode}_original_evidence_ids"]
    if level_id == 3:
        unique_titles_to_decision_labels = data[f"predicted_level{level_id}_{mode}_unique_titles_to_decision_labels"]

    # for fever, the following provides a mapping from wiki sentences to wiki documents; this is used for eval purposes
    # (e.g., to check whether the ground truth documents are found)
    unique_title_ids_to_document_ids = data[f"{mode}_unique_title_ids_to_document_ids"]

    if save_eval_output:
        assert mode in ["dev", "test"], f"ERROR: Currently saving eval output is only available for the dev or test splits."
        #  The only thing keeping this from working for train is to push save train_original_sentences in utils.py AND
        #  (critically), train_original_sentences must be shuffled with the rest of the training data structures.
        #original_sentences = data[f"{mode}_original_sentences"]
        original_titles = data[f"{mode}_original_titles"]

    assert len(claims_to_chosen_title_ids) == len(retrieval_id_to_title_id_and_dist_and_ec), \
        f"{len(claims_to_chosen_title_ids)}, {len(retrieval_id_to_title_id_and_dist_and_ec)}"
    retrieval_id_check = 0
    for _ in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        assert retrieval_id_check in retrieval_id_to_title_id_and_dist_and_ec
        retrieval_id_check += 1

    eval_output_compact = []

    nearest_wrong_level_title_ids = []
    true_top_retrieval_predictions_full_evidence = []  # only 1 if all of size 2 evidence is present in the chosen
    predictions_at_least_1_correct_title_in_beam = []
    predictions_at_least_1_correct_document_in_beam = []
    predictions_k0_at_least_1_correct_document_in_beam = []  # top retrieval is correct document

    # When recording distances for level 3 constraining, we currently only considered the top of the beam, if present
    distances_top_of_beam_all = []  # -1 if predicting unk
    distances_top_of_beam_true_predictions = []
    distances_top_of_beam_false_predictions = []
    distances_top_of_beam_unverifiable_claims = []

    for retrieval_id in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        chosen_title_ids_tuple = claims_to_chosen_title_ids[retrieval_id]
        assert isinstance(chosen_title_ids_tuple, tuple)
        if level_id == 1:
            title_id_and_dist_and_ec = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
        elif level_id == 2:  # in this case, need to convert relative ids back to the original ids
            title_id_and_dist_and_ec_relative_to_level = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
            title_id_and_dist_and_ec = copy.deepcopy(title_id_and_dist_and_ec_relative_to_level)
            for predicted_i in range(len(title_id_and_dist_and_ec)):
                # convert relative ids back to the original ids
                predicted_title_id = title_id_and_dist_and_ec[predicted_i][0]
                original_evidence_id_tuple = predicted_original_evidence_ids[predicted_title_id]
                assert len(original_evidence_id_tuple) == 1, f"ERROR: 1-1 mapping to original titles expected in level 2."
                title_id_and_dist_and_ec[predicted_i][0] = original_evidence_id_tuple[0]

        retrieval_is_correct = False
        nearest_wrong_title_id = constants.UNK_TITLE_ID
        if len(title_id_and_dist_and_ec) >= len(chosen_title_ids_tuple):
            if len(chosen_title_ids_tuple) == 1:
                first_predicted_title_id = title_id_and_dist_and_ec[0][0]
                predicted_retrieval_id_tuple = (first_predicted_title_id,)
                retrieval_is_correct = chosen_title_ids_tuple == predicted_retrieval_id_tuple
                if retrieval_is_correct:
                    if len(title_id_and_dist_and_ec) >= 2:
                        if level_id == 1:
                            nearest_wrong_title_id = title_id_and_dist_and_ec[1][0]
                        else:
                            nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[1][0]
                else:
                    if level_id == 1:
                        nearest_wrong_title_id = first_predicted_title_id
                    else:
                        nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
            else:
                first_predicted_title_id = title_id_and_dist_and_ec[0][0]
                second_predicted_title_id = title_id_and_dist_and_ec[1][0]
                predicted_retrieval_id_tuple = (first_predicted_title_id, second_predicted_title_id)
                retrieval_is_correct = chosen_title_ids_tuple == predicted_retrieval_id_tuple
                if retrieval_is_correct:
                    if len(title_id_and_dist_and_ec) >= 3:
                        if level_id == 1:
                            nearest_wrong_title_id = title_id_and_dist_and_ec[2][0]
                        else:
                            nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[2][0]
                else:
                    # for training, do not want to pick any in true set, even if out of order, since would
                    # contradict reference chosen positive
                    if first_predicted_title_id not in chosen_title_ids_tuple:
                        if level_id == 1:
                            nearest_wrong_title_id = first_predicted_title_id
                        else:
                            nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
                    elif second_predicted_title_id not in chosen_title_ids_tuple:
                        if level_id == 1:
                            nearest_wrong_title_id = second_predicted_title_id
                        else:
                            nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[1][0]
                    else:  # this could occur if the top 2 are correct, but in a different order
                        if len(title_id_and_dist_and_ec) >= 3:
                            if level_id == 1:
                                nearest_wrong_title_id = title_id_and_dist_and_ec[2][0]
                            else:
                                nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[2][0]
            # these tp/fp distances are only calculated for non-unk predictions; these are intended to be recorded on
            # the training set (with labels), and subsequently used to constrain inference
            predicted_title_distance = title_id_and_dist_and_ec[0][1]
            if decision_labels[retrieval_id] != constants.MOREINFO_ID:
                # only calculate retrieval metrics for Supports/Refutes
                if retrieval_is_correct:
                    distances_top_of_beam_true_predictions.append(predicted_title_distance)
                else:
                    distances_top_of_beam_false_predictions.append(predicted_title_distance)
            else:
                distances_top_of_beam_unverifiable_claims.append(predicted_title_distance)

        if decision_labels[retrieval_id] != constants.MOREINFO_ID:
            # only calculate retrieval metrics for Supports/Refutes
            true_top_retrieval_predictions_full_evidence.append(int(retrieval_is_correct))
        nearest_wrong_level_title_ids.append(nearest_wrong_title_id)

        if save_eval_output:
            one_claim_output_line_compact = ""
        at_least_1_correct_title_in_beam = False
        at_least_1_correct_document_in_beam = False
        k0_at_least_1_correct_document_in_beam = False
        # get the set of true document-level id's
        true_document_set = set()
        for one_true_title_id in claims_to_true_titles_ids[retrieval_id]:
            if one_true_title_id in unique_title_ids_to_document_ids:
                true_document_set.add(unique_title_ids_to_document_ids[one_true_title_id])

        predicted_title_distance = -1.0
        for k in range(top_k_nearest_memories):
            if k < len(title_id_and_dist_and_ec):
                title_meta_data = title_id_and_dist_and_ec[k]
                predicted_title_id = title_meta_data[0]
                if k == 0:
                    predicted_title_distance = title_meta_data[1]
                #error_correction_score = title_meta_data[2]
                predicted_title_string = ' '.join(unique_original_titles[predicted_title_id])
                if predicted_title_id in chosen_title_ids_tuple:
                    at_least_1_correct_title_in_beam = True
                    predicted_title_string = f"*{predicted_title_string}*"

                if unique_title_ids_to_document_ids[predicted_title_id] in true_document_set:
                    at_least_1_correct_document_in_beam = True
                    predicted_title_string = f"@{predicted_title_string}@"
                    if k == 0:
                        k0_at_least_1_correct_document_in_beam = True

                if save_eval_output:
                    one_claim_output_line_compact += f"\t{k}: {predicted_title_string}\n"
        distances_top_of_beam_all.append(predicted_title_distance)
        if decision_labels[retrieval_id] != constants.MOREINFO_ID:
            # only calculate retrieval metrics for Supports/Refutes
            predictions_at_least_1_correct_title_in_beam.append(int(at_least_1_correct_title_in_beam))
            predictions_at_least_1_correct_document_in_beam.append(int(at_least_1_correct_document_in_beam))
            predictions_k0_at_least_1_correct_document_in_beam.append(int(k0_at_least_1_correct_document_in_beam))
        if save_eval_output:
            full_output_line_compact = f"Level {level_id}: Sentence {retrieval_id}; Is unverifiable: {decision_labels[retrieval_id]==constants.MOREINFO_ID}; Exact Retrieval Match: {retrieval_is_correct};" \
                                       f" Partial Retrieval Match in Beam: {at_least_1_correct_title_in_beam}; " \
                                       f"Partial Document-Level Retrieval Match in Beam: {at_least_1_correct_document_in_beam}; k=0 retrieval distance: {predicted_title_distance}\n" \
                                       f"\tReference: {' '.join(original_titles[retrieval_id])}\n"
            full_output_line_compact += one_claim_output_line_compact
            eval_output_compact.append(full_output_line_compact)


    print(
        f"\tEval ({mode}) level{level_id}: Number of correct retrieval predictions (strict: complete evidence match within top 2): "
        f"{np.sum(true_top_retrieval_predictions_full_evidence)} out of {len(true_top_retrieval_predictions_full_evidence)}; "
        f"proportion: {np.mean(true_top_retrieval_predictions_full_evidence)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct retrieval predictions (relaxed: at least 1 partial match within top {top_k_nearest_memories}): "
        f"{np.sum(predictions_at_least_1_correct_title_in_beam)} out of {len(predictions_at_least_1_correct_title_in_beam)}; "
        f"proportion: {np.mean(predictions_at_least_1_correct_title_in_beam)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of retrieval predictions containing at least 1 correct document in retrieval beam (relaxed: at least 1 partial match within top {top_k_nearest_memories}): "
        f"{np.sum(predictions_at_least_1_correct_document_in_beam)} out of {len(predictions_at_least_1_correct_document_in_beam)}; "
        f"proportion: {np.mean(predictions_at_least_1_correct_document_in_beam)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of retrieval predictions with at least 1 correct document appearing at top of retrieval beam: "
        f"{np.sum(predictions_k0_at_least_1_correct_document_in_beam)} out of {len(predictions_k0_at_least_1_correct_document_in_beam)}; "
        f"proportion: {np.mean(predictions_k0_at_least_1_correct_document_in_beam)}")

    print_distance_stats(mode, level_id, distances_top_of_beam_all, distances_top_of_beam_true_predictions,
                         distances_top_of_beam_false_predictions, distances_top_of_beam_unverifiable_claims)

    predicted_output[f"level{level_id}_{mode}_nearest_wrong_level_title_ids"] = nearest_wrong_level_title_ids
    predicted_output[f"level{level_id}_{mode}_retrieval_acc"] = np.mean(true_top_retrieval_predictions_full_evidence)
    predicted_output[f"level{level_id}_{mode}_decision_acc"] = 0.0
    predicted_output[f"level{level_id}_{mode}_score_vals"] = []
    predicted_output[f"level{level_id}_{mode}_score_vals_compact"] = eval_output_compact
    predicted_output[f"level{level_id}_{mode}_retrieval_distances_at_top_of_beam"] = distances_top_of_beam_all

    if data["create_exemplar_database"] or data["create_exemplar_query"] or data["save_exemplar_output"] or data["visualize_alignment"]:
        predicted_output[f"level{level_id}_{mode}_retrieval_id_to_title_id_and_dist_and_ec"] = retrieval_id_to_title_id_and_dist_and_ec

    end_time = time.time()
    print(f"eval_nearest_titles_from_memory time: {(end_time - start_time) / 60} minutes")

    return predicted_output


def print_distance_stats(mode, level_id, distances_top_of_beam_all, distances_top_of_beam_true_predictions,
                         distances_top_of_beam_false_predictions, distances_top_of_beam_unverifiable_claims):
    # distances_top_of_beam_all may have some -1 entries for unk predictions; we exclude those in the stats below
    distances_top_of_beam_all_only_covered = [x for x in distances_top_of_beam_all if x != -1.0]
    try:
        print(
            f"\tEval ({mode}) level{level_id}: Retrieval distances at top of the beam for all covered predictions: "
            f"mean: {np.mean(distances_top_of_beam_all_only_covered)}; "
            f"std: {np.std(distances_top_of_beam_all_only_covered)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered correctly predicted retrieval: Retrieval distances (k=0) "
            f"mean: {np.mean(distances_top_of_beam_true_predictions)}; "
            f"std: {np.std(distances_top_of_beam_true_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered wrongly predicted retrieval: Retrieval distances (k=0) "
            f"mean: {np.mean(distances_top_of_beam_false_predictions)}; "
            f"std: {np.std(distances_top_of_beam_false_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        if len(distances_top_of_beam_unverifiable_claims) > 0:
            print(
                f"\tEval ({mode}) level{level_id}: Among unverifiable predicted retrieval: Retrieval distances (k=0) "
                f"mean: {np.mean(distances_top_of_beam_unverifiable_claims)}; "
                f"std: {np.std(distances_top_of_beam_unverifiable_claims)}")
    except:
        print(f"Unexpected distance output")


def eval_nearest_titles_from_memory_for_level3(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data, params, save_eval_output=False, mode="train", level_id=-1):

    # Note that in this case, retrieval metrics are determined by the top of the beam, and only considered for the
    # verifiable claims
    # Note that for the marginalizing-evidence case, use --save_output_for_ec for saving output, since the following
    # only saves the top of the beam for debugging purposes.

    assert level_id == 3

    # use * to indicate correct title
    start_time = time.time()

    unique_original_titles = data[f"{mode}_unique_original_titles"]
    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]
    claims_to_chosen_title_ids = data[f"{mode}_claims_to_chosen_title_ids"]
    claims_to_true_titles_ids = data[f"{mode}_claims_to_true_titles_ids"]
    claims_to_true_title_ids_evidence_sets = data[f"{mode}_claims_to_true_title_ids_evidence_sets"]
    decision_labels = data[f"{mode}_decision_labels"]  # ground-truth decision labels for each claim
    #unique_title_ids_to_decision_labels = data[f"{mode}_unique_title_ids_to_decision_labels"]  # ground-truth decision labels for each unique title

    if level_id in [2, 3]:
        predicted_original_evidence_ids = data[f"predicted_level{level_id}_{mode}_original_evidence_ids"]
        #print(predicted_original_evidence_ids)

    if level_id == 3:
        unique_titles_to_decision_labels = data[f"predicted_level{level_id}_{mode}_unique_titles_to_decision_labels"]
        #print(f"decision: {unique_titles_to_decision_labels}")
        #exit()

    # for fever, the following provides a mapping from wiki sentences to wiki documents; this is used for eval purposes
    # (e.g., to check whether the ground truth documents are found)
    unique_title_ids_to_document_ids = data[f"{mode}_unique_title_ids_to_document_ids"]

    if save_eval_output:
        assert mode in ["dev", "test"], f"ERROR: Currently saving eval output is only available for the dev or test splits."
        #  The only thing keeping this from working for train is to push save train_original_sentences in utils.py AND
        #  (critically), train_original_sentences must be shuffled with the rest of the training data structures.
        #original_sentences = data[f"{mode}_original_sentences"]
        original_titles = data[f"{mode}_original_titles"]

    assert len(claims_to_chosen_title_ids) == len(retrieval_id_to_title_id_and_dist_and_ec), \
        f"{len(claims_to_chosen_title_ids)}, {len(retrieval_id_to_title_id_and_dist_and_ec)}"
    retrieval_id_check = 0
    for _ in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        assert retrieval_id_check in retrieval_id_to_title_id_and_dist_and_ec
        retrieval_id_check += 1

    eval_output_compact = []

    nearest_true_level_title_ids = []
    nearest_wrong_level_title_ids = []
    nearest_neg2_level_title_ids = []
    true_top_retrieval_predictions_full_evidence = []  # only 1 if all of size 2 evidence is present in the chosen
    true_top_retrieval_predictions_full_evidence_partially_correct = []  # 1 if at least 1 evidence match
    predictions_at_least_1_correct_title_in_beam = []
    predictions_at_least_1_correct_document_in_beam = []
    predictions_k0_at_least_1_correct_document_in_beam = []  # top retrieval is correct document

    true_top_decision_predictions = []
    predictions_at_least_1_correct_decision_in_beam = []

    num_nearest_wrong_has_correct_decision_label = 0
    num_nearest_wrong_title_id_not_found = 0
    num_nearest_neg2_title_id_not_found = 0
    for retrieval_id in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        chosen_title_ids_tuple = claims_to_chosen_title_ids[retrieval_id]
        assert isinstance(chosen_title_ids_tuple, tuple)
        # in this case, need to convert relative ids back to the original ids
        title_id_and_dist_and_ec_relative_to_level = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
        title_id_and_dist_and_ec_original_tuples_and_decision = copy.deepcopy(title_id_and_dist_and_ec_relative_to_level)
        for predicted_i in range(len(title_id_and_dist_and_ec_original_tuples_and_decision)):
            # convert relative ids back to the original ids; now first position is a tuple, not an int
            predicted_title_id = title_id_and_dist_and_ec_original_tuples_and_decision[predicted_i][0]
            original_evidence_id_tuple = predicted_original_evidence_ids[predicted_title_id]
            assert len(
                original_evidence_id_tuple) <= 2, f"ERROR: 1 to <=2 mapping to original titles expected in level 3."
            title_id_and_dist_and_ec_original_tuples_and_decision[predicted_i][0] = original_evidence_id_tuple
            # ec position gets overwritten with the decision label
            title_id_and_dist_and_ec_original_tuples_and_decision[predicted_i][2] = unique_titles_to_decision_labels[predicted_title_id]

        decision_is_correct = False
        retrieval_is_correct = False
        retrieval_is_partially_correct = False
        nearest_true_title_id = constants.UNK_TITLE_ID
        nearest_wrong_title_id = constants.UNK_TITLE_ID
        nearest_neg2_title_id = constants.UNK_TITLE_ID

        predicted_decision_id = -1
        if len(title_id_and_dist_and_ec_original_tuples_and_decision) >= 1:
            predicted_decision_id = title_id_and_dist_and_ec_original_tuples_and_decision[0][2]
            if predicted_decision_id == decision_labels[retrieval_id]:
                decision_is_correct = True

            predicted_retrieval_id_tuple = title_id_and_dist_and_ec_original_tuples_and_decision[0][0]
            retrieval_is_correct = chosen_title_ids_tuple == predicted_retrieval_id_tuple
            # alternatively, we could require decision AND retrieval to be correct, but here we aim to emulate inference
            # if retrieval_is_correct:
            if decision_is_correct:  # and retrieval_is_correct:
                nearest_true_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
                # in this version, we're not searching across instances (beyond label switches) in level 3,
                # so in this case, we can just take subsequent beam positions:
                k = 1
                if len(title_id_and_dist_and_ec_original_tuples_and_decision) > k:
                    assert title_id_and_dist_and_ec_original_tuples_and_decision[k][2] != decision_labels[retrieval_id]
                    nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]
                k = 2
                if len(title_id_and_dist_and_ec_original_tuples_and_decision) > k:
                    assert title_id_and_dist_and_ec_original_tuples_and_decision[k][2] != decision_labels[retrieval_id]
                    nearest_neg2_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]

                # k = 1
                # while k < len(title_id_and_dist_and_ec_original_tuples_and_decision):
                #     if title_id_and_dist_and_ec_original_tuples_and_decision[k][2] != decision_labels[retrieval_id]:
                #         nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]
                #         # In this case, the chosen always has the wrong label. The following records second position
                #         # in beam:
                #         if title_id_and_dist_and_ec_original_tuples_and_decision[1][2] == decision_labels[retrieval_id]:
                #             num_nearest_wrong_has_correct_decision_label += 1
                #         break
                #     else:
                #         k += 1
                # Alternatively, we can just always take the second position in the beam, but there's an edge case
                # (when using decision_is_correct) in which the predicted could have the right label (but wrong
                # retrieval), and the 2nd beam position has the right label and the correct retrieval
                # (i.e., matches the gold reference), which would contradict the min-max. (This never occurs if
                # --level3_max_1_evidence_constructions is 1.)
                # if len(title_id_and_dist_and_ec_original_tuples_and_decision) >= 2:
                #     nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[1][0]
                #     if title_id_and_dist_and_ec_original_tuples_and_decision[1][2] == decision_labels[retrieval_id]:
                #         num_nearest_wrong_has_correct_decision_label += 1
            else:
                nearest_wrong_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
                k = 1
                while k < len(title_id_and_dist_and_ec_original_tuples_and_decision):
                    if title_id_and_dist_and_ec_original_tuples_and_decision[k][2] == decision_labels[retrieval_id] and nearest_true_title_id == constants.UNK_TITLE_ID:
                        nearest_true_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]
                    elif title_id_and_dist_and_ec_original_tuples_and_decision[k][2] != decision_labels[retrieval_id] and nearest_neg2_title_id == constants.UNK_TITLE_ID:
                        nearest_neg2_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]
                    else:
                        k += 1
                    if nearest_true_title_id != constants.UNK_TITLE_ID and nearest_neg2_title_id != constants.UNK_TITLE_ID:
                        break
                # search for nearest correct in beam; in basic setup (with constructed predicted to marginalize over),
                # this will always occur in 2nd position, if present
                # k = 1
                # while k < len(title_id_and_dist_and_ec_original_tuples_and_decision):
                #     if title_id_and_dist_and_ec_original_tuples_and_decision[k][2] == decision_labels[retrieval_id]:
                #         nearest_true_title_id = title_id_and_dist_and_ec_relative_to_level[k][0]
                #         break
                #     else:
                #         k += 1
                # # never true if using decision_is_correct as the metric:
                # #assert title_id_and_dist_and_ec_original_tuples_and_decision[0][2] != decision_labels[retrieval_id]
                # if title_id_and_dist_and_ec_original_tuples_and_decision[0][2] == decision_labels[retrieval_id]:
                #     num_nearest_wrong_has_correct_decision_label += 1
            for predicted_retrieval_id in predicted_retrieval_id_tuple:
                if predicted_retrieval_id in chosen_title_ids_tuple:
                    retrieval_is_partially_correct = True

        true_top_decision_predictions.append(int(decision_is_correct))
        nearest_true_level_title_ids.append(nearest_true_title_id)
        nearest_wrong_level_title_ids.append(nearest_wrong_title_id)
        nearest_neg2_level_title_ids.append(nearest_neg2_title_id)

        if decision_labels[retrieval_id] != constants.MOREINFO_ID:
            # only calculate retrieval metrics for Supports/Refutes
            true_top_retrieval_predictions_full_evidence.append(int(retrieval_is_correct))
            true_top_retrieval_predictions_full_evidence_partially_correct.append(int(retrieval_is_partially_correct))

        if nearest_wrong_title_id == constants.UNK_TITLE_ID:
            num_nearest_wrong_title_id_not_found += 1
        if nearest_neg2_title_id == constants.UNK_TITLE_ID:
            num_nearest_neg2_title_id_not_found += 1
        if save_eval_output:
            one_claim_output_line_compact = ""
        at_least_1_correct_decision_in_beam = False
        at_least_1_correct_title_in_beam = False
        at_least_1_correct_document_in_beam = False
        k0_at_least_1_correct_document_in_beam = False
        # get the set of true document-level id's
        true_document_set = set()
        for one_true_title_id in claims_to_true_titles_ids[retrieval_id]:
            if one_true_title_id in unique_title_ids_to_document_ids:
                assert one_true_title_id >= 0
                true_document_set.add(unique_title_ids_to_document_ids[one_true_title_id])

        predicted_title_distance = -1.0
        for k in range(top_k_nearest_memories):
            if k < len(title_id_and_dist_and_ec_original_tuples_and_decision):
                title_meta_data = title_id_and_dist_and_ec_original_tuples_and_decision[k]
                predicted_title_id_tuple = title_meta_data[0]
                if k == 0:
                    predicted_title_distance = title_meta_data[1]
                k_decision_is_correct = title_meta_data[2] == decision_labels[retrieval_id]
                if k_decision_is_correct:
                    at_least_1_correct_decision_in_beam = True

                if len(predicted_title_id_tuple) == 1:
                    predicted_title_id = predicted_title_id_tuple[0]
                    #predicted_title_distance = title_meta_data[1]
                    #error_correction_score = title_meta_data[2]
                    predicted_title_string = ' '.join(unique_original_titles[predicted_title_id])
                    if predicted_title_id in chosen_title_ids_tuple:
                        at_least_1_correct_title_in_beam = True
                        predicted_title_string = f"*{predicted_title_string}*"

                    if unique_title_ids_to_document_ids[predicted_title_id] in true_document_set:
                        at_least_1_correct_document_in_beam = True
                        #predicted_title_string = f"@{predicted_title_string}@"
                        if k == 0:
                            k0_at_least_1_correct_document_in_beam = True

                    if save_eval_output:
                        one_claim_output_line_compact += f"\t{k}: D.: {title_meta_data[2]}; D.Match: {k_decision_is_correct}; {predicted_title_string}\n"
                else:
                    first_predicted_title_id = predicted_title_id_tuple[0]
                    second_predicted_title_id = predicted_title_id_tuple[1]
                    # predicted_title_distance = title_meta_data[1]
                    # error_correction_score = title_meta_data[2]
                    first_predicted_title_string = ' '.join(unique_original_titles[first_predicted_title_id])
                    second_predicted_title_string = ' '.join(unique_original_titles[second_predicted_title_id])
                    if first_predicted_title_id in chosen_title_ids_tuple:
                        at_least_1_correct_title_in_beam = True
                        first_predicted_title_string = f"*{first_predicted_title_string}*"
                    if second_predicted_title_id in chosen_title_ids_tuple:
                        at_least_1_correct_title_in_beam = True
                        second_predicted_title_string = f"*{second_predicted_title_string}*"

                    if unique_title_ids_to_document_ids[first_predicted_title_id] in true_document_set or unique_title_ids_to_document_ids[second_predicted_title_id] in true_document_set:
                        at_least_1_correct_document_in_beam = True
                        #predicted_title_string = f"@{predicted_title_string}@"
                        if k == 0:
                            k0_at_least_1_correct_document_in_beam = True

                    if save_eval_output:
                        one_claim_output_line_compact += f"\t{k}: D.: {title_meta_data[2]}; D.Match: {k_decision_is_correct}; 2 Results: {first_predicted_title_string} {second_predicted_title_string}\n"
        predictions_at_least_1_correct_decision_in_beam.append(int(at_least_1_correct_decision_in_beam))
        if decision_labels[retrieval_id] != constants.MOREINFO_ID:
            # only calculate retrieval metrics for Supports/Refutes
            predictions_at_least_1_correct_title_in_beam.append(int(at_least_1_correct_title_in_beam))
            predictions_at_least_1_correct_document_in_beam.append(int(at_least_1_correct_document_in_beam))
            predictions_k0_at_least_1_correct_document_in_beam.append(int(k0_at_least_1_correct_document_in_beam))
        if save_eval_output:
            full_output_line_compact = f"Level {level_id}: Sentence {retrieval_id}; Decision: {predicted_decision_id}; Decision Match: {decision_is_correct}; Exact Retrieval Match: {retrieval_is_correct};" \
                                       f" Partial Retrieval Match in Beam: {at_least_1_correct_title_in_beam}; " \
                                       f"Partial Document-Level Retrieval Match in Beam: {at_least_1_correct_document_in_beam}; k=0 decision distance: {predicted_title_distance}\n" \
                                       f"\tReference: {' '.join(original_titles[retrieval_id])}\n"
            full_output_line_compact += one_claim_output_line_compact
            eval_output_compact.append(full_output_line_compact)

    print(f"\tEval ({mode}) level{level_id}: Number of nearest wrong ids with the correct decision label in the second beam position (and the top of the beam correct): {num_nearest_wrong_has_correct_decision_label}")
    print(f"\tEval ({mode}) level{level_id}: Number of nearest wrong ids not found (unk): {num_nearest_wrong_title_id_not_found}")
    print(
        f"\tEval ({mode}) level{level_id}: Number of nearest neg2 ids not found (unk): {num_nearest_neg2_title_id_not_found}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct decision predictions (regardless of retrieval): "
        f"{np.sum(true_top_decision_predictions)} out of {len(true_top_decision_predictions)}; "
        f"proportion: {np.mean(true_top_decision_predictions)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct decision predictions within top {top_k_nearest_memories} (regardless of retrieval): "
        f"{np.sum(predictions_at_least_1_correct_decision_in_beam)} out of {len(predictions_at_least_1_correct_decision_in_beam)}; "
        f"proportion: {np.mean(predictions_at_least_1_correct_decision_in_beam)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct retrieval predictions (strict: complete evidence match within top 2): "
        f"{np.sum(true_top_retrieval_predictions_full_evidence)} out of {len(true_top_retrieval_predictions_full_evidence)}; "
        f"proportion: {np.mean(true_top_retrieval_predictions_full_evidence)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct retrieval predictions (relaxed: partial match within top 2): "
        f"{np.sum(true_top_retrieval_predictions_full_evidence_partially_correct)} out of {len(true_top_retrieval_predictions_full_evidence_partially_correct)}; "
        f"proportion: {np.mean(true_top_retrieval_predictions_full_evidence_partially_correct)}")


    print(
        f"\tEval ({mode}) level{level_id}: Number of correct retrieval predictions (relaxed: at least 1 partial match within top {top_k_nearest_memories}): "
        f"{np.sum(predictions_at_least_1_correct_title_in_beam)} out of {len(predictions_at_least_1_correct_title_in_beam)}; "
        f"proportion: {np.mean(predictions_at_least_1_correct_title_in_beam)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of retrieval predictions containing at least 1 correct document in retrieval beam (relaxed: at least 1 partial match within top {top_k_nearest_memories}): "
        f"{np.sum(predictions_at_least_1_correct_document_in_beam)} out of {len(predictions_at_least_1_correct_document_in_beam)}; "
        f"proportion: {np.mean(predictions_at_least_1_correct_document_in_beam)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of retrieval predictions with at least 1 correct document appearing at top of retrieval beam: "
        f"{np.sum(predictions_k0_at_least_1_correct_document_in_beam)} out of {len(predictions_k0_at_least_1_correct_document_in_beam)}; "
        f"proportion: {np.mean(predictions_k0_at_least_1_correct_document_in_beam)}")


    predicted_output[f"level{level_id}_{mode}_nearest_true_level_title_ids"] = nearest_true_level_title_ids
    predicted_output[f"level{level_id}_{mode}_nearest_wrong_level_title_ids"] = nearest_wrong_level_title_ids
    predicted_output[f"level{level_id}_{mode}_retrieval_acc"] = np.mean(true_top_retrieval_predictions_full_evidence)
    predicted_output[f"level{level_id}_{mode}_decision_acc"] = np.mean(true_top_decision_predictions)
    predicted_output[f"level{level_id}_{mode}_score_vals"] = []
    predicted_output[f"level{level_id}_{mode}_score_vals_compact"] = eval_output_compact

    predicted_output[f"level{level_id}_{mode}_nearest_neg2_level_title_ids"] = nearest_neg2_level_title_ids

    if data["create_exemplar_database"] or data["create_exemplar_query"] or data["save_exemplar_output"] or data["visualize_alignment"]:
        predicted_output[f"level{level_id}_{mode}_retrieval_id_to_title_id_and_dist_and_ec"] = retrieval_id_to_title_id_and_dist_and_ec

    end_time = time.time()
    print(f"eval_nearest_titles_from_memory time: {(end_time - start_time) / 60} minutes")

    return predicted_output

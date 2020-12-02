from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
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

import copy


def eval_nearest_titles_from_memory_for_level3_ec_format(predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data, params, save_eval_output=False, mode="train", level_id=-1):

    assert level_id == 3
    assert len(predicted_output[f"level{level_id-1}_{mode}_retrieval_distances_at_top_of_beam"]) == \
           len(retrieval_id_to_title_id_and_dist_and_ec)
    # use * to indicate correct title
    start_time = time.time()

    original_claims = data[f"{mode}_original_sentences"]
    unique_original_titles = data[f"{mode}_unique_original_titles"]
    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]
    claims_to_chosen_title_ids = data[f"{mode}_claims_to_chosen_title_ids"]
    claims_to_true_titles_ids = data[f"{mode}_claims_to_true_titles_ids"]
    claims_to_true_title_ids_evidence_sets = data[f"{mode}_claims_to_true_title_ids_evidence_sets"]
    decision_labels = data[f"{mode}_decision_labels"]  # ground-truth decision labels for each claim
    #unique_title_ids_to_decision_labels = data[f"{mode}_unique_title_ids_to_decision_labels"]  # ground-truth decision labels for each unique title


    predicted_original_evidence_ids = data[f"predicted_level{level_id}_{mode}_original_evidence_ids"]
    unique_titles_to_decision_labels = data[f"predicted_level{level_id}_{mode}_unique_titles_to_decision_labels"]
    predicted_original_evidence_ids_marginalized = data[f"predicted_level{level_id}_{mode}_original_evidence_ids_marginalized"]

    # for fever, the following provides a mapping from wiki sentences to wiki documents; this is used for eval purposes
    # (e.g., to check whether the ground truth documents are found)
    unique_title_ids_to_document_ids = data[f"{mode}_unique_title_ids_to_document_ids"]

    assert len(claims_to_chosen_title_ids) == len(retrieval_id_to_title_id_and_dist_and_ec), \
        f"{len(claims_to_chosen_title_ids)}, {len(retrieval_id_to_title_id_and_dist_and_ec)}"
    retrieval_id_check = 0
    for _ in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        assert retrieval_id_check in retrieval_id_to_title_id_and_dist_and_ec
        retrieval_id_check += 1

    eval_output_compact = []

    true_top_decision_predictions = []
    distances = []
    distances_true_predictions = []
    distances_false_predictions = []
    retrieval_distances_true_predictions = []
    retrieval_distances_false_predictions = []
    retrieval_distance_unk_for_nonunk_prediction = 0
    num_unpredicted_lines = 0
    for retrieval_id in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        output_line = []
        decision_is_correct = False
        # in this case, need to convert relative ids back to the original ids
        title_id_and_dist_and_ec_relative_to_level = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
        if len(title_id_and_dist_and_ec_relative_to_level) >= 1:
            predicted_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
            predicted_retrieval_original_evidence_ids = predicted_original_evidence_ids_marginalized[predicted_title_id]
            predicted_title_strings = []
            for predicted_retrieval_original_evidence_id in predicted_retrieval_original_evidence_ids:
                predicted_title_strings.append(' '.join(unique_original_titles[predicted_retrieval_original_evidence_id]))

            predicted_title_distance = title_id_and_dist_and_ec_relative_to_level[0][1]
            distances.append(predicted_title_distance)

            retrieval_distance_at_top_of_beam = \
                predicted_output[f"level{level_id - 1}_{mode}_retrieval_distances_at_top_of_beam"][retrieval_id]
            predicted_decision_id = unique_titles_to_decision_labels[predicted_title_id]
            if predicted_decision_id == decision_labels[retrieval_id]:
                decision_is_correct = True
                distances_true_predictions.append(predicted_title_distance)
                if retrieval_distance_at_top_of_beam != -1.0:
                    retrieval_distances_true_predictions.append(retrieval_distance_at_top_of_beam)
                else:
                    retrieval_distance_unk_for_nonunk_prediction += 1
            else:
                distances_false_predictions.append(predicted_title_distance)
                if retrieval_distance_at_top_of_beam != -1.0:
                    retrieval_distances_false_predictions.append(retrieval_distance_at_top_of_beam)
                else:
                    retrieval_distance_unk_for_nonunk_prediction += 1

            output_line.append(f"Retrieval (k=0) Distance:")
            output_line.append(f"{retrieval_distance_at_top_of_beam}")
            output_line.append(f"Decision Distance:")
            output_line.append(f"{predicted_title_distance}")
            output_line.append(f"Prediction: {decision_is_correct}")
            output_line.append(f"{predicted_decision_id}")

            if predicted_decision_id == constants.SUPPORTS_ID:
                output_line.append(constants.SUPPORTS_STRING)
            elif predicted_decision_id == constants.REFUTES_ID:
                output_line.append(constants.REFUTES_STRING)
            elif predicted_decision_id == constants.MOREINFO_ID:
                output_line.append(constants.MOREINFO_STRING)
            else:
                assert False

            output_line.append(' '.join(original_claims[retrieval_id]))
            output_line.extend(predicted_title_strings)
        else:
            output_line.append(f"Retrieval (k=0) Distance:")
            output_line.append(f"{-1}")
            output_line.append(f"Decision Distance:")
            output_line.append(f"{-1}")
            output_line.append(f"Prediction: {decision_is_correct}")
            output_line.append(f"{constants.UNK_TITLE_ID}")
            output_line.append("")
            output_line.append(' '.join(original_claims[retrieval_id]))
            num_unpredicted_lines += 1

        # output line is: Distance\tDistance val\tPrediction: True/False\tDecision id\tDecision string\tclaim\evidence (if any)
        eval_output_compact.append("\t".join(output_line)+"\n")
        true_top_decision_predictions.append(int(decision_is_correct))

    print(f"Total unpredicted lines: {num_unpredicted_lines} out of {len(true_top_decision_predictions)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct decision predictions (regardless of retrieval): "
        f"{np.sum(true_top_decision_predictions)} out of {len(true_top_decision_predictions)}; "
        f"proportion: {np.mean(true_top_decision_predictions)}")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered predicted: Distance "
            f"mean: {np.mean(distances)}; std: {np.std(distances)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered correctly predicted: Distance "
            f"mean: {np.mean(distances_true_predictions)}; std: {np.std(distances_true_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered wrongly predicted: Distance "
            f"mean: {np.mean(distances_false_predictions)}; std: {np.std(distances_false_predictions)}")
    except:
        print(f"Unexpected distance output")

    print_distance_stats(mode, level_id, retrieval_distance_unk_for_nonunk_prediction,
                         retrieval_distances_true_predictions,
                         retrieval_distances_false_predictions)

    predicted_output[f"level{level_id}_{mode}_nearest_true_level_title_ids"] = -1
    predicted_output[f"level{level_id}_{mode}_nearest_wrong_level_title_ids"] = -1
    predicted_output[f"level{level_id}_{mode}_retrieval_acc"] = -1
    predicted_output[f"level{level_id}_{mode}_decision_acc"] = np.mean(true_top_decision_predictions)
    predicted_output[f"level{level_id}_{mode}_score_vals"] = []
    predicted_output[f"level{level_id}_{mode}_score_vals_compact"] = eval_output_compact

    end_time = time.time()
    print(f"eval_nearest_titles_from_memory_for_level3_ec_format time: {(end_time - start_time) / 60} minutes")

    return predicted_output


def print_distance_stats(mode, level_id, retrieval_distance_unk_for_nonunk_prediction,
                         retrieval_distances_true_predictions,
                         retrieval_distances_false_predictions):

    print(f"Number of retrieval distances that are -1 for non-unk decision predictions: "
          f"{retrieval_distance_unk_for_nonunk_prediction}")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered correctly predicted decisions: Retrieval distances "
            f"mean: {np.mean(retrieval_distances_true_predictions)}; "
            f"std: {np.std(retrieval_distances_true_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered wrongly predicted decisions: Retrieval distances "
            f"mean: {np.mean(retrieval_distances_false_predictions)}; "
            f"std: {np.std(retrieval_distances_false_predictions)}")
    except:
        print(f"Unexpected distance output")




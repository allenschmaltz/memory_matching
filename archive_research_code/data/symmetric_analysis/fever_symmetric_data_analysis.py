# -*- coding: utf-8 -*-
"""
This scripts runs the basic eval based on distances (not exemplar auditing). The distance constraints are also
included in the exemplar auditing eval scripts. We primarily include this for reference, as it may be easier to
follow as an initial point of contrast.

"""

import sys
import argparse

import string
import codecs

from os import path
import random
from collections import defaultdict
import operator

import numpy as np
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer
# note that the wiki and claims files may use different unicode forms; e.g., "Janelle_Mon\u00e1e" vs.
# "Janelle_Mona\u0301e", so it is necessary to normalize.
import unicodedata
from sklearn.utils import shuffle

from scorer import fever_score
import matplotlib.pyplot as plt

SUPPORTS_ID = 0
REFUTES_ID = 1
MOREINFO_ID = 2

random.seed(1776)


def remove_internal_whitespace(line_string):
    return " ".join(line_string.strip().split())


def update_distance_structure(distances, prediction_type, retrieval_distance, decision_distance,
                              decision_label, predicted_label):
    for level_id, model_distance in zip([2, 3], [retrieval_distance, decision_distance]):
        if predicted_label == decision_label:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"].append(model_distance)
        else:
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"].append(model_distance)
    return distances


def read_predicted_ec_file(filepath_with_name, np_random_state, decision_labels, id_strings,
                           level2_constrained_mean, level2_constrained_std,
                           level3_constrained_mean, level3_constrained_std, is_dev):

    predictions = []
    predictions_orig = []  # the original ids -- double check these are always unchanged
    predictions_new = []

    predictions_constrained = []
    predictions_constrained_exchange = []

    predictions_level2_distances_tuples = []
    predictions_level3_distances_tuples = []

    distances = {}
    for prediction_type in ["all", "orig", "new"]:
        for level_id in [2, 3]:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"] = []
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"] = []

    is_constrained = {}
    for claim_type in ["original", "0000002", "0000003", "0000004"]:
        is_constrained[claim_type] = []

    num_flipped_predictions = 0
    number_of_level3_predicted_unk = 0
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}")

            line = line.strip().split("\t")
            retrieval_distance = float(line[1])
            decision_distance = float(line[3])
            assert len(line) >= 8, f"len(line): {len(line)}; line: {line}"
            predicted_label = int(line[5])
            assert predicted_label in [-1, SUPPORTS_ID, REFUTES_ID] #, MOREINFO_ID]
            if predicted_label == -1:
                number_of_level3_predicted_unk += 1
                predicted_label = np_random_state.randint(2)
                print(f"WARNING: Predicted level 3 label for sentence {line_id} was -1. Setting to random (0, 1, or 2):"
                      f" {predicted_label}")

            predictions_level2_distances_tuples.append(
                (int(predicted_label == decision_labels[line_id]), retrieval_distance))
            predictions_level3_distances_tuples.append(
                (int(predicted_label == decision_labels[line_id]), decision_distance))

            id_string = id_strings[line_id]

            predictions.append(int(predicted_label == decision_labels[line_id]))
            if retrieval_distance < (level2_constrained_mean + 0 * level2_constrained_std) and \
                    decision_distance < (level3_constrained_mean + 0 * level3_constrained_std):

                predictions_constrained.append(int(predicted_label == decision_labels[line_id]))
                predictions_constrained_exchange.append(int(predicted_label == decision_labels[line_id]))
                if "0000002" in id_string or "0000003" in id_string or "0000004" in id_string:
                    for claim_type in ["0000002", "0000003", "0000004"]:
                        if claim_type in id_string:
                            is_constrained[claim_type].append(1)
                            break
                else:
                    is_constrained["original"].append(1)
            else:

                if "0000002" in id_string or "0000003" in id_string or "0000004" in id_string:
                    for claim_type in ["0000002", "0000003", "0000004"]:
                        if claim_type in id_string:
                            is_constrained[claim_type].append(0)
                            break
                else:
                    is_constrained["original"].append(0)

                if "0000002" in id_string or "0000004" in id_string:  # datastore has changed
                    num_flipped_predictions += 1
                    if predicted_label == SUPPORTS_ID:
                        flipped_predicted_label = REFUTES_ID
                    elif predicted_label == REFUTES_ID:
                        flipped_predicted_label = SUPPORTS_ID
                    predictions_constrained_exchange.append(int(flipped_predicted_label == decision_labels[line_id]))
                    # temp to update the result below separating orig from new
                    #predicted_label = flipped_predicted_label
                else:
                    predictions_constrained_exchange.append(int(predicted_label == decision_labels[line_id]))

            # if ("0000002" in id_string or "0000004" in id_string) and retrieval_distance < level2_constrained_mean:
            #     print(f"{id_string}, {' '.join(line)}")

            if "0000002" in id_string or "0000004" in id_string:  # new retrieval evidence
                if decision_labels[line_id] in [0, 1]:
                    predictions_new.append(int(predicted_label == decision_labels[line_id]))
                    distances = update_distance_structure(distances, "new", retrieval_distance, decision_distance,
                                                          decision_labels[line_id], predicted_label)
            elif "0000003" in id_string or is_dev:  # original evidence, modified claim

                if decision_labels[line_id] in [0, 1]:
                    predictions_orig.append(int(predicted_label == decision_labels[line_id]))
                    distances = update_distance_structure(distances, "orig", retrieval_distance, decision_distance,
                                                          decision_labels[line_id], predicted_label)

            distances = update_distance_structure(distances, "all", retrieval_distance, decision_distance,
                                                  decision_labels[line_id], predicted_label)

            line_id += 1

    print(f"Number of flipped predictions: {num_flipped_predictions}")
    print(f"Accuracy: {np.mean(predictions)}: {np.sum(predictions)} out of {len(predictions)}")
    print(f"Accuracy (only original): {np.mean(predictions_orig)}: {np.sum(predictions_orig)} out of {len(predictions_orig)}")
    print(f"Accuracy (only new): {np.mean(predictions_new)}: {np.sum(predictions_new)} out of {len(predictions_new)}")

    for claim_type in ["original", "0000002", "0000003", "0000004"]:
        if len(is_constrained[claim_type]) > 0:
            print(f"Percentage of {claim_type} claims retained: {np.sum(is_constrained[claim_type])} "
                  f"out of {len(is_constrained[claim_type])}: "
                  f"{np.sum(is_constrained[claim_type])/len(is_constrained[claim_type])}")

    for prediction_type in ["all", "orig", "new"]:
        for level_id in [2, 3]:
            try:
                print(f"{prediction_type}: Level {level_id} distance (correct decision): mean: "
                      f"{np.mean(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}, "
                      f"std: {np.std(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}, "
                      f"min: {np.min(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}, "
                      f"max: {np.max(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}, "
                      f"total: {len(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}")
            except:
                print(f"A distance list was empty")
            try:
                print(f"{prediction_type}: Level {level_id} distance (wrong decision): mean: "
                      f"{np.mean(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}, "
                      f"std: {np.std(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}, "
                      f"min: {np.min(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}, "
                      f"max: {np.max(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}, "
                      f"total: {len(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}")
            except:
                print(f"A distance list was empty")

    print(f"Accuracy (constrained by distance): {np.mean(predictions_constrained)}: "
          f"{np.sum(predictions_constrained)} out of {len(predictions_constrained)}")
    print(f"Accuracy (constrained by distance, exchanged): {np.mean(predictions_constrained_exchange)}: "
          f"{np.sum(predictions_constrained_exchange)} out of {len(predictions_constrained_exchange)}")

    # # sort predictions by distances (low to high)
    # sorted_predictions_level2_distances_tuples = sorted(predictions_level2_distances_tuples, key=lambda x: (x[1]))
    # sorted_predictions_level3_distances_tuples = sorted(predictions_level3_distances_tuples, key=lambda x: (x[1]))
    # assert len(sorted_predictions_level2_distances_tuples) == len(sorted_predictions_level3_distances_tuples)
    # sorted_predictions = {}
    # # calculate the running accuracy
    # sorted_predictions["level2_sorted_dist"] = [x[1] for x in sorted_predictions_level2_distances_tuples]
    # sorted_predictions["level3_sorted_dist"] = [x[1] for x in sorted_predictions_level3_distances_tuples]
    # sorted_predictions["running_acc"] = []
    # sorted_predictions["running_n"] = []
    # for instance_i in range(len(sorted_predictions_level3_distances_tuples)):
    #     sorted_predictions["running_acc"].append(
    #         np.mean([x[0] for x in sorted_predictions_level3_distances_tuples[0:instance_i + 1]]))
    #     sorted_predictions["running_n"].append(len(sorted_predictions_level3_distances_tuples[0:instance_i + 1]))
    #
    # return distances, sorted_predictions


def get_decision_labels_from_file(decision_labels_file):
    decision_labels = []
    id_strings = []
    with codecs.open(decision_labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")
            assert len(line) == 3
            label = line[1]
            if label == "SUPPORTS":
                decision_labels.append(SUPPORTS_ID)
            elif label == "REFUTES":
                decision_labels.append(REFUTES_ID)
            elif label == "UNVERIFIABLE":
                decision_labels.append(MOREINFO_ID)
            else:
                assert False
            id_string = str(line[0].strip())
            id_strings.append(id_string)
    return decision_labels, id_strings



def save_jsonl_evidence_lines(filename_with_path, evidence_sets):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for evidence_set in evidence_sets:
            json.dump(evidence_set, f)
            f.write('\n')


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_ec_file', type=str, help="input_ec_file")
    parser.add_argument('--control_file', type=str, help="control_file")
    parser.add_argument("--level2_constrained_mean", default=0.0, type=float,
                        help="level2_constrained_mean")
    parser.add_argument("--level2_constrained_std", default=0.0, type=float,
                        help="level2_constrained_std")
    parser.add_argument("--level3_constrained_mean", default=0.0, type=float,
                        help="level3_constrained_mean")
    parser.add_argument("--level3_constrained_std", default=0.0, type=float,
                        help="level3_constrained_std")

    args = parser.parse_args(arguments)

    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)

    is_dev = False

    print(f"Using level 2 mean: {args.level2_constrained_mean}; level 3 mean: {args.level3_constrained_mean}")
    decision_labels, id_strings = get_decision_labels_from_file(args.control_file)
    read_predicted_ec_file(args.input_ec_file, np_random_state, decision_labels,
                                                           id_strings, args.level2_constrained_mean,
                                                           args.level2_constrained_std, args.level3_constrained_mean,
                                                           args.level3_constrained_std, is_dev)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


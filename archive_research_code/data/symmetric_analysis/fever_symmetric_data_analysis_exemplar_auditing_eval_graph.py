# -*- coding: utf-8 -*-
"""
Exemplar auditing analysis

This is used to generate the exemplar auditing results using a single DB, as well as the accuracy vs. n vs.
exemplar-TP distance graph.

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

import pickle
import matplotlib.pyplot as plt

SUPPORTS_ID = 0
REFUTES_ID = 1
MOREINFO_ID = 2

random.seed(1776)


def load_memory_metadata_pickle(memory_dir, file_identifier_prefix):
    path = f"{memory_dir}/{file_identifier_prefix}_memory.pkl"
    try:
        with open(path, 'rb') as input:
            memory_metadata = pickle.load(input)
        return memory_metadata
    except:
        print(f"No available memory metadata at {path}.")
        exit()


def remove_internal_whitespace(line_string):
    return " ".join(line_string.strip().split())


def filter_with_bert_tokenizer(tokenizer, sentence_tokens):
    filtered_tokens = []
    wordpiece_len = 0
    for token in sentence_tokens:
        bert_tokens = tokenizer.tokenize(token)
        if len(bert_tokens) == 0:  # must be a special character filtered by BERT
            assert False, f"ERROR: Tokenizer filtering is not expected to occur with this data."
            #pass
            #print(f"Ignoring {token} with label {label}")
        else:
            filtered_tokens.append(token)
            wordpiece_len += len(bert_tokens)
    return filtered_tokens, wordpiece_len


def read_ground_truth_jsonl(filepath_with_name):
    true_claims_data = []
    original_claim_ids = []
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}.")
            line_id += 1
            line = line.strip()
            data = json.loads(line)
            id = int(data["id"])
            original_claim_ids.append(id)
            #verifiable = data["verifiable"]
            label = data["label"]
            #laim = remove_internal_whitespace(data["claim"])
            evidence = data["evidence"]
            true_claims_data.append({"label": label, "evidence": evidence})
    return true_claims_data, original_claim_ids


def get_document_title_and_sent_index_from_wiki_sentence_string(wiki_title_reformatted_string):
    # We use this for assigning covered sentence id's to the true data since we are no longer loading the raw titles
    # Consider replacing this by preprocessing the true data with the covered indexes; unk titles are always wrong predictions
    end_of_title_index = wiki_title_reformatted_string.find(", sentence ")
    assert end_of_title_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    # Need to find the next colon in order to pull the sentence index
    end_of_sentence_colon_index = wiki_title_reformatted_string[end_of_title_index:].find(":")
    assert end_of_sentence_colon_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    end_of_sentence_colon_index += end_of_title_index

    sent_id = int(wiki_title_reformatted_string[end_of_title_index+len(", sentence "):end_of_sentence_colon_index])
    start_of_title_index = wiki_title_reformatted_string.find("Evidence:")
    assert start_of_title_index == 0, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    return wiki_title_reformatted_string[start_of_title_index+len("Evidence:"):end_of_title_index].strip(), sent_id


def update_distance_structure(distances, prediction_type, retrieval_distance, decision_distance,
                              decision_label, predicted_label):
    for level_id, model_distance in zip([2, 3], [retrieval_distance, decision_distance]):
        if predicted_label == decision_label:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"].append(model_distance)
        else:
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"].append(model_distance)
    return distances


def read_predicted_exemplar_file(filepath_with_name, np_random_state, decision_labels, id_strings,
                                 exemplar_database_memory_dir, database_decision_labels, database_id_strings,
                                 database_lines):

    cross_sequence_id_to_claim_id = load_memory_metadata_pickle(exemplar_database_memory_dir,
                                f"levels123diff_exemplar_{'database'}_cross_sequence_id_to_claim_id")
    archive_decision_labels = load_memory_metadata_pickle(exemplar_database_memory_dir,
                                f"levels123diff_exemplar_{'database'}_decision_labels_by_claim_indexes")
    assert len(archive_decision_labels) == len(database_decision_labels), f"{len(archive_decision_labels)}, " \
                                                                          f"{len(database_decision_labels)}"
    assert archive_decision_labels == database_decision_labels
    assert len(cross_sequence_id_to_claim_id) == len(database_decision_labels)  # *2 (currently, no data augmentation)

    # summary stats from FEVER training
    TRAIN_TP_DIST_LEVEL3_MEAN = 0.919224547006786
    TRAIN_TP_DIST_LEVEL3_STD = 1.8007443781705101

    # alternatively, we could use E(level 2 distance | correct decision)
    TRAIN_TP_DIST_LEVEL2_MEAN = 0.4862351810180722
    TRAIN_TP_DIST_LEVEL2_STD = 4.751204690501908

    tp_threshold = 0.5

    predictions = []
    predictions_orig = []  # the original ids -- double check these are always unchanged
    predictions_new = []

    predictions_exa_constrained_tp_distances_tuples = []  # prediction true/false (0/1), exa distance | exemplar is TP

    predictions_constrained = []
    predictions_constrained_dist_exchange = []

    predictions_exa_constrained_tp = []
    predictions_exa_constrained_tp_threshold = []

    correct_class0_distribution = np.array([0.0, 0.0, 0.0, 0.0, 0])
    correct_class1_distribution = np.array([0.0, 0.0, 0.0, 0.0, 0])

    distances = {}
    for prediction_type in ["all"]: #, "orig", "new"]:
        for level_id in [2, 3]:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"] = []
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"] = []

    distances[f"min_exemplar_dist_correct_decision{'all'}"] = []
    distances[f"min_exemplar_dist_wrong_decision{'all'}"] = []

    num_flipped_predictions = 0
    number_of_level3_predicted_unk = 0
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}")

            line = line.strip().split("\t")
            exa_offset = 0
            class0_distances = []
            class0_distances_distribution = []  # higher value means closer distance (i.e., more prob. mass)
            class0_db_ids = []
            class1_distances = []
            class1_distances_distribution = []
            class1_db_ids = []
            for class_type in [SUPPORTS_ID, REFUTES_ID]:
                for v_type in ["tp", "fn", "fp", "tn"]:
                    exemplar_tuple = line[exa_offset].split(",")
                    assert exemplar_tuple[0] == f"db_class{class_type}"
                    assert exemplar_tuple[1] == v_type
                    exemplar_db_id = int(exemplar_tuple[2])
                    exemplar_db_dist = float(exemplar_tuple[3])
                    if class_type == SUPPORTS_ID:
                        class0_distances.append(exemplar_db_dist)
                        class0_db_ids.append(exemplar_db_id)
                    elif class_type == REFUTES_ID:
                        class1_distances.append(exemplar_db_dist)
                        class1_db_ids.append(exemplar_db_id)
                    exa_offset += 1
            assert exa_offset == 8
            assert class0_db_ids == class1_db_ids[::-1]
            assert class0_distances == class1_distances[::-1]
            #exa_offset = 8
            retrieval_distance = float(line[exa_offset+1])
            decision_distance = float(line[exa_offset+3])
            assert len(line) >= exa_offset+8, f"len(line): {len(line)}; line: {line}"
            predicted_label = int(line[exa_offset+5])
            assert predicted_label in [SUPPORTS_ID, REFUTES_ID], f"predicted_label: {predicted_label}" #, MOREINFO_ID]

            prefix_string = line[exa_offset+6].strip()
            claim = line[exa_offset+7].strip()
            evidence_sentences = line[exa_offset+8:]

            class0_distances_array = np.array(class0_distances)
            class0_distances_distribution = np.exp(-1 * class0_distances_array) / np.sum(
                np.exp(-1 * class0_distances_array))
            class1_distances_array = np.array(class1_distances)
            class1_distances_distribution = np.exp(-1 * class1_distances_array) / np.sum(
                np.exp(-1 * class1_distances_array))
            id_string = id_strings[line_id]

            if predicted_label == SUPPORTS_ID:  # two branches (SUPPORTS_ID, REFUTES_ID)
                cross_sequence_id = class0_db_ids[np.argmin(class0_distances)]
                database_line_index = cross_sequence_id_to_claim_id[cross_sequence_id]
                database_id_string = database_id_strings[database_line_index]

                predicted_min_distance = np.min(class0_distances)
                predicted_min_id = np.argmin(class0_distances)
                if predicted_label == decision_labels[line_id]:
                    distances[f"min_exemplar_dist_correct_decision{'all'}"].append(predicted_min_distance)
                else:
                    distances[f"min_exemplar_dist_wrong_decision{'all'}"].append(predicted_min_distance)

                if predicted_label == decision_labels[line_id]:
                    correct_class0_distribution[0:4] += class0_distances_distribution
                    correct_class0_distribution[4] += 1

                if predicted_min_id == 0:
                    predictions_exa_constrained_tp.append(int(predicted_label == decision_labels[line_id]))
                    predictions_exa_constrained_tp_distances_tuples.append(
                        (int(predicted_label == decision_labels[line_id]), predicted_min_distance))

                if predicted_min_id == 0 and predicted_min_distance < tp_threshold:
                    predictions_exa_constrained_tp_threshold.append(int(predicted_label == decision_labels[line_id]))

                if ("0000002" in id_string or "0000004" in id_string) \
                        and (retrieval_distance >= TRAIN_TP_DIST_LEVEL2_MEAN
                             or decision_distance >= TRAIN_TP_DIST_LEVEL3_MEAN):  # altered Database
                    predictions_constrained_dist_exchange.append(
                        int(REFUTES_ID == decision_labels[line_id]))
                else:
                    predictions_constrained_dist_exchange.append(int(predicted_label == decision_labels[line_id]))

                # uncomment to print output of various subsets
                # if predicted_label != decision_labels[line_id] and \
                #         database_decision_labels[database_line_index] == decision_labels[line_id] and\
                #         np.argmin(class0_distances) == 2:
                #     print(f"-----------------------")
                #     print(f"{database_lines[database_line_index]}")
                #     print(
                #         f"{id_string}: {retrieval_distance},{decision_distance}, "
                #         f"{prefix_string} {claim} {evidence_sentences}")
                #     print(f"-----------------------")
                # if "0000002" in database_id_string or \
                #         "0000003" in database_id_string or "0000004" in database_id_string:
                #     if np.argmin(class0_distances) == 0 and class0_distances_distribution[0] > threshold:
                #         if "0000002" in id_string or "0000003" in id_string or "0000004" in id_string:
                #             if retrieval_distance < (TRAIN_TP_DIST_LEVEL2_MEAN + 0 * TRAIN_TP_DIST_LEVEL2_STD) and \
                #                     decision_distance < (TRAIN_TP_DIST_LEVEL3_MEAN + 0 * TRAIN_TP_DIST_LEVEL3_STD):
                #                 if database_decision_labels[database_line_index] == predicted_label:
                #                     if predicted_label != decision_labels[line_id]:
                #                         print(f"-----------------------")
                #                         print(f"{database_lines[database_line_index]}")
                #                         print(f"{id_string}: {retrieval_distance},{decision_distance}, "
                #                               f"{prefix_string} {claim} {evidence_sentences}")
                #                         print(f"-----------------------")

            elif predicted_label == REFUTES_ID:
                cross_sequence_id = class1_db_ids[np.argmin(class1_distances)]
                database_line_index = cross_sequence_id_to_claim_id[cross_sequence_id]
                database_id_string = database_id_strings[database_line_index]

                predicted_min_distance = np.min(class1_distances)
                predicted_min_id = np.argmin(class1_distances)
                if predicted_label == decision_labels[line_id]:
                    distances[f"min_exemplar_dist_correct_decision{'all'}"].append(predicted_min_distance)
                else:
                    distances[f"min_exemplar_dist_wrong_decision{'all'}"].append(predicted_min_distance)

                if predicted_label == decision_labels[line_id]:
                    correct_class1_distribution[0:4] += class1_distances_distribution
                    correct_class1_distribution[4] += 1

                if predicted_min_id == 0:
                    predictions_exa_constrained_tp.append(int(predicted_label == decision_labels[line_id]))
                    predictions_exa_constrained_tp_distances_tuples.append(
                        (int(predicted_label == decision_labels[line_id]), predicted_min_distance))

                if predicted_min_id == 0 and predicted_min_distance < tp_threshold:
                    predictions_exa_constrained_tp_threshold.append(int(predicted_label == decision_labels[line_id]))

                if ("0000002" in id_string or "0000004" in id_string) \
                        and (retrieval_distance >= TRAIN_TP_DIST_LEVEL2_MEAN
                             or decision_distance >= TRAIN_TP_DIST_LEVEL3_MEAN):  # altered Database

                    predictions_constrained_dist_exchange.append(
                        int(SUPPORTS_ID == decision_labels[line_id]))
                else:
                    predictions_constrained_dist_exchange.append(int(predicted_label == decision_labels[line_id]))

                # uncomment to print output of various subsets
                # if predicted_label != decision_labels[line_id] and \
                #         database_decision_labels[database_line_index] == decision_labels[line_id] and \
                #         np.argmin(class1_distances) == 2:
                #     print(f"-----------------------")
                #     print(f"{database_lines[database_line_index]}")
                #     print(
                #         f"{id_string}: {retrieval_distance},{decision_distance}, "
                #         f"{prefix_string} {claim} {evidence_sentences}")
                #     print(f"-----------------------")
                #
                # if "0000002" in database_id_string or \
                #         "0000003" in database_id_string or "0000004" in database_id_string:
                #     if np.argmin(class1_distances) == 0 and class1_distances_distribution[0] > threshold:
                #         if "0000002" in id_string or "0000003" in id_string or "0000004" in id_string:
                #             if retrieval_distance < (TRAIN_TP_DIST_LEVEL2_MEAN + 0 * TRAIN_TP_DIST_LEVEL2_STD) and \
                #                     decision_distance < (TRAIN_TP_DIST_LEVEL3_MEAN + 0 * TRAIN_TP_DIST_LEVEL3_STD):
                #                 if database_decision_labels[database_line_index] == predicted_label:
                #                     if predicted_label != decision_labels[line_id]:
                #                         print(f"-----------------------")
                #                         print(f"{database_lines[database_line_index]}")
                #                         print(f"{id_string}: {retrieval_distance},{decision_distance}, "
                #                               f"{prefix_string} {claim} {evidence_sentences}")
                #                         print(f"-----------------------")

            id_string = id_strings[line_id]
            if retrieval_distance < (TRAIN_TP_DIST_LEVEL2_MEAN + 0 * TRAIN_TP_DIST_LEVEL2_STD) and \
                decision_distance < (TRAIN_TP_DIST_LEVEL3_MEAN + 0 * TRAIN_TP_DIST_LEVEL3_STD):

                predictions_constrained.append(int(predicted_label == decision_labels[line_id]))

            if "0000002" in id_string or "0000003" in id_string or "0000004" in id_string:
                if decision_labels[line_id] in [0, 1]:
                    predictions_new.append(int(predicted_label == decision_labels[line_id]))

            else:
                if decision_labels[line_id] in [0, 1]:
                    predictions_orig.append(int(predicted_label == decision_labels[line_id]))

            predictions.append(int(predicted_label == decision_labels[line_id]))

            distances = update_distance_structure(distances, "all", retrieval_distance, decision_distance,
                                                  decision_labels[line_id], predicted_label)

            line_id += 1

    #print(correct_class0_distribution[0:4]/correct_class0_distribution[4])
    #print(correct_class1_distribution[0:4]/correct_class1_distribution[4])

    print(
        f"ExA accuracy (only admitted distance exchange): {np.mean(predictions_constrained_dist_exchange)}: "
        f"{np.sum(predictions_constrained_dist_exchange)} out of {len(predictions_constrained_dist_exchange)}")
    print(
        f"ExA accuracy (only admitted tp): {np.mean(predictions_exa_constrained_tp)}: "
        f"{np.sum(predictions_exa_constrained_tp)} out of {len(predictions_exa_constrained_tp)}")

    print(
        f"ExA accuracy (only admitted tp threshold of {tp_threshold}): "
        f"{np.mean(predictions_exa_constrained_tp_threshold)}: "
        f"{np.sum(predictions_exa_constrained_tp_threshold)} out of {len(predictions_exa_constrained_tp_threshold)}")

    print(f"Number of flipped predictions: {num_flipped_predictions}")
    print(f"Accuracy: {np.mean(predictions)}: {np.sum(predictions)} out of {len(predictions)}")
    print(f"Accuracy (only original): {np.mean(predictions_orig)}: {np.sum(predictions_orig)} out of "
          f"{len(predictions_orig)}")
    print(f"Accuracy (only new): {np.mean(predictions_new)}: {np.sum(predictions_new)} out of {len(predictions_new)}")

    for prediction_type in ["all"]: #, "orig", "new"]:
        for level_id in [2, 3]:
            print(f"{prediction_type}: Level {level_id} distance (correct decision): mean: "
                  f"{np.mean(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}, "
                  f"std: {np.std(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])},"
                  f"total: {len(distances[f'level{level_id}_dist_correct_decision{prediction_type}'])}")
            print(f"{prediction_type}: Level {level_id} distance (wrong decision): mean: "
                  f"{np.mean(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}, "
                  f"std: {np.std(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])},"
                  f"total: {len(distances[f'level{level_id}_dist_wrong_decision{prediction_type}'])}")

    for prediction_status, exemplar_distances in zip(["correct", "wrong"],
                                                     [distances[f"min_exemplar_dist_correct_decision{'all'}"],
                                                      distances[f"min_exemplar_dist_wrong_decision{'all'}"]]):
        print(f"Exemplar distance for {prediction_status} decisions: "
            f"mean: {np.mean(exemplar_distances)}, "
            f"std: {np.std(exemplar_distances)}, "
            f"min: {np.min(exemplar_distances)}, "
            f"max: {np.max(exemplar_distances)}, "
            f"total: {len(exemplar_distances)}")

    print(f"Accuracy (only admitted distance): {np.mean(predictions_constrained)}: "
          f"{np.sum(predictions_constrained)} out of {len(predictions_constrained)}")

    # sort Exemplar-TP-constrained predictions by exemplar distances (low to high)
    sorted_predictions_exa_constrained_tp_distances_tuples = sorted(predictions_exa_constrained_tp_distances_tuples,
                                                                    key=lambda x: (x[1]))
    sorted_predictions = {}
    # calculate the running accuracy
    sorted_predictions["exa_constrained_tp_sorted_dist"] = \
        [x[1] for x in sorted_predictions_exa_constrained_tp_distances_tuples]
    sorted_predictions["running_acc"] = []
    sorted_predictions["running_n"] = []
    for instance_i in range(len(sorted_predictions_exa_constrained_tp_distances_tuples)):
        sorted_predictions["running_acc"].append(
            np.mean([x[0] for x in sorted_predictions_exa_constrained_tp_distances_tuples[0:instance_i + 1]]))
        sorted_predictions["running_n"].append(len(sorted_predictions_exa_constrained_tp_distances_tuples[0:instance_i + 1]))
    return sorted_predictions


def rescale_vals(x):
    #return np.log(x)
    return x


def generate_dual_line_graph(output_graph_file, sorted_predictions):
    # adapted from https://scipy-cookbook.readthedocs.io/items/Matplotlib_LaTeX_Examples.html
    fig_width_pt = 219.1 #246.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              #'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'font.family': 'serif'}
    plt.rcParams.update(params)

    fig, ax1 = plt.subplots()

    accuracy_color = 'tab:blue'
    n_color = 'tab:red'
    acc_line, = ax1.plot(rescale_vals(sorted_predictions["exa_constrained_tp_sorted_dist"]),
                         sorted_predictions["running_acc"], color=accuracy_color, label="Admitted Acc.")
    ax1.set_xlabel('Exemplar TP distance threshold')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Accuracy', color=accuracy_color)
    ax1.tick_params('y', colors=accuracy_color)
    ax1.set_ylim([0, 1])
    ax2 = ax1.twinx()

    n_line, = ax2.plot(rescale_vals(sorted_predictions["exa_constrained_tp_sorted_dist"]),
                       sorted_predictions["running_n"], color=n_color, linestyle='dashed', label="Admitted n")
    ax2.set_ylabel('n', color=n_color)
    ax2.tick_params('y', colors=n_color)
    ax2.set_ylim([0, 650])

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    fig.tight_layout()

    plt.legend((acc_line, n_line), ("Admitted Acc.", "Admitted n"), frameon=False, loc="lower right")
    plt.savefig(output_graph_file)
    print(f"Saved figure to: {output_graph_file}")

    print(f"ExA-TP Distance: min value: {sorted_predictions['exa_constrained_tp_sorted_dist'][0]};"
          f" min value (log): {rescale_vals(sorted_predictions['exa_constrained_tp_sorted_dist'][0])}")
    print(f"ExA-TP Distance: max value: {sorted_predictions['exa_constrained_tp_sorted_dist'][-1]};"
          f" max value (log): {rescale_vals(sorted_predictions['exa_constrained_tp_sorted_dist'][-1])}")

    print(f"Note that we have manually adjusted some of the axis ranges. Adjust accordingly for another dataset.")


def read_predicted_ec_file(filepath_with_name, np_random_state, decision_labels, id_strings):

    database_lines = []
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
            assert predicted_label in [-1, SUPPORTS_ID, REFUTES_ID]  # , MOREINFO_ID]
            if predicted_label == -1:
                number_of_level3_predicted_unk += 1
                predicted_label = np_random_state.randint(2)
                print(
                    f"WARNING: Predicted level 3 label for sentence {line_id} was -1. Setting to random (0, 1, or 2): "
                    f"{predicted_label}")
            database_lines.append(f"{id_strings[line_id]}: {','.join(line)}")
            line_id += 1
    return database_lines


def read_predicted_jsonl(filepath_with_name):
    true_claims_data = []
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}.")
            line_id += 1
            line = line.strip()
            data = json.loads(line)
            id = int(data["id"])
            verifiable = data["verifiable"]
            label = data["label"]
            #claim = remove_internal_whitespace(data["claim"])
            evidence = data["evidence"]

            page_line_list = []
            #if verifiable == "VERIFIABLE":
            for evidence_set in evidence:
                for sent_meta_data in evidence_set:
                    wiki_url = sent_meta_data[2]
                    sent_id = sent_meta_data[3]
                    if wiki_url is not None:
                        page_line_list.append([wiki_url, sent_id])
                    else:
                        page_line_list.append(["asdfasdfasd123900", 0])
            true_claims_data.append({"predicted_label": label, "predicted_evidence": page_line_list})
    return true_claims_data


def get_filtered_title_to_title_hash(filepath_with_name):
    filtered_title_to_title_hash = {}
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 500000 == 0:
                print(f"\tTitle hashes: Currently processing line {line_id}.")
            line_id += 1
            line = line.strip().split("\t")
            filtered_title = line[0].strip()
            title_hash = line[1].strip()
            assert filtered_title not in filtered_title_to_title_hash, f"ERROR: This version assumes no title clashes."
            filtered_title_to_title_hash[filtered_title] = title_hash
    return filtered_title_to_title_hash


def get_claim_ids_from_control_file(filepath_with_name):
    claim_ids = []
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 500000 == 0:
                print(f"\tTitle hashes: Currently processing line {line_id}.")
            line_id += 1
            line = line.strip().split(",")
            claim_id = int(line[0].strip())
            claim_ids.append(claim_id)
    return claim_ids


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
    parser.add_argument('--input_database_ec_file', type=str, help="input_database_ec_file")
    parser.add_argument('--input_exa_file', type=str, help="input_exa_file")
    parser.add_argument("--exemplar_database_memory_dir", default="", help="exemplar_database_memory_dir")
    parser.add_argument('--database_control_file', type=str, help="database_control_file")
    parser.add_argument('--query_control_file', type=str, help="query_control_file")
    parser.add_argument('--output_graph_file', type=str, help="output_graph_file")

    args = parser.parse_args(arguments)

    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)

    database_decision_labels, database_id_strings = get_decision_labels_from_file(args.database_control_file)
    database_lines = read_predicted_ec_file(args.input_database_ec_file, np_random_state, database_decision_labels,
                                            database_id_strings)
    decision_labels, id_strings = get_decision_labels_from_file(args.query_control_file)
    sorted_predictions = read_predicted_exemplar_file(args.input_exa_file, np_random_state, decision_labels,
                                                      id_strings, args.exemplar_database_memory_dir,
                                                      database_decision_labels, database_id_strings, database_lines)

    generate_dual_line_graph(args.output_graph_file, sorted_predictions)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


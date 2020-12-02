# -*- coding: utf-8 -*-
"""
Exemplar auditing analysis -- Here, we allow the database to consist of two separate files. We simply combine the two,
taking the min applicable distances, so it is as if there were only one database.

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


def read_predicted_exemplar_file_conditional_on_db(filepath_with_name):

    exa_data_conditional_on_db_by_claim = []

    distances = {}
    for prediction_type in ["all"]:
        for level_id in [2, 3]:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"] = []
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"] = []

    distances[f"min_exemplar_dist_correct_decision{'all'}"] = []
    distances[f"min_exemplar_dist_wrong_decision{'all'}"] = []

    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}")

            line = line.strip().split("\t")
            exa_data_dict = {}
            exa_offset = 0
            class0_distances = []
            class0_db_ids = []
            class1_distances = []
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

            exa_data_dict["class0_distances"] = class0_distances
            exa_data_dict["class0_db_ids"] = class0_db_ids

            exa_data_dict["class1_distances"] = class1_distances
            exa_data_dict["class1_db_ids"] = class1_db_ids
            # some of the subsequent fields are duplicated across the exa_predicted files; we include them here as a
            # check to ensure that there isn't a mis-match across the files (i.e., checking that correct query file
            # is provided)
            exa_data_dict["retrieval_distance"] = retrieval_distance
            exa_data_dict["decision_distance"] = decision_distance
            exa_data_dict["predicted_label"] = predicted_label
            exa_data_dict["prefix_string"] = prefix_string
            exa_data_dict["claim"] = claim
            exa_data_dict["evidence_sentences"] = evidence_sentences

            exa_data_conditional_on_db_by_claim.append(exa_data_dict)
            line_id += 1

    return exa_data_conditional_on_db_by_claim


def get_exemplar_metadata(db1, db2, cross_sequence_id_to_claim_id1, cross_sequence_id_to_claim_id2,
                          database_decision_labels1, database_id_strings1, database_lines1,
                          database_decision_labels2, database_id_strings2, database_lines2,
                          predicted_label):
    # merge two databases, returning the distance, id, and database instance id for the predicted class
    # (Note that the exemplars are symmetric across the classes (flip tp with tn, for example, but separating
    # avoids this confusion)
    if predicted_label == SUPPORTS_ID:
        selected_class_distances_key = "class0_distances"
        selected_class_db_key = "class0_db_ids"
    elif predicted_label == REFUTES_ID:
        selected_class_distances_key = "class1_distances"
        selected_class_db_key = "class1_db_ids"
    else:
        assert False

    if np.min(db1[selected_class_distances_key]) < np.min(db2[selected_class_distances_key]):
        is_db1 = True
        selected_db = db1
        selected_cross_sequence_id_to_claim_id = cross_sequence_id_to_claim_id1
        selected_database_id_strings = database_id_strings1
        selected_database_decision_labels = database_decision_labels1
        selected_database_lines = database_lines1
    else:
        is_db1 = False
        selected_db = db2
        selected_cross_sequence_id_to_claim_id = cross_sequence_id_to_claim_id2
        selected_database_id_strings = database_id_strings2
        selected_database_decision_labels = database_decision_labels2
        selected_database_lines = database_lines2
    predicted_min_distance = np.min(selected_db[selected_class_distances_key])
    predicted_min_id = np.argmin(selected_db[selected_class_distances_key])

    cross_sequence_id = selected_db[selected_class_db_key][predicted_min_id]
    database_line_index = selected_cross_sequence_id_to_claim_id[cross_sequence_id]
    database_id_string = selected_database_id_strings[database_line_index]
    database_decision_label = selected_database_decision_labels[database_line_index]  # gold
    database_line = selected_database_lines[database_line_index]
    return is_db1, predicted_min_distance, predicted_min_id, \
           database_line_index, database_id_string, database_decision_label, database_line


def eval_by_exemplar(exa_data_conditional_on_db1_by_claim, exa_data_conditional_on_db2_by_claim,
                                 np_random_state,
                                 database_decision_labels1, database_id_strings1, database_lines1,
                                 database_decision_labels2, database_id_strings2, database_lines2,
                                 query_decision_labels, query_id_strings,
                                 exemplar_database_memory_dir1,
                                 exemplar_database_memory_dir2):

    cross_sequence_id_to_claim_id1 = load_memory_metadata_pickle(exemplar_database_memory_dir1,
                                f"levels123diff_exemplar_{'database'}_cross_sequence_id_to_claim_id")
    archive_decision_labels1 = load_memory_metadata_pickle(exemplar_database_memory_dir1,
                                f"levels123diff_exemplar_{'database'}_decision_labels_by_claim_indexes")
    assert len(archive_decision_labels1) == len(database_decision_labels1), f"{len(archive_decision_labels1)}, " \
                                                                          f"{len(database_decision_labels1)}"
    assert archive_decision_labels1 == database_decision_labels1
    assert len(cross_sequence_id_to_claim_id1) == len(database_decision_labels1)  #*2

    cross_sequence_id_to_claim_id2 = load_memory_metadata_pickle(exemplar_database_memory_dir2,
                                f"levels123diff_exemplar_{'database'}_cross_sequence_id_to_claim_id")
    archive_decision_labels2 = load_memory_metadata_pickle(exemplar_database_memory_dir2,
                                f"levels123diff_exemplar_{'database'}_decision_labels_by_claim_indexes")
    assert len(archive_decision_labels2) == len(database_decision_labels2), f"{len(archive_decision_labels2)}, " \
                                                                          f"{len(database_decision_labels2)}"
    assert archive_decision_labels2 == database_decision_labels2
    assert len(cross_sequence_id_to_claim_id2) == len(database_decision_labels2)  #*2

    assert len(exa_data_conditional_on_db1_by_claim) == len(exa_data_conditional_on_db2_by_claim)
    total_claims = len(exa_data_conditional_on_db1_by_claim)

    TRAIN_TP_DIST_LEVEL3_MEAN = 0.919224547006786
    TRAIN_TP_DIST_LEVEL3_STD = 1.8007443781705101

    # alternatively, we could use E(level 2 distance | correct decision)
    TRAIN_TP_DIST_LEVEL2_MEAN = 0.4862351810180722
    TRAIN_TP_DIST_LEVEL2_STD = 4.751204690501908

    predictions = []

    predictions_new = []  # If the datastore is changed, we use the prediction from the nearest exemplar in the updated exemplar database.

    predictions_dist_constrained = []  # only admit if less than level 2 and level 3 mean from training
    predictions_dist_exchange = []  # reference only: if not (<level 2 mean and <level 3 mean) AND new datastore, flip label

    predictions_exa_constrained_tp = []  # only admit if nearest exemplar is TP

    distances = {}
    for prediction_type in ["all"]: #, "orig", "new"]:
        for level_id in [2, 3]:
            distances[f"level{level_id}_dist_correct_decision{prediction_type}"] = []
            distances[f"level{level_id}_dist_wrong_decision{prediction_type}"] = []

    distances[f"min_exemplar_dist_correct_decision_{'all'}"] = []
    distances[f"min_exemplar_dist_wrong_decision_{'all'}"] = []

    distances[f"min_exemplar_dist_correct_decision_{'new'}"] = []
    distances[f"min_exemplar_dist_wrong_decision_{'new'}"] = []

    num_flipped_predictions = 0
    number_of_level3_predicted_unk = 0
    general_counter = 0
    for line_id in range(total_claims):

        exa_data_dict_by_db = {}
        exa_data_dict_by_db["db1"] = exa_data_conditional_on_db1_by_claim[line_id]
        exa_data_dict_by_db["db2"] = exa_data_conditional_on_db2_by_claim[line_id]

        assert exa_data_dict_by_db["db1"]["retrieval_distance"] == exa_data_dict_by_db["db2"]["retrieval_distance"]
        assert exa_data_dict_by_db["db1"]["decision_distance"] == exa_data_dict_by_db["db2"]["decision_distance"]
        assert exa_data_dict_by_db["db1"]["predicted_label"] == exa_data_dict_by_db["db2"]["predicted_label"]
        assert exa_data_dict_by_db["db1"]["prefix_string"] == exa_data_dict_by_db["db2"]["prefix_string"]
        assert exa_data_dict_by_db["db1"]["claim"] == exa_data_dict_by_db["db2"]["claim"]
        assert exa_data_dict_by_db["db1"]["evidence_sentences"] == exa_data_dict_by_db["db2"]["evidence_sentences"]

        retrieval_distance = exa_data_dict_by_db["db1"]["retrieval_distance"]
        decision_distance = exa_data_dict_by_db["db1"]["decision_distance"]
        predicted_label = exa_data_dict_by_db["db1"]["predicted_label"]
        prefix_string = exa_data_dict_by_db["db1"]["prefix_string"]
        claim = exa_data_dict_by_db["db1"]["claim"]
        evidence_sentences = exa_data_dict_by_db["db1"]["evidence_sentences"]


        is_db1, predicted_min_distance, predicted_min_id, \
        database_line_index, database_id_string, database_decision_label, database_line = \
            get_exemplar_metadata(exa_data_dict_by_db["db1"], exa_data_dict_by_db["db2"],
                                  cross_sequence_id_to_claim_id1, cross_sequence_id_to_claim_id2,
                                  database_decision_labels1, database_id_strings1, database_lines1,
                                  database_decision_labels2, database_id_strings2, database_lines2,
                                  predicted_label)

        query_decision_label = query_decision_labels[line_id]  # gold labels
        query_id_string = query_id_strings[line_id]

        if predicted_min_id == 0:  # this is without accounting for changes in datastores
            predictions_exa_constrained_tp.append(int(predicted_label == query_decision_label))

        predictions.append(int(predicted_label == query_decision_label))

        if retrieval_distance < (TRAIN_TP_DIST_LEVEL2_MEAN + 0 * TRAIN_TP_DIST_LEVEL2_STD) and \
                decision_distance < (TRAIN_TP_DIST_LEVEL3_MEAN + 0 * TRAIN_TP_DIST_LEVEL3_STD):
            predictions_dist_exchange.append(int(predicted_label == query_decision_label))
            predictions_dist_constrained.append(int(predicted_label == query_decision_label))

        else:
            if "0000002" in query_id_string or "0000004" in query_id_string:  # altered Database
                num_flipped_predictions += int(is_db1)

                if predicted_label == SUPPORTS_ID:
                    predictions_dist_exchange.append(int(REFUTES_ID == query_decision_label))
                else:
                    predictions_dist_exchange.append(int(SUPPORTS_ID == query_decision_label))
            else:
                predictions_dist_exchange.append(int(predicted_label == query_decision_label))

        if predicted_label == query_decision_label:
            distances[f"min_exemplar_dist_correct_decision_{'all'}"].append(predicted_min_distance)
        else:
            distances[f"min_exemplar_dist_wrong_decision_{'all'}"].append(predicted_min_distance)

        # In the following cases, we take special action if the datastore has changed. This occurs with the id
        # strings that end in "0000002" and "0000004". By contrast, the id strings that end in "0000003" have
        # claims that are modified to flip the label, but the datastore contains the original, unchanged Wikipedia
        # sentence. We cannot, in general, take special action in those cases.
        if "0000002" in query_id_string or "0000004" in query_id_string:  # ids with altered datastores
            # in the case of altered database, we access the DB for altered instances; this is analogous to updating
            # the exemplar database for these instances
            _, predicted_min_distance, predicted_min_id, \
            database_line_index, database_id_string, database_decision_label, database_line = \
                get_exemplar_metadata(exa_data_dict_by_db["db2"], exa_data_dict_by_db["db2"],
                                      cross_sequence_id_to_claim_id2, cross_sequence_id_to_claim_id2,
                                      database_decision_labels2, database_id_strings2, database_lines2,
                                      database_decision_labels2, database_id_strings2, database_lines2,
                                      predicted_label)
            # In this branch, we then discard the model prediction and replace it with the ground-truth label
            # associated with the nearest exemplar. This is valid, as the exemplar is NOT from the test data.
            predictions_new.append(int(database_decision_label == query_decision_label))

            if database_decision_label == query_decision_label:
                distances[f"min_exemplar_dist_correct_decision_{'new'}"].append(predicted_min_distance)
            else:
                distances[f"min_exemplar_dist_wrong_decision_{'new'}"].append(predicted_min_distance)

        else:
            # For instances with unaltered datastores, we use the model prediction, as normal.
            predictions_new.append(int(predicted_label == query_decision_label))

        # The following is only used for printing some low-distance exemplars (for instances with unchanged
        # datastore) for the paper
        if predicted_label == query_decision_label:
            # only considering the train db
            _, predicted_min_distance, predicted_min_id, \
            database_line_index, database_id_string, database_decision_label, database_line = \
                get_exemplar_metadata(exa_data_dict_by_db["db1"], exa_data_dict_by_db["db1"],
                                      cross_sequence_id_to_claim_id1, cross_sequence_id_to_claim_id1,
                                      database_decision_labels1, database_id_strings1, database_lines1,
                                      database_decision_labels1, database_id_strings1, database_lines1,
                                      predicted_label)
            if "0000002" in query_id_string or "0000003" in query_id_string or "0000004" in query_id_string:
                pass
            else:  # printing examples for the paper of in-domain claims
                if predicted_min_distance < 0.25:
                    print(
                        f"Query {query_id_string}: {predicted_label == query_decision_label}: {prefix_string}, "
                        f"{claim}; {' '.join(evidence_sentences)}")
                    print(f"\tQuery retrieval_distance: {retrieval_distance}; Query decision_distance: "
                          f"{decision_distance}")
                    print(f"\tExemplar distance: {predicted_min_distance}; is DB is TP: {predicted_min_id == 0}")
                    print(f"\tDB: {database_line}")
                    print(f"***************************************************")
                    general_counter += 1

    print(f"Number of flipped predictions: {num_flipped_predictions}")
    print(f"Gen counter: {general_counter}")
    print(f"Accuracy: {np.mean(predictions)}: {np.sum(predictions)} out of {len(predictions)}")
    print(
        f"ExA accuracy (distance exchange): {np.mean(predictions_dist_exchange)}: "
        f"{np.sum(predictions_dist_exchange)} out of {len(predictions_dist_exchange)}")
    print(
        f"ExA accuracy (distance constrained): {np.mean(predictions_dist_constrained)}: "
        f"{np.sum(predictions_dist_constrained)} out of {len(predictions_dist_constrained)}")

    print(f"Accuracy (for instances with updated datastore, we discard the model prediction and use the "
          f"label associated with the nearest exemplar in the updated exemplar database): {np.mean(predictions_new)}: "
          f"{np.sum(predictions_new)} out of {len(predictions_new)}")
    print(
        f"ExA accuracy (only admitted tp): {np.mean(predictions_exa_constrained_tp)}: "
        f"{np.sum(predictions_exa_constrained_tp)} out of {len(predictions_exa_constrained_tp)}")

    print(f"Exemplar distances (correct): mean: {np.mean(distances['min_exemplar_dist_correct_decision_all'])};"
          f" min: {np.min(distances['min_exemplar_dist_correct_decision_all'])}; "
          f"max: {np.max(distances['min_exemplar_dist_correct_decision_all'])}; "
          f"out of {len(distances['min_exemplar_dist_correct_decision_all'])}")
    print(f"Exemplar distances (wrong): mean: {np.mean(distances['min_exemplar_dist_wrong_decision_all'])};"
          f" min: {np.min(distances['min_exemplar_dist_wrong_decision_all'])}; "
          f"max: {np.max(distances['min_exemplar_dist_wrong_decision_all'])}; "
          f"out of {len(distances['min_exemplar_dist_wrong_decision_all'])}")

    print(f"Exemplar distances (correct DB prediction), only new datastore claims: "
          f"mean: {np.mean(distances['min_exemplar_dist_correct_decision_new'])};"
          f" min: {np.min(distances['min_exemplar_dist_correct_decision_new'])}; "
          f"max: {np.max(distances['min_exemplar_dist_correct_decision_new'])}; "
          f"out of {len(distances['min_exemplar_dist_correct_decision_new'])}")
    print(f"Exemplar distances (wrong DB prediction), only new datastore claims: "
          f"mean: {np.mean(distances['min_exemplar_dist_wrong_decision_new'])};"
          f" min: {np.min(distances['min_exemplar_dist_wrong_decision_new'])}; "
          f"max: {np.max(distances['min_exemplar_dist_wrong_decision_new'])}; "
          f"out of {len(distances['min_exemplar_dist_wrong_decision_new'])}")


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
            assert predicted_label in [-1, SUPPORTS_ID, REFUTES_ID], f"ERROR: Here, we only consider 2-class."  # , MOREINFO_ID]
            if predicted_label == -1:
                number_of_level3_predicted_unk += 1
                predicted_label = np_random_state.randint(2)
                print(
                    f"WARNING: Predicted level 3 label for sentence {line_id} was -1. Setting to random (0, 1, or 2): {predicted_label}")
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
    parser.add_argument('--input_database_ec_file1', type=str, help="input_database_ec_file1")
    parser.add_argument('--input_exa_file1', type=str, help="input_exa_file1")
    parser.add_argument("--exemplar_database_memory_dir1", default="", help="exemplar_database_memory_dir1")
    parser.add_argument('--database_control_file1', type=str, help="database_control_file1")

    parser.add_argument('--input_database_ec_file2', type=str, help="input_database_ec_file2")
    parser.add_argument('--input_exa_file2', type=str, help="input_exa_file2")
    parser.add_argument("--exemplar_database_memory_dir2", default="", help="exemplar_database_memory_dir2")
    parser.add_argument('--database_control_file2', type=str, help="database_control_file2")

    parser.add_argument('--query_control_file', type=str, help="query_control_file")

    args = parser.parse_args(arguments)

    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)

    database_decision_labels1, database_id_strings1 = get_decision_labels_from_file(args.database_control_file1)
    database_decision_labels2, database_id_strings2 = get_decision_labels_from_file(args.database_control_file2)

    database_lines1 = read_predicted_ec_file(args.input_database_ec_file1, np_random_state, database_decision_labels1,
                                             database_id_strings1)
    database_lines2 = read_predicted_ec_file(args.input_database_ec_file2, np_random_state, database_decision_labels2,
                                             database_id_strings2)
    query_decision_labels, query_id_strings = get_decision_labels_from_file(args.query_control_file)

    exa_data_conditional_on_db1_by_claim = read_predicted_exemplar_file_conditional_on_db(args.input_exa_file1)
    exa_data_conditional_on_db2_by_claim = read_predicted_exemplar_file_conditional_on_db(args.input_exa_file2)

    eval_by_exemplar(exa_data_conditional_on_db1_by_claim, exa_data_conditional_on_db2_by_claim,
                     np_random_state,
                     database_decision_labels1, database_id_strings1, database_lines1,
                     database_decision_labels2, database_id_strings2, database_lines2,
                     query_decision_labels, query_id_strings,
                     args.exemplar_database_memory_dir1,
                     args.exemplar_database_memory_dir2)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


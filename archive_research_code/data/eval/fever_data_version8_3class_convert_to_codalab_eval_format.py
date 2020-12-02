# -*- coding: utf-8 -*-
"""
This scripts converts the output (in ec_format) for eval to submit to codalab.

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

import unicodedata
from sklearn.utils import shuffle

from scorer import fever_score

SUPPORTS_ID = 0
REFUTES_ID = 1
MOREINFO_ID = 2

random.seed(1776)


def read_blind_test_jsonl(filepath_with_name):
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
    return original_claim_ids


def get_document_title_and_sent_index_from_wiki_sentence_string(wiki_title_reformatted_string):
    # We recover the Wikipedia title and the sentence id from the evidence string

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


def read_predicted_ec_file(filepath_with_name, filtered_title_to_title_hash, np_random_state, predicted_claim_ids):
    predicted_claim_ids_to_claims_data = {}
    number_of_level3_predicted_unk = 0
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently processing line {line_id}")

            line = line.strip().split("\t")
            assert len(line) >= 8, f"len(line): {len(line)}; line: {line}"
            predicted_label = int(line[5])
            assert predicted_label in [-1, SUPPORTS_ID, REFUTES_ID, MOREINFO_ID]
            if predicted_label == -1:
                number_of_level3_predicted_unk += 1
                predicted_label = np_random_state.randint(3)
                print(f"WARNING: Predicted level 3 label for sentence {line_id} was -1. Setting to random (0, 1, or 2):"
                      f" {predicted_label}")

            if predicted_label == 0:
                label = "SUPPORTS"
            elif predicted_label == 1:
                label = "REFUTES"
            elif predicted_label == 2:
                label = "NOT ENOUGH INFO"
            # prefix_string = line[6].strip()
            # claim = line[7].strip()
            evidence_sentences = line[8:]
            page_line_list = []
            for evidence_sentence in evidence_sentences:
                evidence_sentence = evidence_sentence.strip()
                if len(evidence_sentence) > 0:
                    wiki_title_reformatted_string, sent_id = \
                        get_document_title_and_sent_index_from_wiki_sentence_string(evidence_sentence)
                    # print(evidence_sentence)
                    # print(f"\t{wiki_title_reformatted_string}")
                    # print(f"\t{sent_id}")
                    assert wiki_title_reformatted_string in filtered_title_to_title_hash
                    page_line_list.append([filtered_title_to_title_hash[wiki_title_reformatted_string], sent_id])
            assert predicted_claim_ids[line_id] not in predicted_claim_ids_to_claims_data
            predicted_claim_ids_to_claims_data[predicted_claim_ids[line_id]] = {"id": predicted_claim_ids[line_id],
                                                                                "predicted_label": label,
                                                                                "predicted_evidence": page_line_list}
            line_id += 1

    print(f"Number of original level 3 predictions that were unk: {number_of_level3_predicted_unk}")
    return predicted_claim_ids_to_claims_data


def get_resorted_predicted_claims(predicted_claim_ids_to_claims_data, original_claim_ids):
    predicted_claims_data = []
    for claim_id in original_claim_ids:
        assert claim_id in predicted_claim_ids_to_claims_data
        predicted_claims_data.append(predicted_claim_ids_to_claims_data[claim_id])
    return predicted_claims_data


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


def save_jsonl_lines(filename_with_path, list_of_dictionaries_to_save):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for dictionary_to_save in list_of_dictionaries_to_save:
            json.dump(dictionary_to_save, f)
            f.write('\n')


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_jsonl_file', type=str, help="The FEVER .jsonl file. This is used to get the order"
                                                             "of the claims. The input to coda lab must be in this "
                                                             "order.")
    parser.add_argument('--input_ec_file', type=str, help="input_ec_file")
    parser.add_argument('--control_file', type=str, help="control_file")
    parser.add_argument('--filtered_titles_to_title_hashes_file', type=str, help="filtered_titles_to_title_hashes_file")
    parser.add_argument('--output_codalab_jsonl_dir', type=str, help="Output dir for coda lab. Filename is saved as "
                                                                      "predictions.jsonl")
    parser.add_argument('--output_codalab_jsonl_filename', default="predictions.jsonl", type=str,
                        help="Filename must be called 'predictions.jsonl' (the default) for upload to CodaLab.")

    args = parser.parse_args(arguments)

    original_claim_ids = read_blind_test_jsonl(args.input_jsonl_file)
    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)
    # Why would any test claims not have predictions? If the claim is not associated with any covered titles.
    # In these cases, we just randomly predict. These are very rare in practice, to the degree that the metrics are
    # not affected either way. For datasets on which this is common, it would be preferable to at least
    # make a claim-only prediction.
    filtered_title_to_title_hash = get_filtered_title_to_title_hash(args.filtered_titles_to_title_hashes_file)
    # Our original pre-processing grouped claims by ground-truth evidence titles to aid in debugging. For blind
    # test, we use the same pre-processing for consistency. In this case there are no known titles, but
    # it is possible the instances could be in an order that differs from the original file.
    # Here, we just need to get the claim id's to map to the order of original_claim_ids.
    predicted_claim_ids = get_claim_ids_from_control_file(args.control_file)
    predicted_claim_ids_to_claims_data = read_predicted_ec_file(args.input_ec_file, filtered_title_to_title_hash,
                                                                np_random_state, predicted_claim_ids)

    predicted_claims_data = get_resorted_predicted_claims(predicted_claim_ids_to_claims_data, original_claim_ids)
    save_jsonl_lines(path.join(args.output_codalab_jsonl_dir, args.output_codalab_jsonl_filename),
                     predicted_claims_data)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


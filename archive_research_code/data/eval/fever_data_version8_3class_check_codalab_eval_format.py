# -*- coding: utf-8 -*-
"""
This scripts checks the codalab format. This can be used with the dev set.

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
            true_claims_data.append(data)

    return true_claims_data


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_jsonl_file', type=str, help="input_jsonl_file")
    parser.add_argument('--input_predictions_jsonl_file', type=str, help="input_predictions_jsonl_file")
    args = parser.parse_args(arguments)

    true_claims_data, original_claim_ids = read_ground_truth_jsonl(args.input_jsonl_file)
    predicted_claims_data = read_predicted_jsonl(args.input_predictions_jsonl_file)

    strict_score, label_accuracy, precision, recall, f1 = fever_score(predicted_claims_data, true_claims_data)
    print(f"strict_score: {strict_score}")
    print(f"label_accuracy: {label_accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


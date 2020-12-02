# -*- coding: utf-8 -*-
"""
This is the 3 class version, including the unverifiable cases.

This converts the fever dataset to the format for training the memory network. No data augmentation is performed.
To simplify training, we only choose 1 evidence set for each claim, and that evidence set is truncated to a max size
of two. This results in the "_chosen" data files. The "_true" file retains all titles and evidence sets and must be
used for evaluating the evidence. (On the other hand, the classification label is not affected by this filtering.)
In order to reduce the search space of possible wiki titles, for each claim, we only consider titles that have a
greedy lexical cover over the claim (as in get_titles_in_claim()). About 97% of claims have at least 1 supporting
wiki sentence that survive this filtering, and this results in around 300-400 article titles per claim. This means that
a small number of claims will have evidence sets that are not recoverable, but we find that the saving in computation
is worth that expense given computational resources. When performing search/inference, it is important to use the
"_covered" data files to guide search and NOT the "_true" file. Also, note that while some claims may have no
recoverable evidence sets in this scheme, no claims themselves are dropped, so the classification results are always
consistent.

Since some claims have multiple pieces of evidence sets, we choose one piece of evidence per id, under the following
scheme:

    We sort the evidence sets by [proportion of titles not in the claim, the first sentence not in the claim,
    the length of the evidence, the total number of unique articles in the evidence set,
    the index of the first sentence in the evidence set,
    and the total sum of the indexes of the sentences in the evidence set] and take the first in the resulting list.

The intuition is that lower values of the above will be 'easier' to learn.

Note that in the current version, we do not use the "_titles" files in the actual model code. We only generate them for
debugging/analysis purposes.

Since this version relies on the indexing of the covered wiki sentences, use the option --size_of_debug_set to create
smaller debug sets (rather than head -n X on the files).

The final eval data does not include the ground-truth (evidence nor labels) in the .jsonl file. In that case, the option
 --input_is_blind_test is used to generate the data for such cases (as for evaluation on coda lab).
 We simply create place holders for the unknown evidence, in a manner analogous to unverifiable cases in
 the ground-truth data.

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


SUPPORTS_STRING = "Supports:"
REFUTES_STRING = "Refutes:"
MOREINFO_STRING = "Unverifiable:"

UNK_TITLE_ID = -1
UNVERIFIABLE_TITLE_ID = -2

MOREINFO_HOLDER_TITLE_SYM = "UNVERIFIABLE_TITLE_HOLDER_SYM"  # just a unique string that never occurs as a real title

random.seed(1776)



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


def fix_negations(sentence):
    """
    De-tokenize 'did n't'-style negations
    :param sentence: List of tokens
    :return: De-tokenized list of tokens:
        Examples:
            did n't -> didn't
            can n't -> can't
    Assumes sentence does not start with "n't", which would suggest data problem.
    """
    formatted_sentence = []
    for token in sentence:
        if token == "n't":
            prev_token = formatted_sentence.pop()
            formatted_sentence.append(prev_token+"n't")
        else:
            formatted_sentence.append(token)
    return formatted_sentence


def reformat_wiki_url(wiki_url):
    """
    Reformat wiki url for use in downstream. Note for matching titles in the wiki data and the claims data, instead,
    only normalize with unicodedata.normalize('NFC', wiki_url.strip()).
    :param wiki_url:
    :return:
    """
    wiki_url = wiki_url.replace("_", " ")
    wiki_url = wiki_url.replace("-LRB-", "(")
    wiki_url = wiki_url.replace("-RRB-", ")")
    wiki_url = wiki_url.replace("-COLON-", ":")
    wiki_url = wiki_url.replace("-LSB-", "[")
    wiki_url = wiki_url.replace("-RSB-", "]")
    wiki_url = wiki_url.replace("-LCB-", "{")
    wiki_url = wiki_url.replace("-RCB-", "}")

    wiki_url = unicodedata.normalize('NFC', wiki_url.strip())
    return remove_internal_whitespace(wiki_url)


def remove_parens_from_string(input_string):
    # only remove depth 0 single set of well-matched parens
    # everything within (and including) the parens is dropped
    left_paren_index = input_string.find("(")
    right_paren_index = input_string.find(")")
    if left_paren_index != -1:
        if right_paren_index != -1:
            input_string = input_string[0:left_paren_index] + input_string[right_paren_index+len(")"):]
    return remove_internal_whitespace(input_string)



def filter_string(input_string):
    # assumes that input_string is normalized
    # lowercases
    # removes punctuation

    filtered = []
    for c in input_string:
        if c in string.punctuation:
            filtered.append(" ")
        else:
            filtered.append(c)

    return remove_internal_whitespace("".join(filtered)).lower()



def get_titles_in_claim(claim, titles_dict, original_titles_dict):
    titles_in_claim = set()
    claim_tokens = claim.split()

    for i in range(len(claim_tokens)):
        for j in range(len(claim_tokens), i, -1):
            possible_title = " ".join(claim_tokens[i:j])
            if possible_title in titles_dict:
                titles_in_claim.update(original_titles_dict[possible_title])  # multiple titles may correspond to this filtered title
                break  # greedily take longest for each starting token from the left
    return titles_in_claim


def construct_sentence(label_string, claim_string, chosen_title, chosen_sent_id, second_chosen_title, second_sent_id,
                       original_titles_to_wiki_sentences):
    if label_string == MOREINFO_STRING:
        # for unverifiable claims, we do not include evidence
        return f"{label_string} Claim: {claim_string}"
    else:
        if second_chosen_title is None:
            return f"{label_string} Claim: {claim_string} " \
                   f"Evidence: {chosen_title}, sentence {chosen_sent_id}: " \
                   f"{original_titles_to_wiki_sentences[chosen_title][chosen_sent_id]}"
        else:
            return f"{label_string} Claim: {claim_string} " \
                   f"Evidence: {chosen_title}, sentence {chosen_sent_id}: " \
                   f"{original_titles_to_wiki_sentences[chosen_title][chosen_sent_id]} " \
                   f"Evidence: {second_chosen_title}, sentence {second_sent_id}: {original_titles_to_wiki_sentences[second_chosen_title][second_sent_id]}"


def construct_evidence_only_sentence(chosen_title, chosen_sent_id, original_titles_to_wiki_sentences):
    return f"Evidence: {chosen_title}, sentence {chosen_sent_id}: " \
               f"{original_titles_to_wiki_sentences[chosen_title][chosen_sent_id]}"


def read_jsonl(filepath_with_name, tokenizer, titles_dict, original_titles_dict, original_titles_to_wiki_sentences,
               input_is_blind_test):
    id_to_claim = {}
    id_to_control = {}
    wiki_title_to_ids = defaultdict(list)  # chosen titles
    id_to_covered_titles = defaultdict(set)
    id_to_true_titles = defaultdict(set)

    id_to_chosen_wiki_sentence = defaultdict(list)  # chosen sentences -- human readable full claim + evidence
    id_to_chosen_wiki_sentence_only_evidence = defaultdict(list)  # chosen sentences, but only the evidence tuples
    id_to_covered_sentences = defaultdict(set)  # these are ALL sentences from the articles of the covered titles
    id_to_true_sentences = defaultdict(set)  # these are only the sentences that appear in the ground-truth evidence sets


    all_combined_covered_titles_set = set()

    number_of_verifiable_claims_with_at_least_1_covered_title = 0
    chosen_title_is_the_first_title = 0
    size_of_covered_title_sets = []

    size_of_covered_sentences_sets = []
    size_of_true_sentences_sets = []

    chosen_evidence_set_sizes = []

    num_dropped_blank_wiki_sentences = 0
    total_num_covered_sentences = 0  # this includes overlaps
    num_verifiable = 0
    num_supports = 0
    num_refutes = 0
    num_unverifiable = 0
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 1000 == 0:
                print(f"Currently processing line {line_id}.")
            line_id += 1
            line = line.strip()
            data = json.loads(line)
            id = int(data["id"])
            if input_is_blind_test:
                # To match the training/dev, here, we just create dummy vals for the label---here, always unverifiable
                verifiable = "NOT VERIFIABLE"
                label = "NOT ENOUGH INFO"
            else:
                verifiable = data["verifiable"]
                label = data["label"]
            claim = remove_internal_whitespace(data["claim"])
            claim_reformatted = " ".join(fix_negations(claim.split()))
            if claim != claim_reformatted:
                print(f"The wiki data has 'could n't' style tokenizations, but the main data does not. Check input.")
                print(f"claim: {claim}\nclaim2: {claim_reformatted}")
                assert False
            claim = unicodedata.normalize('NFC', claim.strip())
            claim_tokens, claim_wordpiece_len = filter_with_bert_tokenizer(tokenizer, claim.split())

            if input_is_blind_test:
                evidence = None
            else:
                evidence = data["evidence"]

            if verifiable == "VERIFIABLE":
                num_verifiable += 1
                if label == "SUPPORTS":
                    num_supports += 1
                elif label == "REFUTES":
                    num_refutes += 1
                else:
                    assert False
            else:
                assert verifiable == "NOT VERIFIABLE" and label == "NOT ENOUGH INFO"
                label = "UNVERIFIABLE"
                num_unverifiable += 1


            filtered_claim = filter_string(' '.join(claim_tokens))
            titles_in_claim = get_titles_in_claim(filtered_claim, titles_dict, original_titles_dict)
            all_combined_covered_titles_set.update(titles_in_claim)
            size_of_covered_title_sets.append(len(titles_in_claim))

            if len(titles_in_claim) == 0:
                print(f"WARNING: 0 covered claims in {' '.join(claim_tokens)}")

            sorted_evidence_sets = []
            all_true_titles_set = set()
            all_true_sentences_from_evidence_sets = set()
            # Note that for "NOT VERIFIABLE", evidence is still a list of lists (potentially from multiple
            # annotators), but the values are 'null', so here we never loop through evidence for those claims
            if verifiable == "VERIFIABLE":
                for evidence_set in evidence:
                    wiki_urls = []
                    wiki_urls_sent_id_pairs = []
                    cumulative_sent_ids = 0

                    for sent_meta_data in evidence_set:
                        wiki_url = sent_meta_data[2]
                        sent_id = sent_meta_data[3]
                        wiki_url_reformatted_string = reformat_wiki_url(wiki_url.strip())
                        wiki_urls.append(wiki_url_reformatted_string)
                        wiki_urls_sent_id_pairs.append((wiki_url_reformatted_string, sent_id))
                        cumulative_sent_ids += sent_id

                    all_true_sentences_from_evidence_sets.add(tuple(wiki_urls_sent_id_pairs))

                    num_evidence_titles_in_claim = 0
                    wiki_urls_set = set(wiki_urls)
                    all_true_titles_set.update(wiki_urls_set)
                    for wiki_url_reformatted_string in wiki_urls_set:
                        if wiki_url_reformatted_string in titles_in_claim:
                            num_evidence_titles_in_claim += 1
                    first_sent_id = wiki_urls_sent_id_pairs[0][1]
                    first_sent_title_in_claim = int(wiki_urls_sent_id_pairs[0][0] in titles_in_claim)
                    # the subtraction from 1 given the direction of the subsequent sort:
                    proportion_titles_not_in_claim = 1 - (num_evidence_titles_in_claim / len(wiki_urls_set))
                    sorted_evidence_sets.append([proportion_titles_not_in_claim, 1-first_sent_title_in_claim,
                                                 len(evidence_set), len(wiki_urls_set), first_sent_id,
                                                 cumulative_sent_ids, wiki_urls_sent_id_pairs])

                sorted_evidence_sets = sorted(sorted_evidence_sets, key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))

                # chose the evidence set and save all true and covered titles
                chosen_evidence_set = sorted_evidence_sets[0]
                wiki_urls_sent_id_pairs = chosen_evidence_set[6]
                chosen_title = wiki_urls_sent_id_pairs[0][0]
                chosen_sent_id = wiki_urls_sent_id_pairs[0][1]
                if chosen_title not in titles_in_claim:  # try to find the next title that is in the claim, if it exists
                    for wiki_urls_sent_id_pair in wiki_urls_sent_id_pairs:
                        possible_title = wiki_urls_sent_id_pair[0]
                        if possible_title in titles_in_claim:
                            chosen_title = possible_title
                            chosen_sent_id = wiki_urls_sent_id_pair[1]
                            break
                else:
                    chosen_title_is_the_first_title += 1
                if chosen_title in titles_in_claim:
                    number_of_verifiable_claims_with_at_least_1_covered_title += 1

                second_chosen_title = None
                second_sent_id = None
                if len(wiki_urls_sent_id_pairs) > 1:
                    for wiki_urls_sent_id_pair in wiki_urls_sent_id_pairs:
                        possible_title = wiki_urls_sent_id_pair[0]
                        possible_sent_id = wiki_urls_sent_id_pair[1]
                        if (possible_title != chosen_title) or \
                            (possible_title == chosen_title and possible_sent_id != chosen_sent_id):
                            second_chosen_title = possible_title
                            second_sent_id = possible_sent_id
                            break

                chosen_wiki_urls_sent_id_pairs = [(chosen_title, chosen_sent_id)]
                if second_chosen_title is None:
                    chosen_evidence_set_sizes.append(1)
                else:
                    chosen_wiki_urls_sent_id_pairs.append((second_chosen_title, second_sent_id))
                    chosen_evidence_set_sizes.append(2)
            else:
                # Need to create holders for the nonverifiable claims
                chosen_title = MOREINFO_HOLDER_TITLE_SYM
                chosen_sent_id = -1
                second_chosen_title = None
                second_sent_id = None
                chosen_wiki_urls_sent_id_pairs = []
            assert id not in id_to_claim
            id_to_claim[id] = f"Claim: {' '.join(claim_tokens)}"
            wiki_title_to_ids[chosen_title].append(id)
            id_to_control[id] = f"{label},{chosen_sent_id}"

            id_to_covered_titles[id] = titles_in_claim
            id_to_true_titles[id] = all_true_titles_set

            #  In this version, we pre-pend the correct decision label, title, and sentence id to the sentence
            if label == "SUPPORTS":
                label_string = SUPPORTS_STRING
            elif label == "REFUTES":
                label_string = REFUTES_STRING
            elif label == "UNVERIFIABLE":
                label_string = MOREINFO_STRING
            else:
                assert False

            id_to_chosen_wiki_sentence[id] = construct_sentence(label_string, ' '.join(claim_tokens), chosen_title,
                                                                chosen_sent_id, second_chosen_title, second_sent_id,
                                                                original_titles_to_wiki_sentences)

            id_to_chosen_wiki_sentence_only_evidence[id] = chosen_wiki_urls_sent_id_pairs

            # In this version, we add ALL sentences from the covered titles:
            # (Note that some sentences could be blank, such as new lines in the original article; we drop them
            # in this version.)
            sentences_from_covered_titles_in_claim = set()
            for covered_title in titles_in_claim:
                for wiki_sent_id in original_titles_to_wiki_sentences[covered_title]:
                    if len(original_titles_to_wiki_sentences[covered_title][wiki_sent_id]) > 0 and \
                            len(covered_title) > 0:
                        sentences_from_covered_titles_in_claim.add(
                            construct_evidence_only_sentence(covered_title, wiki_sent_id,
                                                             original_titles_to_wiki_sentences))
                        total_num_covered_sentences += 1
                    else:
                        num_dropped_blank_wiki_sentences += 1
            id_to_covered_sentences[id] = sentences_from_covered_titles_in_claim
            size_of_covered_sentences_sets.append(len(sentences_from_covered_titles_in_claim))
            id_to_true_sentences[id] = list(all_true_sentences_from_evidence_sets)
            if len(all_true_sentences_from_evidence_sets) > 0:
                size_of_true_sentences_sets.append(len(all_true_sentences_from_evidence_sets))

    if not input_is_blind_test:
        print(f"Proportion of verifiable claims with at least one lexically covered title: "
              f"{number_of_verifiable_claims_with_at_least_1_covered_title/num_verifiable}")
        print(
            f"Proportion of verifiable claims for which the chosen title is the first title in the chosen evidence set: "
            f"{chosen_title_is_the_first_title / num_verifiable}")
    print(f"Total unique covered titles: {len(all_combined_covered_titles_set)}")
    print(f"Size of covered titles sets: mean: {np.mean(size_of_covered_title_sets)}; "
          f"min: {np.min(size_of_covered_title_sets)}, max: {np.max(size_of_covered_title_sets)}")
    print(
        f"Size of covered sentences sets: mean: {np.mean(size_of_covered_sentences_sets)}; "
        f"min: {np.min(size_of_covered_sentences_sets)}, max: {np.max(size_of_covered_sentences_sets)}")
    if not input_is_blind_test:
        print(
            f"Size of non-zero (i.e., verifiable) true sentences sets: mean: {np.mean(size_of_true_sentences_sets)}; "
            f"min: {np.min(size_of_true_sentences_sets)}, max: {np.max(size_of_true_sentences_sets)}")
    print(f"Number of blank wiki sentences dropped in the covered titles sets: {num_dropped_blank_wiki_sentences}")
    print(f"Total verifiable claims: {num_verifiable}; number supported: {num_supports}; number refuted: {num_refutes}")
    print(f"Total unverifiable claims: {num_unverifiable} out of {num_unverifiable+num_verifiable}: proportion: "
          f"{num_unverifiable/(num_unverifiable+num_verifiable)}")
    print(f"Total number of covered sentences (across all claims, including duplicates): {total_num_covered_sentences}")

    if not input_is_blind_test:
        print(
            f"Size of evidence set for the chosen sentences: mean: {np.mean(chosen_evidence_set_sizes)}; "
            f"min: {np.min(chosen_evidence_set_sizes)}, max: {np.max(chosen_evidence_set_sizes)}, "
            f"len: {len(chosen_evidence_set_sizes)}")

        chosen_evidence_set_sizes_e1 = []
        chosen_evidence_set_sizes_e2 = []
        for set_size in chosen_evidence_set_sizes:
            if set_size == 2:
                chosen_evidence_set_sizes_e2.append(set_size)
            if set_size == 1:
                chosen_evidence_set_sizes_e1.append(set_size)

        print(
            f"Size of evidence set for the chosen sentence == 1: len: {len(chosen_evidence_set_sizes_e1)},"
            f"proportion of total: {len(chosen_evidence_set_sizes_e1)/len(chosen_evidence_set_sizes)}")

        print(
            f"Size of evidence set for the chosen sentence == 2: len: {len(chosen_evidence_set_sizes_e2)},"
            f"proportion of total: {len(chosen_evidence_set_sizes_e2)/len(chosen_evidence_set_sizes)}")

    return id_to_claim, wiki_title_to_ids, id_to_control, id_to_covered_titles, id_to_true_titles, \
           id_to_chosen_wiki_sentence, id_to_covered_sentences, id_to_true_sentences, \
           id_to_chosen_wiki_sentence_only_evidence


def get_claim_to_titles_lines(original_titles_to_wiki_sentences,
                              id_to_claim, wiki_title_to_ids, id_to_control, id_to_covered_titles, id_to_true_titles,
                              id_to_chosen_wiki_sentence, id_to_covered_sentences, id_to_true_sentences,
                              id_to_chosen_wiki_sentence_only_evidence, size_of_debug_set):
    out_lines = []
    control_out_lines = []
    out_true_titles_lines = []
    out_covered_titles_lines = []

    out_chosen_sentences_lines = []

    out_true_sentences_lines = []
    out_covered_sentences_lines = []

    filtered_wiki_sents_to_wiki_id = defaultdict(int)
    if size_of_debug_set != -1:
        remaining_sentences = size_of_debug_set
    # The files are grouped by wiki title to make it easier to inspect the output for debugging/analysis purposes
    # For unverifiable claims, here we just use a dummy holder symbol to capture those ids
    assert MOREINFO_HOLDER_TITLE_SYM in wiki_title_to_ids
    for wiki_title in wiki_title_to_ids:
        for id in wiki_title_to_ids[wiki_title]:  # more than one claim may use this document (hence, the loop)
            if size_of_debug_set != -1:
                if remaining_sentences <= 0:
                    break
                else:
                    remaining_sentences -= 1
            out_lines.append(f"{id_to_claim[id]}\t{wiki_title}\n")
            control_out_lines.append(f"{id},{id_to_control[id]}\n")
            out_true_titles_lines.append("\t".join(id_to_true_titles[id])+"\n")
            out_covered_titles_lines.append("\t".join(id_to_covered_titles[id]) + "\n")

            out_chosen_sentences_lines.append(f"{id_to_claim[id]}\t{id_to_chosen_wiki_sentence[id]}\n")

            wiki_ids = []
            for covered_sent in id_to_covered_sentences[id]:
                if covered_sent not in filtered_wiki_sents_to_wiki_id:
                    # assign a new id to this wiki sent
                    filtered_wiki_sents_to_wiki_id[covered_sent] = len(filtered_wiki_sents_to_wiki_id)
                wiki_ids.append(f"{filtered_wiki_sents_to_wiki_id[covered_sent]}")
            out_covered_sentences_lines.append("\t".join(wiki_ids) + "\n")

            # the following are now saved as json:
            out_true_sentences_lines.append(id_to_true_sentences[id])

    filtered_wiki_sents_to_wiki_id_lines = []  # mapping from evidence to wiki sentence ids
    for covered_sent in filtered_wiki_sents_to_wiki_id:
        filtered_wiki_sents_to_wiki_id_lines.append(f"{covered_sent}\t{filtered_wiki_sents_to_wiki_id[covered_sent]}\n")

    # Second pass to associate the chosen evidence with wiki dictionary
    out_chosen_sentences_only_evidence_lines = []
    num_chosen_evidence_not_in_covered_dict = 0

    if size_of_debug_set != -1:
        remaining_sentences = size_of_debug_set

    for wiki_title in wiki_title_to_ids:
        for id in wiki_title_to_ids[wiki_title]:  # more than one claim may use this document (hence, the loop)
            if size_of_debug_set != -1:
                if remaining_sentences <= 0:
                    break
                else:
                    remaining_sentences -= 1
            wiki_ids = []
            chosen_wiki_urls_sent_id_pairs = id_to_chosen_wiki_sentence_only_evidence[id]
            for wiki_urls_sent_id_pair in chosen_wiki_urls_sent_id_pairs:
                covered_sent = construct_evidence_only_sentence(wiki_urls_sent_id_pair[0], wiki_urls_sent_id_pair[1],
                                                                original_titles_to_wiki_sentences)
                if covered_sent not in filtered_wiki_sents_to_wiki_id:
                    num_chosen_evidence_not_in_covered_dict += 1
                    wiki_ids.append(f"{UNK_TITLE_ID}")
                else:
                    wiki_ids.append(f"{filtered_wiki_sents_to_wiki_id[covered_sent]}")
            if len(wiki_ids) == 0:
                assert id in wiki_title_to_ids[MOREINFO_HOLDER_TITLE_SYM], f"{id_to_claim[id]}"
            if id in wiki_title_to_ids[MOREINFO_HOLDER_TITLE_SYM]:
                assert len(wiki_ids) == 0, f"{id_to_claim[id]}"
                wiki_ids = [str(UNVERIFIABLE_TITLE_ID)]  # for unverifiable claims, we set the chosen evidence id to -2
            out_chosen_sentences_only_evidence_lines.append("\t".join(wiki_ids) + "\n")
    print(f"Number of chosen evidence sentences not in the covered sentences (including for other claims): "
          f"{num_chosen_evidence_not_in_covered_dict}")
    return out_lines, control_out_lines, out_true_titles_lines, out_covered_titles_lines, out_chosen_sentences_lines, \
           out_true_sentences_lines, out_covered_sentences_lines, out_chosen_sentences_only_evidence_lines, \
           filtered_wiki_sents_to_wiki_id_lines


def save_jsonl_evidence_lines(filename_with_path, evidence_sets):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for evidence_set in evidence_sets:
            json.dump(evidence_set, f)
            f.write('\n')


def read_titles_and_sentences_from_wiki_file(filepath_with_name):
    """
    Read the wikipedia articles data.
    :param filepath_with_name: Preprocessed wikipedia file
    :return:
        titles_dict: filtered title->(currently unused int val)
        original_titles_dict: filtered title->set(corresponding titles)
        original_titles_to_wiki_sentences: title->sent_num->sentence string
        original_titles_to_title_hashes: title->original title hash
    """
    titles_dict = {}
    original_titles_dict = defaultdict(set)
    original_titles_to_wiki_sentences = defaultdict(dict)
    original_titles_to_title_hashes = defaultdict(set)
    number_of_duplicates_due_to_filtering = 0
    line_id = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 100000 == 0:
                print(f"Currently reading titles line {line_id}.")
            line_id += 1
            line = line.strip().split("\t")
            assert len(line) >= 4
            wiki_title = line[0].strip()  # this is the original title hash
            wiki_title_reformatted = reformat_wiki_url(wiki_title)
            filtered_title = filter_string(remove_parens_from_string(wiki_title_reformatted))
            if filtered_title in titles_dict:
                number_of_duplicates_due_to_filtering += 1
            titles_dict[filtered_title] = 0
            # Note that some titles may be collapsed, so we're adding the un-filtered title as a set
            original_titles_dict[filtered_title].add(wiki_title_reformatted)

            assert wiki_title_reformatted not in original_titles_to_wiki_sentences, f"ERROR: Unexpected " \
                                                                                    f"title collapse: " \
                                                                                    f"{wiki_title_reformatted}"
            # collect the numbered sentences in the article:
            for sent in line[3:]:
                sent_split = sent.split("::")
                assert len(sent_split) >= 2, f"Unexpected line formatting: line: {sent_split}"
                sent_num = int(sent_split[0].strip())
                #  The sentence itself might include ::, so we treat the rest as a string
                sent_string = sent[len(f"{sent_num}::"):].strip()
                sent_string = unicodedata.normalize('NFC', sent_string)
                original_titles_to_wiki_sentences[wiki_title_reformatted][sent_num] = sent_string

            # Need to retain the mapping back to the title hashes, as these are used in the final eval:
            original_titles_to_title_hashes[wiki_title_reformatted].add(wiki_title)

    print(f"Total number of titles: {len(original_titles_to_wiki_sentences)}")
    print(f"number_of_duplicates_due_to_filtering: {number_of_duplicates_due_to_filtering}")
    return titles_dict, original_titles_dict, original_titles_to_wiki_sentences, original_titles_to_title_hashes


def get_title_mapping_lines(original_titles_to_title_hashes):
    titles_lines = []
    one_to_many_title_mappings = 0
    for original_title in original_titles_to_title_hashes:
        if len(original_titles_to_title_hashes[original_title]) > 1:
            one_to_many_title_mappings += 1
        title_hash_line = "\t".join(original_titles_to_title_hashes[original_title])
        titles_lines.append(f"{original_title}\t{title_hash_line}\n")
    print(f"Number of reformatted titles mapped to multiple title hashes: {one_to_many_title_mappings}")
    return titles_lines


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_jsonl_file', type=str, help="input_jsonl_file")
    parser.add_argument('--input_wiki_file', type=str, help="input_wiki_file")
    parser.add_argument('--output_file', type=str, help="output_binaryevalformat_file: claim\tchosen title")
    parser.add_argument('--output_control_file', type=str, help="output_control_file")
    parser.add_argument('--output_true_titles_file', type=str, help="output_true_titles_file")
    parser.add_argument('--output_covered_titles_file', type=str, help="output_covered_titles_file")

    parser.add_argument('--output_chosen_sentences_file', type=str, help="output_chosen_sentences_file")
    parser.add_argument('--output_true_sentences_file', type=str, help="output_true_sentences_file")
    parser.add_argument('--output_covered_sentences_file', type=str, help="output_covered_sentences_file")
    parser.add_argument('--output_covered_sentences_dictionary_file', type=str, help="output_covered_sentences_dictionary_file")

    parser.add_argument('--output_filtered_titles_to_title_hashes_file', type=str, help="output_filtered_titles_to_title_hashes_file")
    parser.add_argument('--output_chosen_sentences_only_evidence_file', type=str,
                        help="output_chosen_sentences_only_evidence_file")

    parser.add_argument('--size_of_debug_set', type=int, default=-1, help="If not -1, this will create a small set "
                                                                          "for debugging that only contains the first"
                                                                          "--size_of_debug_set claims in "
                                                                          "--input_jsonl_file. This, in effect,"
                                                                          "filters "
                                                                          "--output_covered_sentences_dictionary_file.")
    # for blind test data
    parser.add_argument("--input_is_blind_test", default=False, action='store_true',
                        help="For preprocessing the blind test data (which lacks ground-truth labels and evidence"
                             "in --input_jsonl_file).")

    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)
    # First, we retrieve the wiki titles. Downstream, we primarily use a version of the titles with the underscores
    # and related holder symbols removed. Additionally, when filtering the titles over covered tokens in the claims,
    # we lowercase and remove parenthetical information from the titles.
    titles_dict, original_titles_dict, original_titles_to_wiki_sentences, original_titles_to_title_hashes = \
        read_titles_and_sentences_from_wiki_file(args.input_wiki_file)
    # Next, we read and filter the evidence sets from the original jsonl file.
    id_to_claim, wiki_title_to_ids, id_to_control, id_to_covered_titles, id_to_true_titles, id_to_chosen_wiki_sentence,\
        id_to_covered_sentences, id_to_true_sentences, id_to_chosen_wiki_sentence_only_evidence = \
        read_jsonl(args.input_jsonl_file, tokenizer, titles_dict, original_titles_dict,
                   original_titles_to_wiki_sentences, args.input_is_blind_test)
    # Finally, we reformat the lines (sorted by the first chosen title), and establish the mapping to the wiki evidence
    # sentences
    out_lines, control_out_lines, out_true_titles_lines, out_covered_titles_lines, out_chosen_sentences_lines, \
        out_true_sentences_lines, out_covered_sentences_lines, out_chosen_sentences_only_evidence_lines, \
        filtered_wiki_sents_to_wiki_id_lines = \
        get_claim_to_titles_lines(original_titles_to_wiki_sentences, id_to_claim, wiki_title_to_ids, id_to_control,
                                  id_to_covered_titles, id_to_true_titles, id_to_chosen_wiki_sentence,
                                  id_to_covered_sentences, id_to_true_sentences,
                                  id_to_chosen_wiki_sentence_only_evidence, args.size_of_debug_set)
    save_lines(args.output_file, out_lines)
    save_lines(args.output_control_file, control_out_lines)
    save_lines(args.output_true_titles_file, out_true_titles_lines)
    save_lines(args.output_covered_titles_file, out_covered_titles_lines)

    save_lines(args.output_chosen_sentences_file, out_chosen_sentences_lines)
    save_lines(args.output_chosen_sentences_only_evidence_file, out_chosen_sentences_only_evidence_lines)
    save_lines(args.output_covered_sentences_file, out_covered_sentences_lines)
    save_lines(args.output_covered_sentences_dictionary_file, filtered_wiki_sents_to_wiki_id_lines)

    save_jsonl_evidence_lines(args.output_true_sentences_file, out_true_sentences_lines)

    save_lines(args.output_filtered_titles_to_title_hashes_file,
               get_title_mapping_lines(original_titles_to_title_hashes))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


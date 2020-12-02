# -*- coding: utf-8 -*-
"""
This converts the fever wiki dataset to the following:

[wiki_id] TAB [num lines] TAB [detokenized wiki] TAB [sent 0] ... TAB ... [sent N]

The input sentences are filtered with the BERT tokenizer to eliminate tokens that are elided by the tokenizer to avoid
downstream mis-matches with label alignments.

To process the wiki text, we:
1. Revert holder symbols (e.g., -LRB-) and remove extraneous internal whitespace
2. Fix negations
3. Detokenize to remove spaces, etc. via MosesDetokenizer

Note that [detokenized wiki] includes the full text (with line info lost), and that [wiki_id] is the raw id (unprocessed).
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
from sacremoses import MosesDetokenizer
import glob

random.seed(1776)



def remove_internal_whitespace(line_string):
    return " ".join(line_string.strip().split())

def filter_with_bert_tokenizer(tokenizer, sentence_tokens):
    filtered_tokens = []
    wordpiece_len = 0
    num_filtered_tokens = 0
    for token in sentence_tokens:
        bert_tokens = tokenizer.tokenize(token)
        if len(bert_tokens) == 0:  # must be a special character filtered by BERT
            #assert False, f"ERROR: Tokenizer filtering is not expected to occur with this data."
            #pass
            print(f"Ignoring {token} with label {label}")
            num_filtered_tokens += 1
        else:
            filtered_tokens.append(token)
            wordpiece_len += len(bert_tokens)
    return filtered_tokens, wordpiece_len, num_filtered_tokens


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


def split_wiki_url(wiki_url):
    wiki_url = wiki_url.replace("_", " ")
    wiki_url = wiki_url.replace("-LRB-", "(")
    wiki_url = wiki_url.replace("-RRB-", ")")
    wiki_url = wiki_url.replace("-COLON-", ":")
    return remove_internal_whitespace(wiki_url)


def detokenize_wiki(wiki_text, md):
    wiki_text = wiki_text.replace("-LRB-", "(")
    wiki_text = wiki_text.replace("-RRB-", ")")
    wiki_text = wiki_text.replace("-COLON-", ":")
    wiki_text = wiki_text.replace("-LSB-", "[")
    wiki_text = wiki_text.replace("-RSB-", "]")
    wiki_text = wiki_text.replace("-LCB-", "{")
    wiki_text = wiki_text.replace("-RCB-", "}")
    wiki_text = wiki_text.replace("``", '"')
    wiki_text = wiki_text.replace("''", '"')

    wiki_text = remove_internal_whitespace(wiki_text)
    wiki_text = " ".join(fix_negations(wiki_text.split()))  # this is used as MosesDetokenizer does not collapse these
    wiki_text = md.detokenize(wiki_text.split())  # attempt to remove the wiki tokenization, since claims do not have punctuation splits
    return wiki_text


def read_wiki_jsonl_and_write_stream(input_jsonl_files, tokenizer, output_file):
    processed_lines_out_file_object = codecs.open(output_file, 'w', 'utf-8')

    md = MosesDetokenizer(lang='en')

    wiki_text_lens_by_wordpiece = []
    num_sentences_by_wiki = []

    num_empty_lines = 0
    num_filtered_tokens_total = 0
    total_saved_lines = 0

    for file_i, filepath_with_name in enumerate(input_jsonl_files):
        line_id = 0
        print(f"Currently processing {file_i} of {len(input_jsonl_files)}: {filepath_with_name}")
        with codecs.open(filepath_with_name, encoding="utf-8") as f:
            for line in f:
                if line_id % 10000 == 0:
                    print(f"\tCurrently processing line {line_id}.")
                line_id += 1

                line = line.strip()
                data = json.loads(line)
                id = data["id"].strip()
                wiki_text = data["text"].strip()
                wiki_by_lines = data["lines"]
                if len(id) != 0 and len(wiki_text) != 0:
                    wiki_text = detokenize_wiki(wiki_text, md)
                    filtered_tokens, wordpiece_len, num_filtered_tokens = filter_with_bert_tokenizer(tokenizer, wiki_text.split())

                    wiki_text_lens_by_wordpiece.append(wordpiece_len)
                    num_filtered_tokens_total += num_filtered_tokens
                    # the sentences are indexed by 0, but the final line is always blank, so this is the correct count:
                    num_sentences = int(wiki_by_lines.strip().split()[-1])
                    if num_sentences == 0:
                        print(f"WARNING: 0 sentences found in line {line_id-1} of {filepath_with_name}.")
                    num_sentences_by_wiki.append(num_sentences)
                    wiki_by_sent = []
                    for meta_sent in wiki_by_lines.split("\n"):
                        meta_sent_split = meta_sent.split("\t")
                        if len(meta_sent_split) < 2:
                            print(f"WARNING: Mal-formed wiki by sent in line {line_id - 1} of {filepath_with_name}: {meta_sent}.")
                        else:
                            if meta_sent_split[0].isdigit():
                                # first index is sentence number; indexes >= 2 are NER entries
                                meta_sent_split_index = int(meta_sent_split[0])
                                meta_sent_string = meta_sent_split[1]
                                if meta_sent_string == "":  # some lines may be blank (corresponding to new lines)
                                    meta_sent_filtered_tokens = []
                                else:
                                    meta_sent_string = detokenize_wiki(meta_sent_string, md)
                                    meta_sent_filtered_tokens, _, _ = filter_with_bert_tokenizer(tokenizer, meta_sent_string.split())

                                wiki_by_sent.append(' '.join([f"{meta_sent_split_index}::"] + meta_sent_filtered_tokens))
                            else:
                                print(
                                    f"WARNING: Mal-formed wiki by sent (missing sentence number) in line {line_id - 1} of {filepath_with_name}: {meta_sent}.")
                    if len(wiki_by_sent) == 0:
                        print(
                            f"WARNING: Mal-formed wiki by sent (len 0) in line {line_id - 1} of {filepath_with_name}: {id}.")
                        print(f"\tSetting the wiki by sent to ERROR_BLANK")
                        wiki_by_sent = f"ERROR_BLANK"
                    else:
                        wiki_by_sent = '\t'.join(wiki_by_sent) #[0:-1])  # last entry is blank
                    processed_line_to_save = f"{id}\t{num_sentences}\t{' '.join(filtered_tokens)}\t{wiki_by_sent}\n"
                    processed_lines_out_file_object.write(processed_line_to_save)
                    processed_lines_out_file_object.flush()
                    total_saved_lines += 1

                else:
                    num_empty_lines += 1

    processed_lines_out_file_object.close()
    print(f"Number of empty lines: {num_empty_lines}")
    print(f"Number of BERT-filtered tokens: {num_filtered_tokens_total}")
    print(f"Wiki: By wordpiece: Mean length: {np.mean(wiki_text_lens_by_wordpiece)}; std: {np.std(wiki_text_lens_by_wordpiece)}; "
          f"min: {np.min(wiki_text_lens_by_wordpiece)}, max: {np.max(wiki_text_lens_by_wordpiece)}")
    print(
        f"Number of sentences per wiki document: Mean: {np.mean(num_sentences_by_wiki)}; std: {np.std(num_sentences_by_wiki)}; "
        f"min: {np.min(num_sentences_by_wiki)}, max: {np.max(num_sentences_by_wiki)}")

    print(f"Total lines: {total_saved_lines}")


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', type=str, help="input_dir")
    parser.add_argument('--output_file', type=str, help="output_file")

    # for BERT tokenizer:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    args = parser.parse_args(arguments)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case,
                                              cache_dir=args.bert_cache_dir)

    input_jsonl_files = glob.glob(f"{args.input_dir}/*.jsonl")
    print(f"Total jsonl files under consideration: {len(input_jsonl_files)}")
    read_wiki_jsonl_and_write_stream(input_jsonl_files, tokenizer, args.output_file)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


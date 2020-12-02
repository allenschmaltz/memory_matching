# TODO: update documentation
# Note that the conventions here will get a major update in the refactored version, most of which hinges around
# simplifying the input data structures for query and support sequences.

import constants
import utils_search

from sklearn.utils import shuffle

import codecs

from collections import defaultdict

import torch

import re  # used in fce_formatter

import time
import numpy as np
import json

def fce_formatter(sentence):
    """
    Format sentence
    :param sentence: List of tokens
    :return: List of tokens, each of which is lowercased, with digits replaced with 0
    """
    formatted_sentence = []
    for token in sentence:
        token = token.lower()
        token = re.sub(r'\d', '0', token)
        formatted_sentence.append(token)
    return formatted_sentence

def lowercase_formatter(sentence):
    """
    Format sentence by lowercasing every token
    :param sentence: List of tokens
    :return: List of tokens, each of which is lowercased
    """
    formatted_sentence = []
    for token in sentence:
        token = token.lower()
        formatted_sentence.append(token)
    return formatted_sentence

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


def bert_tokenize(tokenizer, sentence_tokens):
    return tokenizer.tokenize(" ".join(sentence_tokens))

def get_lines(filepath_with_name, class_labels, data_formatter=None, tokenizer=None, input_is_untokenized=False):
    """
    Tokenize the input
    :param filepath_with_name: Filename string with: One sentence per line: classification label followed by one space and then the sentence: 0|1 Sentence text\n
    :param class_labels: List of int labels; labels in filepath_with_name must be members of this set
    :param data_formatter: None or "fce", which lowercases and replaces digits to 0; Only exists to match some previous work
    :param tokenizer: None, or BertTokenizer; Converts input to WordPieces, if provided
    :return:
        sentences - List of tokenized sentences (each of which is a list of tokens)
        labels - List of int sentence labels
        bert_to_original_tokenization_maps - [] or if tokenizer is provided, (list of) list of ints mapping indecies in tokenized sentences with index
            in original_sentences: e.g., len(bert_to_original_tokenization_maps[10]) == len(sentences[10])
        original_sentences: The untokenized sentences
    """
    titles = []
    sentences = []
    all_titles2freq = defaultdict(int)
    bert_to_original_tokenization_maps = []
    original_sentences = []  # without additional tokenization
    original_titles = []  # without additional tokenization
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            assert len(line) == 2
            title = line[1].split()
            sentence = line[0].split()
            original_sentence = list(sentence)
            original_title = list(title)
            if data_formatter and data_formatter != "":
                if data_formatter == "fce":
                    sentence = fce_formatter(sentence)
                elif data_formatter == "lowercase":
                    sentence = lowercase_formatter(sentence)
                else:
                    assert False, "ERROR: undefined data formatter"

            if tokenizer is not None:
                if not input_is_untokenized:
                    sentence = fix_negations(sentence)  # this is used to make the input match the expected input of the BERT tokenizer
                bert_to_original_tokenization_map = []  # maintain alignment for word-level labels
                token_i_neg_offset = 0
                bert_local_tokenization_check = []  # this is to check whether per-(whitespace-delimited-)word tokenization matches tokenization of the full sentence string (i.e., ensuring that running the tokenizer at the word level is the same as running at the sentence level)
                for token_i, token in enumerate(sentence):
                    bert_tokens = tokenizer.tokenize(token)
                    bert_local_tokenization_check.extend(bert_tokens)
                    for bert_token_i, _ in enumerate(bert_tokens):
                        if token.endswith("n't") and not input_is_untokenized:
                            if bert_token_i == 0:
                                bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                                token_i_neg_offset += 1 # original token was two tokens
                            else:
                                bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                        else:
                            bert_to_original_tokenization_map.append(token_i+token_i_neg_offset)
                #sentence = tokenizer.tokenize(" ".join(sentence))
                sentence = bert_tokenize(tokenizer, sentence)
                if len(bert_local_tokenization_check) != len(sentence) or " ".join(bert_local_tokenization_check) != " ".join(sentence):
                    assert False, F"ERROR: UNEXPECTED TOKENIZATION: per-token: {bert_local_tokenization_check} || sentence-level: {sentence}"
                bert_to_original_tokenization_maps.append(bert_to_original_tokenization_map)
                # also tokenize the title:
                title = bert_tokenize(tokenizer, title)

            titles.append(title)
            sentences.append(sentence)
            original_sentences.append(original_sentence)
            original_titles.append(original_title)
            all_titles2freq[tuple(title)] += 1  # note this is the BERT tokenized title
    return sentences, titles, bert_to_original_tokenization_maps, original_sentences, original_titles, all_titles2freq


def convert_target_labels_lines_to_bert_tokenization(train_seq_y, train_bert_to_original_tokenization_maps):
    assert len(train_seq_y) == len(train_bert_to_original_tokenization_maps)
    converted_seq_y = []
    for sentence_id in range(len(train_seq_y)):
        one_seq_y = train_seq_y[sentence_id]
        one_bert_to_original_tokenization_maps = train_bert_to_original_tokenization_maps[sentence_id]
        new_seq_y = []
        assert len(one_seq_y) <= len(one_bert_to_original_tokenization_maps)
        for token_id in range(len(one_bert_to_original_tokenization_maps)):
            new_seq_y.append( one_seq_y[one_bert_to_original_tokenization_maps[token_id]] )
        converted_seq_y.append(new_seq_y)

    return converted_seq_y

def read_metric_data(training_file, dev_file, test_file, training_seq_labels_file=None,
                     dev_seq_labels_file=None, test_seq_labels_file=None, data_formatter=None, tokenizer=None,
                     input_is_untokenized=False):
    """
    Process the training, dev, and test files (text and sequence labels), as applicable (for train, test, etc.)
    """
    data = {}
    if training_file != "":
        if training_seq_labels_file is None:
            data["train_x"], data["train_y"], _, _, data["train_original_titles"], data["train_all_titles2freq"] = get_lines(training_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            # we can't shuffle here, because the true and covered title files are aligned with the above
            #data["train_x"], data["train_y"], data["train_original_titles"] = shuffle(train_x, train_y, data["train_original_titles"], random_state=1776)
            data["dev_x"], data["dev_y"], _, data["dev_original_sentences"], data["dev_original_titles"], data["dev_all_titles2freq"] = get_lines(dev_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
        else:
            assert False
            assert dev_seq_labels_file is not None, "The dev sequence labels file must be provided, as well."
            train_x, train_y, train_bert_to_original_tokenization_maps, _ = get_lines(training_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            train_seq_y = get_target_labels_lines(training_seq_labels_file, convert_to_int=True)
            if tokenizer is not None:
                assert len(train_bert_to_original_tokenization_maps) > 0
                print(f"Converting/spreading training sequence labels to match Bert tokenization")
                train_seq_y = convert_target_labels_lines_to_bert_tokenization(train_seq_y, train_bert_to_original_tokenization_maps)

            data["train_x"], data["train_y"], data["train_seq_y"] = shuffle(train_x, train_y, train_seq_y, random_state=1776)
            data["dev_seq_y"] = get_target_labels_lines(dev_seq_labels_file, convert_to_int=True)
            data["dev_x"], data["dev_y"], data["dev_bert_to_original_tokenization_maps"], data["dev_sentences"] = get_lines(dev_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
            if tokenizer is not None:
                data["dev_spread_seq_y"] = convert_target_labels_lines_to_bert_tokenization(data["dev_seq_y"],
                                                                               data["dev_bert_to_original_tokenization_maps"])
    elif test_file != "":
        data["test_x"], data["test_y"], _, data["test_original_sentences"], data["test_original_titles"], data["test_all_titles2freq"] = get_lines(
            test_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)

        # assert False
        # data["test_x"], data["test_y"], data["test_bert_to_original_tokenization_maps"], data["test_sentences"] = get_lines(test_file, constants.AESW_CLASS_LABELS, data_formatter, tokenizer, input_is_untokenized)
        # if test_seq_labels_file is not None:
        #     data["test_seq_y"] = get_target_labels_lines(test_seq_labels_file, convert_to_int=True)
        #     if tokenizer is not None:
        #         data["test_spread_seq_y"] = convert_target_labels_lines_to_bert_tokenization(data["test_seq_y"],
        #                                                                        data["test_bert_to_original_tokenization_maps"])
    return data


def read_aesw_test_target_comparison_file(filepath_with_name):
    sentences = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            sentences.append(line)
    return sentences


def preprocess_sentences_seq_labels_for_training(seq_labels, max_length):
    padded_seq_y = []
    padded_seq_y_mask = []

    for labels in seq_labels:
        labels_truncated = labels[0:max_length]
        mask = [1] * len(labels_truncated)
        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        labels_truncated.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(labels_truncated)))
        mask.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(mask)))
        # add start padding
        labels_truncated = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + labels_truncated
        mask = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + mask
        assert len(labels_truncated) == max_length + 2*constants.PADDING_SIZE and len(labels_truncated) == len(mask)
        padded_seq_y.append(labels_truncated)
        padded_seq_y_mask.append(mask)
    return padded_seq_y, padded_seq_y_mask


def preprocess_sentences_without_bert(sentences, word_to_idx, max_length):
    idx_sentences = []
    for sentence in sentences:
        idx_sentence = []
        sentence_truncated = sentence[0:max_length]
        for token in sentence_truncated:
            if token in word_to_idx:
                idx_sentence.append(word_to_idx[token])
            else:
                idx_sentence.append(constants.UNK_SYM_ID)
        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        idx_sentence.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(idx_sentence)))
        # add start padding
        idx_sentence = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + idx_sentence
        assert len(idx_sentence) == max_length + 2*constants.PADDING_SIZE
        idx_sentences.append(idx_sentence)
    return idx_sentences


def preprocess_sentences(sentences, word_to_idx, max_length, tokenizer):
    if tokenizer is None:
        return preprocess_sentences_without_bert(sentences, word_to_idx, max_length), [], []

    # Note that the BERT [CLS] token is at the index of the last prefix padding for the CNN
    original_lens = []
    truncated_lens = []
    num_truncated = 0

    idx_sentences = []
    bert_idx_sentences = []
    bert_input_masks = []

    idx_sentences_final_index = []  # the index of the final real token, which INCLUDES prefix padding
    bert_idx_sentences_final_sep_index = []  # final sep index
    for sentence in sentences:
        idx_sentence = []
        bert_sentence = []
        bert_input_mask = []
        original_lens.append(len(sentence))
        sentence_truncated = sentence[0:max_length]
        truncated_lens.append(len(sentence_truncated))

        if len(sentence_truncated) != len(sentence):
            num_truncated += 1
        bert_sentence.append("[CLS]")
        bert_input_mask.append(1)
        for token in sentence_truncated:
            bert_sentence.append(token)
            bert_input_mask.append(1)
            if token in word_to_idx:
                idx_sentence.append(word_to_idx[token])
            else:
                idx_sentence.append(constants.UNK_SYM_ID)
        bert_sentence.append("[SEP]")
        bert_input_mask.append(1)

        bert_idx_sentence = tokenizer.convert_tokens_to_ids(bert_sentence)

        final_index = len(idx_sentence) - 1 + constants.PADDING_SIZE  # account for prefix padding
        final_sep_index = len(bert_idx_sentence) - 1  # accounts for initial [CLS], since this is already concatenated
        idx_sentences_final_index.append(final_index)
        bert_idx_sentences_final_sep_index.append(final_sep_index)

        # add end padding; note that sentences always have at least constants.PADDING_SIZE trailing padding
        idx_sentence.extend([constants.PAD_SYM_ID] * (max_length + constants.PADDING_SIZE - len(idx_sentence)))
        # add start padding
        idx_sentence = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + idx_sentence

        # bert inputs are only padded at the end (to avoid complications with position embeddings with prefix padding)
        # The (max_length+2) is to account for [CLS] and [SEP]
        bert_idx_sentence.extend([0] * ((max_length+2) - len(bert_idx_sentence)))
        bert_input_mask.extend([0] * ((max_length+2) - len(bert_input_mask)))

        assert len(idx_sentence) == max_length + 2*constants.PADDING_SIZE
        assert len(idx_sentence) == len(bert_idx_sentence) + 2*constants.PADDING_SIZE - 2 and len(bert_idx_sentence) == len(bert_input_mask)
        idx_sentences.append(idx_sentence)
        bert_idx_sentences.append(bert_idx_sentence)
        bert_input_masks.append(bert_input_mask)

    print(f"Original lengths: mean: {np.mean(original_lens)}; std: {np.std(original_lens)}; "
          f"min: {np.min(original_lens)}, max: {np.max(original_lens)}")
    print(f"Truncated lengths: mean: {np.mean(truncated_lens)}; std: {np.std(truncated_lens)}; "
          f"min: {np.min(truncated_lens)}, max: {np.max(truncated_lens)}")
    print(f"Total number of sentences that were truncated: {num_truncated}")

    return idx_sentences, bert_idx_sentences, bert_input_masks, idx_sentences_final_index, bert_idx_sentences_final_sep_index


def get_vocab(train_x, max_vocab_size):
    vocab = {}
    word_to_idx = {}
    idx_to_word = {}

    raw_vocab = defaultdict(int)
    for sentence in train_x:
        for word in sentence:
            raw_vocab[word] += 1

    print(f"Raw vocabulary size: {len(raw_vocab)}")
    # the max_vocab_size most frequent tokens, sorted:
    vocab_sorted_by_most_frequent = sorted(raw_vocab.items(), key=lambda kv: kv[1], reverse=True)[0:max_vocab_size]

    # the padding symbol and unk symbol must be the first two entries (in that order)
    assert [constants.PAD_SYM_ID, constants.UNK_SYM_ID] == [0, 1]

    for idx, word in enumerate([constants.PAD_SYM, constants.UNK_SYM]):
        vocab[word] = -1
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    for i, (word, freq) in enumerate(vocab_sorted_by_most_frequent):
        idx = i+2 # account for pad, unk
        vocab[word] = freq
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    print(f"Filtered vocabulary size: {len(vocab)} (including padding and unk symbols)")
    assert len(vocab) == len(word_to_idx) and len(word_to_idx) == len(idx_to_word) and len(idx_to_word) <= max_vocab_size + 2
    return vocab, word_to_idx, idx_to_word


def save_vocab(vocab_file, vocab, word_to_idx):
    """
    Save the vocabulary

    :param vocab_file: output filename at which to save the vocabulary
    :param vocab: word -> frequency (or -1 for special symbols)
    :param word_to_idx: word -> index
    :return: None; save file with the following format:
        token\t\index\tfrequency\n
    """
    vocab_lines = []
    vocab_sorted_by_idx = sorted(word_to_idx.items(), key=lambda kv: kv[1], reverse=False)
    for word, idx in vocab_sorted_by_idx:
        vocab_lines.append(f"{word}\t{idx}\t{vocab[word]}\n")

    save_lines(vocab_file, vocab_lines)
    print(f"Vocabulary saved to {vocab_file}")


def load_vocab(vocab_file):
    """
    Load vocabulary. See save_vocab() for the appropriate format.

    :param vocab_file: input filename from which to load the vocabulary
    :return: vocab, word_to_idx, idx_to_word
    """
    print(f"Loading existing vocabulary from {vocab_file}")
    vocab = {}
    word_to_idx = {}
    idx_to_word = {}
    with codecs.open(vocab_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 3
            word = line[0]
            idx = int(line[1])
            freq = int(line[2])
            vocab[word] = freq
            word_to_idx[word] = idx
            idx_to_word[idx] = word

    assert word_to_idx[constants.PAD_SYM] == constants.PAD_SYM_ID and word_to_idx[constants.UNK_SYM] == constants.UNK_SYM_ID
    print(f"Loaded vocabulary size: {len(vocab)} (including padding and unk symbols)")
    return vocab, word_to_idx, idx_to_word


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def save_model_torch(model, params, epoch):
    path = f"{params['save_dir']}/{params['DATASET']}_{params['MODEL']}_{epoch}.pt"
    torch.save(model, path)
    print(f"A model was saved successfully as {path}")


def load_model_torch(filename, onto_gpu_id):

    try:
        if onto_gpu_id != -1:
            model = torch.load(filename, map_location=lambda storage, loc: storage.cuda(onto_gpu_id))
        else:
            model = torch.load(filename, map_location=lambda storage, loc: storage)
        print(f"Model in {filename} loaded successfully!")

        return model
    except:
        print(f"No available model such as {filename}.")


################################ for zero shot:

def get_target_labels_lines(filepath_with_name, convert_to_int=True):
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split()
            if convert_to_int:
                line = [int(x) for x in line]
            lines.append(line)
    return lines

################################ metric:

def get_document_title_from_wiki_sentence_string(wiki_title_reformatted_string):
    end_of_title_index = wiki_title_reformatted_string.find(", sentence ")
    assert end_of_title_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    start_of_title_index = wiki_title_reformatted_string.find("Evidence:")
    assert start_of_title_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    return wiki_title_reformatted_string[start_of_title_index+len("Evidence:")+1:end_of_title_index].strip()


def get_document_title_and_sent_index_from_wiki_sentence_string(wiki_title_reformatted_string):
    # We use this for assigning covered sentence id's to the true data since we are no longer loading the raw titles
    # Consider replacing this by preprocessing the true data with the covered indexes; unk titles are always wrong predictions
    end_of_title_index = wiki_title_reformatted_string.find(", sentence ")
    assert end_of_title_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    # Need to find the next colon in order to pull the sentence index
    end_of_sentence_colon_index = wiki_title_reformatted_string[end_of_title_index:].find(":")
    assert end_of_sentence_colon_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    end_of_sentence_colon_index += end_of_title_index
    start_of_title_index = wiki_title_reformatted_string.find("Evidence:")
    assert start_of_title_index != -1, f"ERROR: Unexpected formatting in {wiki_title_reformatted_string}"
    return wiki_title_reformatted_string[start_of_title_index+len("Evidence:")+1:end_of_sentence_colon_index].strip()


def read_covered_titles_from_file(filepath_with_name, tokenizer, eval_symmetric_data=False):
    start_time = time.time()
    print(f"Reading titles from file")

    unique_title_ids_to_document_ids = {}
    string_document_labels_to_document_ids = {}

    unique_string_titles_to_titles_id = {}
    unique_title_ids_to_unique_string_title = {}

    string_document_sentence_id_labels_to_token_ids = {}  # The keys uniquely describe the wiki sentences (but do not contain the text); only use for processing true data

    unique_titles = []
    unique_original_titles = []

    line_id = 0

    filtered_wiki_sents_to_wiki_ids = []
    # First pass is to collect the title, title_id pairs
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if line_id % 10000 == 0:
                print(f"Currently reading titles line {line_id}.")
            line_id += 1
            line = line.strip().split("\t")
            assert len(line) == 2
            wiki_title_reformatted_string = line[0].strip()
            assert wiki_title_reformatted_string != ""
            title_id = int(line[1].strip())
            assert wiki_title_reformatted_string not in unique_string_titles_to_titles_id
            unique_string_titles_to_titles_id[wiki_title_reformatted_string] = title_id

            assert title_id not in unique_title_ids_to_unique_string_title
            unique_title_ids_to_unique_string_title[title_id] = wiki_title_reformatted_string

            filtered_wiki_sents_to_wiki_ids.append((wiki_title_reformatted_string, title_id))

    # In second pass, we add the sorted titles to the data structures. We do this so that the title_id can be used
    # as the index.
    filtered_wiki_sents_to_wiki_ids = sorted(filtered_wiki_sents_to_wiki_ids, key=lambda x: x[1])
    for wiki_title_reformatted_string, title_id in filtered_wiki_sents_to_wiki_ids:
        wiki_title_reformatted = wiki_title_reformatted_string.split()
        title_tokenized = bert_tokenize(tokenizer, wiki_title_reformatted)
        # note that some rare tokens could be mapped together, for example, Chinese characters due to
        # BERT tokenization; here, they remain distinct because we use the strings as the keys
        assert len(unique_titles) == title_id, f"ERROR: The covered_sentences_dictionary_file has a discontinuous index." \
                                               f"Check formatting."
        unique_titles.append(title_tokenized)
        unique_original_titles.append(wiki_title_reformatted)

        # update document id structures:
        document_title = get_document_title_from_wiki_sentence_string(wiki_title_reformatted_string)
        if document_title not in string_document_labels_to_document_ids:
            string_document_labels_to_document_ids[document_title] = len(string_document_labels_to_document_ids)
        unique_title_ids_to_document_ids[unique_string_titles_to_titles_id[wiki_title_reformatted_string]] = \
            string_document_labels_to_document_ids[document_title]

        # update document + sent id structures (only used to recover id's from the true jsonl files)
        # TODO: consider removing in future versions by preprocessing the true data
        document_title_sentence_id_string = get_document_title_and_sent_index_from_wiki_sentence_string(wiki_title_reformatted_string)
        if not eval_symmetric_data:
            assert document_title_sentence_id_string not in string_document_sentence_id_labels_to_token_ids
        string_document_sentence_id_labels_to_token_ids[document_title_sentence_id_string] = title_id

    assert len(unique_string_titles_to_titles_id) == len(unique_titles)
    assert len(unique_titles) == len(unique_original_titles)
    assert len(unique_title_ids_to_unique_string_title) == len(unique_titles)
    print(f"Cumulative overall read_covered_titles_from_file() time: {(time.time() - start_time) / 60} minutes")
    return unique_titles, unique_original_titles, unique_string_titles_to_titles_id, \
           unique_title_ids_to_unique_string_title, unique_title_ids_to_document_ids, \
            string_document_sentence_id_labels_to_token_ids


def get_title_ids_from_title_sets_file(titles_sets_file, unique_title_ids_to_unique_string_title):
    # used for covered sentences
    claims_to_title_ids_sets = []
    claims_to_title_ids_tensors = []
    with codecs.open(titles_sets_file, encoding="utf-8") as f:
        for line_i, line in enumerate(f):
            line = line.strip().split("\t")
            title_ids_set = set()
            if len(line) == 0:
                title_ids_set.add(constants.UNK_TITLE_ID)
            else:
                for title_id in line:
                    try:
                        title_id = int(title_id)
                        assert title_id in unique_title_ids_to_unique_string_title, \
                            f"ERROR: {title_id} is missing in the dictionary."
                        assert title_id != constants.UNK_TITLE_ID, f"ERROR: In the current version, all covered titles" \
                                                                   f"are expected to appear in the dictionary."
                        title_ids_set.add(title_id)
                    except:
                        print(f"WARNING: Unexpected input at line {line_i} in {titles_sets_file}; Possible blank line. Adding constants.UNK_TITLE_ID.")
                        title_ids_set.add(constants.UNK_TITLE_ID)

            claims_to_title_ids_sets.append(title_ids_set)
            tensor_ids = sorted(title_ids_set)
            # do not include UNK title id
            if len(tensor_ids) > 0 and tensor_ids[0] == constants.UNK_TITLE_ID:
                tensor_ids = tensor_ids[1:]
            claims_to_title_ids_tensors.append(torch.LongTensor(tensor_ids))
    return claims_to_title_ids_sets, claims_to_title_ids_tensors


def get_title_id_from_document_string_and_sentence_id(title_string, sent_id, string_document_sentence_id_labels_to_token_ids):
    document_sentence_string = f"{title_string}, sentence {sent_id}"
    if document_sentence_string in string_document_sentence_id_labels_to_token_ids:
        return string_document_sentence_id_labels_to_token_ids[document_sentence_string]
    return constants.UNK_TITLE_ID


def get_true_title_id_sets_from_true_title_sets_file(titles_sets_file, string_document_sentence_id_labels_to_token_ids):
    # each claim is matched to a list of set of title ids; when evaluating,
    # use set operations (intersection, etc.)
    # For unverifiable claims, this is always an empty list.
    claims_to_title_ids_evidence_sets = []
    claims_to_true_titles_ids = []  # just a flat list, disregarding evidence groups; used for convenience in analysis
    number_of_claims_with_no_fully_covered_true_evidence_sets = 0
    number_of_unverifiable_claims = 0
    with codecs.open(titles_sets_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            evidence_sets = json.loads(line)
            true_evidence_sets = []
            all_title_ids = []
            if len(evidence_sets) == 0:
                number_of_unverifiable_claims += 1
            else:
                no_fully_covered_true_evidence_sets = True
                # a verifiable claim may have multiple evidence sets
                for wiki_urls_sent_id_pairs in evidence_sets:
                    title_ids_set = set()
                    # 1 full set of evidence may consist of multiple wikipedia articles
                    for wiki_urls_sent_id_pair in wiki_urls_sent_id_pairs:
                        title_id = get_title_id_from_document_string_and_sentence_id(wiki_urls_sent_id_pair[0],
                                                                                     wiki_urls_sent_id_pair[1],
                                                                          string_document_sentence_id_labels_to_token_ids)
                        title_ids_set.add(title_id)
                        all_title_ids.append(title_id)
                    if no_fully_covered_true_evidence_sets and constants.UNK_TITLE_ID not in title_ids_set:
                        no_fully_covered_true_evidence_sets = False
                    true_evidence_sets.append(title_ids_set)

                if no_fully_covered_true_evidence_sets:
                    number_of_claims_with_no_fully_covered_true_evidence_sets += 1
            claims_to_title_ids_evidence_sets.append(true_evidence_sets)
            claims_to_true_titles_ids.append(all_title_ids)
    print(f"Number of verifiable claims with no fully covered true evidence sets (in these cases, a complete "
          f"evidence set can never be completely correctly predicted/retrieved, since at least one Wiki sentence "
          f"is not in the set of covered sentences): "
          f"{number_of_claims_with_no_fully_covered_true_evidence_sets}")
    print(f"Number of unverifiable claims (as determined by the true evidence jsonl file): "
          f"{number_of_unverifiable_claims}")
    return claims_to_title_ids_evidence_sets, claims_to_true_titles_ids


def get_title_ids_from_chosen_sentences_file(chosen_sentences_only_evidence_file, unique_title_ids_to_unique_string_title):
    # Note that these are NOT sorted, as the first piece of evidence is preferred (i.e., due to the pre-processing, it
    # is the 'chosen' piece of evidence to use for training)
    # Chosen ids are ALWAYS tuples of size 1 or 2, where the second element might be empty. The predicted must also be tuples
    # for comparisons when order matters; otherwise, use sets.
    claims_to_chosen_title_ids = []
    num_chosen_sentences_with_at_least_one_uncovered_id = 0
    total_admitted_chosen_sentences = 0  # claims associated with non-unk evidence (used for estimating training optimization runs)
    total_admitted_chosen_evidence = 0
    total_unverifiable_sentences = 0
    with codecs.open(chosen_sentences_only_evidence_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            title_ids = []
            if len(line) == 0:
                title_ids.append(constants.UNK_TITLE_ID)
            else:
                assert len(line) <= 2
                for title_id in line:
                    title_id = int(title_id)
                    if title_id not in unique_title_ids_to_unique_string_title:
                        if title_id == constants.UNK_TITLE_ID:
                            num_chosen_sentences_with_at_least_one_uncovered_id += 1
                        else:
                            assert title_id == constants.UNVERIFIABLE_TITLE_ID
                    title_ids.append(title_id)
            claims_to_chosen_title_ids.append(tuple(title_ids))
            if constants.UNK_TITLE_ID not in title_ids and constants.UNVERIFIABLE_TITLE_ID not in title_ids:
                total_admitted_chosen_sentences += 1
                total_admitted_chosen_evidence += len(title_ids)
            if constants.UNVERIFIABLE_TITLE_ID in title_ids:
                total_unverifiable_sentences += 1
    print(f"Number of chosen sentences with at least one uncovered title: {num_chosen_sentences_with_at_least_one_uncovered_id}")
    print(f"Total chosen claims with no UNK (nor unverifiable) titles: {total_admitted_chosen_sentences} out of {len(claims_to_chosen_title_ids)}")
    print(f"Total chosen evidence pieces: {total_admitted_chosen_evidence}")
    print(f"Total unverifiable claims: {total_unverifiable_sentences}")
    return claims_to_chosen_title_ids, total_admitted_chosen_sentences, total_admitted_chosen_evidence, total_unverifiable_sentences


def get_title_data_structures(string_document_sentence_id_labels_to_token_ids, unique_title_ids_to_unique_string_title,
                              chosen_sentences_only_evidence_file, covered_titles_file, test_true_titles_file):

    claims_to_chosen_title_ids, total_admitted_chosen_sentences, total_admitted_chosen_evidence, total_unverifiable_sentences = get_title_ids_from_chosen_sentences_file(chosen_sentences_only_evidence_file,
                                                                       unique_title_ids_to_unique_string_title)

    claims_to_covered_titles_ids, claims_to_covered_titles_ids_tensors = \
        get_title_ids_from_title_sets_file(covered_titles_file, unique_title_ids_to_unique_string_title)

    claims_to_true_title_ids_evidence_sets, claims_to_true_titles_ids = \
        get_true_title_id_sets_from_true_title_sets_file(test_true_titles_file,
                                                         string_document_sentence_id_labels_to_token_ids)

    return claims_to_chosen_title_ids, total_admitted_chosen_sentences, total_admitted_chosen_evidence, \
           total_unverifiable_sentences, claims_to_covered_titles_ids, claims_to_covered_titles_ids_tensors, \
           claims_to_true_title_ids_evidence_sets, claims_to_true_titles_ids


def save_memory_structure_torch(memory_dir, file_identifier_prefix, memory_structure, chunk_id):

    path = f"{memory_dir}/{file_identifier_prefix}_memory_{chunk_id}.pt"
    torch.save(memory_structure, path)


def load_memory_structure_torch(memory_dir, file_identifier_prefix, chunk_id, onto_gpu_id):
    path = f"{memory_dir}/{file_identifier_prefix}_memory_{chunk_id}.pt"
    try:
        if onto_gpu_id != -1:
            memory_structure = torch.load(path, map_location=lambda storage, loc: storage.cuda(onto_gpu_id))
        else:
            memory_structure = torch.load(path, map_location=lambda storage, loc: storage)
        return memory_structure
    except:
        print(f"No available memory structure at {path}.")


def get_decision_labels_from_file(decision_labels_file):
    decision_labels = []
    with codecs.open(decision_labels_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")
            assert len(line) == 3
            label = line[1]
            if label == "SUPPORTS":
                decision_labels.append(constants.SUPPORTS_ID)
            elif label == "REFUTES":
                decision_labels.append(constants.REFUTES_ID)
            elif label == "UNVERIFIABLE":
                decision_labels.append(constants.MOREINFO_ID)
            else:
                assert False
    return decision_labels


def init_data_structure_for_split(data, split_mode, word_to_idx, max_length, tokenizer, covered_titles_file,
                                  true_titles_file, decision_labels_file,
                                  chosen_sentences_only_evidence_file, covered_sentences_dictionary_file, options):

    start_time = time.time()

    #data["eval_symmetric_data"] = options.eval_symmetric_data

    # The titles memory is based on the covered titles. This is valid for inference, since these are determined only
    # from the claims. This also means that a small number of true titles (~3%) can never be correctly predicted,
    # because they exist outside this subset. In the standard setup for training, claims with uncovered titles are
    # simply dropped, but for eval, we count them as wrong predictions.
    data[f"{split_mode}_unique_titles"], data[f"{split_mode}_unique_original_titles"], \
        unique_string_titles_to_titles_id, data[f"{split_mode}_unique_title_ids_to_unique_string_title"], \
        data[f"{split_mode}_unique_title_ids_to_document_ids"], string_document_sentence_id_labels_to_token_ids = \
        read_covered_titles_from_file(covered_sentences_dictionary_file, tokenizer,
                                      eval_symmetric_data=options.eval_symmetric_data)

    data[f"{split_mode}_idx_unique_titles"], data[f"{split_mode}_bert_idx_unique_titles"], \
    data[f"{split_mode}_bert_input_masks_unique_titles"], data[f"{split_mode}_idx_unique_titles_final_index"], \
    data[f"{split_mode}_bert_idx_unique_titles_final_sep_index"] = \
        preprocess_sentences(data[f"{split_mode}_unique_titles"], word_to_idx, max_length, tokenizer)

    # Tokenize the claims prefix labels, which are concatenated to the claims below
    data = init_claims_prefix_tokens(data, tokenizer)

    # The claims lines for all levels are added below, since the claims are constant and do not need to be dynamically updated.
    # Note, though, that in levels 2 and 3, the claim is added to the title for the searched instances.
    # Note that the max_length differs depending on the level (given that progressively more information is concatenated
    # to the title).
    for level_id in [1, 2, 3]:
        # for convenience; note that max length is changing with the level_id
        # We pre-pend a suffix for levels 2 and 3 to the claim to differentiate.
        if level_id == 1:
            prefix_tokens = []
        elif level_id == 2:
            prefix_tokens = data[f"claims_{constants.CLAIMS_CONSIDER_PREFIX_STRING}_wordpiece_tokens"]
        elif level_id == 3:
            prefix_tokens = data[f"claims_{constants.CLAIMS_PREDICT_PREFIX_STRING}_wordpiece_tokens"]

        data[f"level{level_id}_idx_{split_mode}_x"], \
        data[f"level{level_id}_{split_mode}_bert_idx_sentences"], \
        data[f"level{level_id}_{split_mode}_bert_input_masks"], \
        data[f"level{level_id}_{split_mode}_idx_x_final_index"], \
        data[f"level{level_id}_{split_mode}_bert_idx_final_sep_index"] = \
            preprocess_sentences([prefix_tokens + x for x in data[f"{split_mode}_x"]], word_to_idx, max_length * level_id, tokenizer)

        if level_id == 3:  # also add the gold reference to level 3:
            prefix_tokens = data[f"claims_{constants.CLAIMS_REF_PREFIX_STRING}_wordpiece_tokens"]
            data[f"reference_level{level_id}_idx_{split_mode}_x"], \
            data[f"reference_level{level_id}_{split_mode}_bert_idx_sentences"], \
            data[f"reference_level{level_id}_{split_mode}_bert_input_masks"], \
            data[f"reference_level{level_id}_{split_mode}_idx_x_final_index"], \
            data[f"reference_level{level_id}_{split_mode}_bert_idx_final_sep_index"] = \
                preprocess_sentences([prefix_tokens + x for x in data[f"{split_mode}_x"]], word_to_idx, max_length * level_id, tokenizer)
    # for level 1, the titles are the above unique_titles
    # for level 2 titles, we need to construct them from idx_unique_titles (appending consider AND the claims)
    # for level 3 titles, we also need to construct them from unique_titles. Note that we cannot use the following
    # ground-truth during training nor inference, since the truncation of the evidence may be subtly different, which
    # could introduce pathologies (since BERT can pick-up on very subtle distributional differences). As such, the
    # ground-truth final level is only used for convenience for constructing the vocab, but is otherwise discarded.
    # level_id = 3
    # data[f"level{level_id}_idx_{split_mode}_titles"], data[f"level{level_id}_{split_mode}_bert_idx_titles"], \
    # data[f"level{level_id}_{split_mode}_bert_input_masks_titles"], \
    # data[f"level{level_id}_{split_mode}_idx_titles_final_index"], \
    # data[f"level{level_id}_{split_mode}_bert_idx_titles_final_sep_index"] = \
    #     preprocess_sentences(data[f"{split_mode}_y"], word_to_idx, max_length*3, tokenizer)

    # note that now data[f"{split_mode}_claims_to_chosen_title_ids"] is a mapping to a list (of len 1 or 2); [-1] is unk
    # data[f"{split_mode}_claims_to_true_titles_ids"] is a mapping to a set of sorted tuples (of len up to largest evidence set in data)
    # For eval, sort the predicted tuples (or use set operations) since order does not matter in the eval
    # True document ids are similarly used for analysis purposes, but document ids are recovered via
    # data[f"{split_mode}_unique_title_ids_to_document_ids"] during inference, so there is not a separate structure here.

    # Note that data[f"level{level_id}_{split_mode}_claims_to_covered_titles_ids_tensors"] is prefixed by level_id. Level1 is
    # the base wiki sentences. In other levels, THESE MUST BE CONSTRUCTED FROM SEARCH.

    data[f"{split_mode}_claims_to_chosen_title_ids"], data[f"{split_mode}_total_admitted_chosen_sentences"], \
    data[f"{split_mode}_total_admitted_chosen_evidence"], \
    data[f"{split_mode}_total_unverifiable_sentences"], \
    data[f"{split_mode}_claims_to_covered_titles_ids"], \
    data[f"level{1}_{split_mode}_claims_to_covered_titles_ids_tensors"], \
    data[f"{split_mode}_claims_to_true_title_ids_evidence_sets"], data[f"{split_mode}_claims_to_true_titles_ids"] = \
        get_title_data_structures(string_document_sentence_id_labels_to_token_ids,
                                  data[f"{split_mode}_unique_title_ids_to_unique_string_title"],
                                  chosen_sentences_only_evidence_file, covered_titles_file, true_titles_file)

    # Note that we currently pull the true decision labels from the control file
    data[f"{split_mode}_decision_labels"] = get_decision_labels_from_file(decision_labels_file)
    assert len(data[f"level{1}_{split_mode}_claims_to_covered_titles_ids_tensors"]) == len(data[f"{split_mode}_decision_labels"])

    # Prepare the various prefix labels, which are concatenated, as applicable, during search
    data = init_prefix_structures(data, word_to_idx, tokenizer)

    # For convenience, we pre-calculate the 'chosen' titles for training for levels 1, 2, and 3
    if split_mode == "train":
        level_id = 1
        # note in level 1, titles are a list of lists
        data[f"chosen_level{level_id}_{split_mode}_idx_unique_titles"] = []
        data[f"chosen_level{level_id}_{split_mode}_bert_idx_unique_titles"] = []
        data[f"chosen_level{level_id}_{split_mode}_bert_input_masks_unique_titles"] = []
        for claim_index, _ in enumerate(data[f"level{level_id}_idx_{split_mode}_x"]):
            level1_idx_train_titles = []
            level1_train_bert_idx_titles = []
            level1_train_bert_input_masks_titles = []
            chosen_title_ids = data[f"{split_mode}_claims_to_chosen_title_ids"][claim_index]

            # note that for training, we exclude cases where at least 1 of the pieces of evidence is unk
            # or the claim is unverifiable
            if constants.UNK_TITLE_ID not in chosen_title_ids and constants.UNVERIFIABLE_TITLE_ID not in chosen_title_ids:
                for chosen_title_id in chosen_title_ids:
                    assert chosen_title_id != constants.UNK_TITLE_ID and chosen_title_id != constants.UNVERIFIABLE_TITLE_ID
                    level1_idx_train_titles.append(data["train_idx_unique_titles"][chosen_title_id])
                    level1_train_bert_idx_titles.append(data["train_bert_idx_unique_titles"][chosen_title_id])
                    level1_train_bert_input_masks_titles.append(
                        data["train_bert_input_masks_unique_titles"][chosen_title_id])
            data[f"chosen_level{level_id}_{split_mode}_idx_unique_titles"].append(level1_idx_train_titles)
            data[f"chosen_level{level_id}_{split_mode}_bert_idx_unique_titles"].append(level1_train_bert_idx_titles)
            data[f"chosen_level{level_id}_{split_mode}_bert_input_masks_unique_titles"].append(level1_train_bert_input_masks_titles)

        # now, construct the 'next' level, which in this case, is 2 and then 3
        # For unverifiable claims, we do not create a true reference (since the IR is unknown), but do create a hard
        # negative for the reference supports and refutes cases.
        for next_level_id in [2, 3]:
            data[f"chosen_level{next_level_id}_{split_mode}_idx_unique_titles"] = []
            data[f"chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"] = []
            data[f"chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"] = []
            for claim_index, decision_label in enumerate(data[f"{split_mode}_decision_labels"]):  # indexes every claim; in level 2, actual label is not used
                if next_level_id == 2:
                    label_string = constants.CONSIDER_STRING
                else:
                    if decision_label == constants.SUPPORTS_ID:
                        label_string = constants.SUPPORTS_STRING
                    elif decision_label == constants.REFUTES_ID:
                        label_string = constants.REFUTES_STRING
                    elif decision_label == constants.MOREINFO_ID:
                        label_string = constants.MOREINFO_STRING
                chosen_title_ids = data[f"{split_mode}_claims_to_chosen_title_ids"][claim_index]
                first_title_idx = chosen_title_ids[0]
                second_title_idx = chosen_title_ids[1] if len(chosen_title_ids) > 1 else None
                if first_title_idx == constants.UNK_TITLE_ID or (second_title_idx is not None and second_title_idx == constants.UNK_TITLE_ID) or \
                    first_title_idx == constants.UNVERIFIABLE_TITLE_ID:
                    # in this case, at least 1 of the chosen titles is not covered, so in this version, they will not
                    # be used for training, but we need to update the data structure with an empty list to maintain alignment
                    # Note that with the chosen instances, we maintain alignment between claims and titles, so no
                    # search is necessary. These MUST be shuffled together with the other training instances.
                    data[f"chosen_level{next_level_id}_{split_mode}_idx_unique_titles"].append([])
                    data[f"chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"].append([])
                    data[f"chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"].append([])
                else:
                    new_max_length = max_length * next_level_id
                    data = utils_search.construct_predicted_title_structures(new_max_length, label_string, split_mode,
                                                                             next_level_id,
                                                                             claim_index, first_title_idx, data,
                                                                             None, second_title_idx=second_title_idx,
                                                                             constuct_ground_truth=True,
                                                                             constuct_ground_truth_negative=False,
                                                                             second_negative=False)
        # for (currently only) level 3, we also include the opposite of the reference as a hard negative
        # for 2 classes, this is just the opposite class; for three class, we also construct a second hard negative
        for next_level_id in [3]:
            data[f"neg_chosen_level{next_level_id}_{split_mode}_idx_unique_titles"] = []
            data[f"neg_chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"] = []
            data[f"neg_chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"] = []

            # these are always ignored in training for 2 class; easier to just push through to simplify the shuffle
            data[f"neg2_chosen_level{next_level_id}_{split_mode}_idx_unique_titles"] = []
            data[f"neg2_chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"] = []
            data[f"neg2_chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"] = []

            for claim_index, decision_label in enumerate(data[f"{split_mode}_decision_labels"]):  # indexes every claim; in level 2, actual label is not used
                if next_level_id == 2:
                    assert False, f"ERROR: Hard negatives for level 2 are determined dynamically via search."
                    #label_string = constants.CONSIDER_STRING
                else:
                    # NOTE: Here, we flip the true labels (contrast with above)
                    if decision_label == constants.SUPPORTS_ID:
                        # this is the true reference label: label_string = constants.SUPPORTS_STRING
                        # false:
                        neg_label_string = constants.REFUTES_STRING
                        neg2_label_string = constants.MOREINFO_STRING
                    elif decision_label == constants.REFUTES_ID:
                        # this is the true reference label: label_string = constants.REFUTES_STRING
                        # false:
                        neg_label_string = constants.SUPPORTS_STRING
                        neg2_label_string = constants.MOREINFO_STRING
                    elif decision_label == constants.MOREINFO_ID:
                        neg_label_string = None
                        neg2_label_string = None
                    else:
                        assert False
                chosen_title_ids = data[f"{split_mode}_claims_to_chosen_title_ids"][claim_index]
                first_title_idx = chosen_title_ids[0]
                second_title_idx = chosen_title_ids[1] if len(chosen_title_ids) > 1 else None
                if first_title_idx == constants.UNK_TITLE_ID or (second_title_idx is not None and second_title_idx == constants.UNK_TITLE_ID) or \
                    first_title_idx == constants.UNVERIFIABLE_TITLE_ID:
                    if first_title_idx == constants.UNVERIFIABLE_TITLE_ID:
                        assert neg_label_string is None and neg2_label_string is None
                    # in this case, at least 1 of the chosen titles is not covered, so in this version, they will not
                    # be used for training, but we need to update the data structure with an empty list to maintain alignment
                    data[f"neg_chosen_level{next_level_id}_{split_mode}_idx_unique_titles"].append([])
                    data[f"neg_chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"].append([])
                    data[f"neg_chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"].append([])

                    data[f"neg2_chosen_level{next_level_id}_{split_mode}_idx_unique_titles"].append([])
                    data[f"neg2_chosen_level{next_level_id}_{split_mode}_bert_idx_unique_titles"].append([])
                    data[f"neg2_chosen_level{next_level_id}_{split_mode}_bert_input_masks_unique_titles"].append([])
                else:
                    assert neg_label_string is not None and neg2_label_string is not None
                    new_max_length = max_length * next_level_id
                    data = utils_search.construct_predicted_title_structures(new_max_length, neg_label_string, split_mode,
                                                                             next_level_id,
                                                                             claim_index, first_title_idx, data,
                                                                             None, second_title_idx=second_title_idx,
                                                                             constuct_ground_truth=False,
                                                                             constuct_ground_truth_negative=True,
                                                                             second_negative=False)
                    data = utils_search.construct_predicted_title_structures(new_max_length, neg2_label_string, split_mode,
                                                                             next_level_id,
                                                                             claim_index, first_title_idx, data,
                                                                             None, second_title_idx=second_title_idx,
                                                                             constuct_ground_truth=False,
                                                                             constuct_ground_truth_negative=True,
                                                                             second_negative=True)

    print(f"Cumulative overall init_data_structure_for_split() time: {(time.time() - start_time) / 60} minutes")
    return data


def init_prefix_tokens_for_vocab(len_of_train, tokenizer):
    # This creates an artificial list of WordPieces with all of the prefix strings to ensure that they are always
    # within the word2vec vocab
    # In future versions, add these as special symbols directly to vocab constructor.
    prefix_tokens = []
    for label_string in [constants.CLAIMS_CONSIDER_PREFIX_STRING, constants.CLAIMS_REF_PREFIX_STRING,
                         constants.CLAIMS_PREDICT_PREFIX_STRING, constants.SUPPORTS_STRING, constants.REFUTES_STRING,
                         constants.MOREINFO_STRING, constants.CONSIDER_STRING]:
        wordpieces = bert_tokenize(tokenizer, label_string.split())
        print(f"Prefix string: {label_string}; WordPieces: {wordpieces}")
        prefix_tokens.extend(wordpieces)
    return [prefix_tokens] * len_of_train


def init_claims_prefix_tokens(data, tokenizer):
    for label_string in [constants.CLAIMS_CONSIDER_PREFIX_STRING, constants.CLAIMS_REF_PREFIX_STRING,
                         constants.CLAIMS_PREDICT_PREFIX_STRING]:
        data[f"claims_{label_string}_wordpiece_tokens"] = bert_tokenize(tokenizer, label_string.split())
        print(f'claims prefix string: {label_string}; WordPieces: {data[f"claims_{label_string}_wordpiece_tokens"]}')
    return data


def init_prefix_structures(data, word_to_idx, tokenizer):
    for label_string in [constants.SUPPORTS_STRING, constants.REFUTES_STRING, constants.MOREINFO_STRING,
                         constants.CONSIDER_STRING]:
        data[f"{label_string}_idx_sentence"], data[f"{label_string}_bert_idx_sentence"], \
        data[f"{label_string}_bert_input_mask"] = preprocess_prefix(label_string, word_to_idx, tokenizer)
        print(f"TEMP: label_string: {label_string}")
        print(data[f"{label_string}_idx_sentence"])
        print(data[f"{label_string}_bert_idx_sentence"])
        print(data[f"{label_string}_bert_input_mask"])
    # final sep is also used when concatenating
    bert_sep_sym_bert_idx = tokenizer.convert_tokens_to_ids(["[SEP]"])
    assert len(bert_sep_sym_bert_idx) == 1, f"ERROR: In the current version, the sep. symbol is expected to be one token."
    data[f"bert_sep_sym_bert_idx"] = bert_sep_sym_bert_idx[0]

    return data


def preprocess_prefix(label_string, word_to_idx, tokenizer):
    """

    :param label_string:
    :param word_to_idx:
    :param tokenizer:
    :return:
        idx_sentence embedding ids of prefix WITH prefix padding
        bert_idx_sentence wordpiece ids of prefix WITH [CLS] and [SEP]
        bert_input_mask mask that tracks bert_idx_sentence
    """
    tokenized_label = bert_tokenize(tokenizer, label_string.split())
    idx_sentence = []
    bert_sentence = []
    bert_input_mask = []

    bert_sentence.append("[CLS]")
    bert_input_mask.append(1)
    for token in tokenized_label:
        bert_sentence.append(token)
        bert_input_mask.append(1)
        if token in word_to_idx:
            idx_sentence.append(word_to_idx[token])
        else:
            idx_sentence.append(constants.UNK_SYM_ID)
    # the final separator is always dropped, since these are always concatenated to the claims, so we do not include
    # it here:
    #bert_sentence.append("[SEP]")
    #bert_input_mask.append(1)

    bert_idx_sentence = tokenizer.convert_tokens_to_ids(bert_sentence)

    # add start padding
    idx_sentence = [constants.PAD_SYM_ID] * constants.PADDING_SIZE + idx_sentence

    return idx_sentence, bert_idx_sentence, bert_input_mask


def _check_top_k_value(top_k_nearest_memories, min_value):
    assert top_k_nearest_memories >= min_value, f"ERROR: In this version, at least k=={min_value} is needed to retrieve nearest " \
                                       f"wrong title in all cases."


def update_memory_parameters(data, options):
    data["titles_memory_dir"] = options.titles_memory_dir
    data["retrieval_memory_dir"] = options.retrieval_memory_dir

    data["level1_top_k_nearest_memories"] = options.level1_top_k_nearest_memories
    data["level2_top_k_nearest_memories"] = options.level2_top_k_nearest_memories
    data["level3_top_k_nearest_memories"] = options.level3_top_k_nearest_memories

    data["level3_top_k_stratifications"] = options.level3_top_k_stratifications
    data["level3_max_1_evidence_constructions"] = options.level3_max_1_evidence_constructions
    data["level3_max_2_evidence_constructions"] = options.level3_max_2_evidence_constructions

    data["do_not_marginalize_over_level3_evidence"] = options.do_not_marginalize_over_level3_evidence
    data["level3_top_k_evidence_predictions"] = options.level3_top_k_evidence_predictions

    if data["do_not_marginalize_over_level3_evidence"]:
        assert data["level3_max_1_evidence_constructions"] >= 1
    if data["level3_max_2_evidence_constructions"] == 0 and data["do_not_marginalize_over_level3_evidence"]:
        print(f"NOTE: --level3_max_2_evidence_constructions is 0, so --level3_top_k_stratifications is being ignored "
              f"and no evidence sets of size 2 will be considered among predictions in the beam of level 3.")
    for level_id in [1, 2, 3]:
        _check_top_k_value(data[f"level{level_id}_top_k_nearest_memories"], 3)

    if not data["do_not_marginalize_over_level3_evidence"]:
        print(f"Maginalizing over {data['level3_top_k_evidence_predictions']} pieces of predicted evidence "
              f"and ignoring --level3_top_k_stratifications, "
              f"--level3_max_1_evidence_constructions, --level3_max_2_evidence_constructions")

    data["level1_memory_batch_size"] = options.level1_memory_batch_size
    data["level1_retrieval_batch_size"] = options.level1_retrieval_batch_size
    data["level1_titles_chunk_size"] = options.level1_titles_chunk_size
    data["level1_retrieval_chunk_size"] = options.level1_retrieval_chunk_size

    data["level2_memory_batch_size"] = options.level2_memory_batch_size
    data["level2_retrieval_batch_size"] = options.level2_retrieval_batch_size
    data["level2_titles_chunk_size"] = options.level2_titles_chunk_size
    data["level2_retrieval_chunk_size"] = options.level2_retrieval_chunk_size

    data["level3_memory_batch_size"] = options.level3_memory_batch_size
    data["level3_retrieval_batch_size"] = options.level3_retrieval_batch_size
    data["level3_titles_chunk_size"] = options.level3_titles_chunk_size
    data["level3_retrieval_chunk_size"] = options.level3_retrieval_chunk_size

    data["only_levels_1_and_2"] = options.only_levels_1_and_2
    data["only_level_1"] = options.only_level_1
    assert not (data["only_levels_1_and_2"] and data["only_level_1"])

    data["init_level_2_with_level_1_weights"] = options.init_level_2_with_level_1_weights
    data["init_level_3_with_level_2_weights"] = options.init_level_3_with_level_2_weights

    data["continue_training"] = options.continue_training

    if data["continue_training"] and not options.load_ft_bert:
        print(f"WARNING: --continue_training is provided but not --load_ft_bert. Training is continuing from the "
              f"saved CNN memory filters, but a stock BERT model is otherwise being used. Typically, this only "
              f"would make sense for the final ec level, because otherwise there would be a mismatch between "
              f"the CNN parameters and BERT.")

    data["only_level_3"] = options.only_level_3
    if data["only_level_3"]:
        assert data["continue_training"] and options.load_ft_bert, f"ERROR: Training only level 3 requires an existing pre-trained level 2 model."
        assert not data["do_not_marginalize_over_level3_evidence"], f"ERROR: Currently, the data structure shuffling" \
                                                                f"when holding levels 1 and 2 constant assumes" \
                                                                f"margainlization over level 3 evidence."
    #data["eval_covered_reference"] = options.eval_dev_reference
    data["save_output_for_ec"] = options.save_output_for_ec

    data["eval_constrained"] = options.eval_constrained

    data["level2_constrained_mean"] = options.level2_constrained_mean
    data["level2_constrained_std"] = options.level2_constrained_std

    data["level3_constrained_mean"] = options.level3_constrained_mean
    data["level3_constrained_std"] = options.level3_constrained_std

    data["only_ec"] = options.only_ec
    if data["only_ec"]:
        assert data["continue_training"]
    data["ec_model_suffix"] = options.ec_model_suffix

    # for ExA:
    data["create_exemplar_database"] = options.create_exemplar_database
    data[f"exemplar_chunk_size"] = options.exemplar_chunk_size
    data[f"exemplar_memory_batch_size"] = options.exemplar_memory_batch_size
    data[f"exemplar_database_memory_dir"] = options.exemplar_database_memory_dir
    data[f"exemplar_query_memory_dir"] = options.exemplar_query_memory_dir

    data["create_exemplar_query"] = options.create_exemplar_query
    data["save_exemplar_output"] = options.save_exemplar_output
    data["add_full_beam_to_exemplar_database"] = options.add_full_beam_to_exemplar_database

    data["constrain_to_2_class_at_inference"] = options.constrain_to_2_class_at_inference

    data["exemplar_match_type"] = options.exemplar_match_type

    data["eval_symmetric_data"] = options.eval_symmetric_data
    data["visualize_alignment"] = options.visualize_alignment

    return data
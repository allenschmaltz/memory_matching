from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
import utils_search

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

def get_bert_representations_ft(train_bert_idx_sentences, train_bert_input_masks, bert_model, bert_device, bert_layers, total_length):
    """
    Get BERT output representations and reshape them to match the padding of the CNN output (in preparation for concat)
    :param train_bert_idx_sentences:
    :param train_bert_input_masks:
    :param bert_model:
    :param bert_device:
    :param bert_layers:
    :param total_length:
    :return:
    """
    train_bert_idx_sentences = torch.tensor(train_bert_idx_sentences, dtype=torch.long).to(bert_device)
    train_bert_input_masks = torch.tensor(train_bert_input_masks, dtype=torch.long).to(bert_device)
    all_encoder_layers, _ = bert_model(train_bert_idx_sentences, token_type_ids=None, attention_mask=train_bert_input_masks)
    top_layers = []
    for one_layer_index in bert_layers:
        top_layers.append(all_encoder_layers[one_layer_index])
    layer_output = torch.cat(top_layers, 2)

    # zero trailing padding
    masks = torch.unsqueeze(train_bert_input_masks, 2).float()
    layer_output = layer_output*masks

    bert_output_reshaped = torch.zeros((layer_output.shape[0], total_length, layer_output.shape[2])).to(bert_device)
    bert_output_reshaped[:, constants.PADDING_SIZE-1:(constants.PADDING_SIZE-1)+layer_output.shape[1], :] = layer_output

    return bert_output_reshaped


def shuffle_training_data_structures(data, train_debug_indexes, np_random_state):
    #  ALL SUBSEQUENTLY USED TRAINING STRUCTURES MUST BE SHUFFLED TOGETHER HERE
    # data[f"level{2 and 3}_train_claims_to_covered_titles_ids_tensors"] are created after the shuffle, so currently
    # they cannot be frozen across epochs; alternatively re-shuffle them using train_debug_indexes
    # Note that the base unique titles are not shuffled (and should never be shuffled), since they are always
    # indexed via mappings to the claims.
    if data["only_level_1"]:
        (train_debug_indexes,
         data[f"level{1}_idx_train_x"],
         data[f"level{1}_train_bert_idx_sentences"],
         data[f"level{1}_train_bert_input_masks"],
         data[f"level{1}_train_idx_x_final_index"],
         data[f"level{1}_train_bert_idx_final_sep_index"],
         data[f"chosen_level{1}_train_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
         data[f"train_claims_to_chosen_title_ids"],
         data[f"train_claims_to_covered_titles_ids"],
         data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
         data[f"train_claims_to_true_title_ids_evidence_sets"],
         data[f"train_claims_to_true_titles_ids"],
         data[f"train_decision_labels"]) = \
            shuffle(train_debug_indexes,
                    data[f"level{1}_idx_train_x"],
                    data[f"level{1}_train_bert_idx_sentences"],
                    data[f"level{1}_train_bert_input_masks"],
                    data[f"level{1}_train_idx_x_final_index"],
                    data[f"level{1}_train_bert_idx_final_sep_index"],
                    data[f"chosen_level{1}_train_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
                    data[f"train_claims_to_chosen_title_ids"],
                    data[f"train_claims_to_covered_titles_ids"],
                    data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
                    data[f"train_claims_to_true_title_ids_evidence_sets"],
                    data[f"train_claims_to_true_titles_ids"],
                    data[f"train_decision_labels"], random_state=np_random_state)
        #return data, train_debug_indexes

    elif data["only_levels_1_and_2"]:
        (train_debug_indexes,
         data[f"level{1}_idx_train_x"],
         data[f"level{1}_train_bert_idx_sentences"],
         data[f"level{1}_train_bert_input_masks"],
         data[f"level{1}_train_idx_x_final_index"],
         data[f"level{1}_train_bert_idx_final_sep_index"],
         data[f"level{2}_idx_train_x"],
         data[f"level{2}_train_bert_idx_sentences"],
         data[f"level{2}_train_bert_input_masks"],
         data[f"level{2}_train_idx_x_final_index"],
         data[f"level{2}_train_bert_idx_final_sep_index"],
         data[f"chosen_level{1}_train_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
         data[f"chosen_level{2}_train_idx_unique_titles"],
         data[f"chosen_level{2}_train_bert_idx_unique_titles"],
         data[f"chosen_level{2}_train_bert_input_masks_unique_titles"],
         data[f"train_claims_to_chosen_title_ids"],
         data[f"train_claims_to_covered_titles_ids"],
         data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
         data[f"train_claims_to_true_title_ids_evidence_sets"],
         data[f"train_claims_to_true_titles_ids"],
         data[f"train_decision_labels"]) = \
            shuffle(train_debug_indexes,
                    data[f"level{1}_idx_train_x"],
                    data[f"level{1}_train_bert_idx_sentences"],
                    data[f"level{1}_train_bert_input_masks"],
                    data[f"level{1}_train_idx_x_final_index"],
                    data[f"level{1}_train_bert_idx_final_sep_index"],
                    data[f"level{2}_idx_train_x"],
                    data[f"level{2}_train_bert_idx_sentences"],
                    data[f"level{2}_train_bert_input_masks"],
                    data[f"level{2}_train_idx_x_final_index"],
                    data[f"level{2}_train_bert_idx_final_sep_index"],
                    data[f"chosen_level{1}_train_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
                    data[f"chosen_level{2}_train_idx_unique_titles"],
                    data[f"chosen_level{2}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{2}_train_bert_input_masks_unique_titles"],
                    data[f"train_claims_to_chosen_title_ids"],
                    data[f"train_claims_to_covered_titles_ids"],
                    data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
                    data[f"train_claims_to_true_title_ids_evidence_sets"],
                    data[f"train_claims_to_true_titles_ids"],
                    data[f"train_decision_labels"], random_state=np_random_state)
        #return data, train_debug_indexes
    else:
        (train_debug_indexes,
         data[f"level{1}_idx_train_x"],
         data[f"level{1}_train_bert_idx_sentences"],
         data[f"level{1}_train_bert_input_masks"],
         data[f"level{1}_train_idx_x_final_index"],
         data[f"level{1}_train_bert_idx_final_sep_index"],
         data[f"level{2}_idx_train_x"],
         data[f"level{2}_train_bert_idx_sentences"],
         data[f"level{2}_train_bert_input_masks"],
         data[f"level{2}_train_idx_x_final_index"],
         data[f"level{2}_train_bert_idx_final_sep_index"],
         data[f"level{3}_idx_train_x"],
         data[f"level{3}_train_bert_idx_sentences"],
         data[f"level{3}_train_bert_input_masks"],
         data[f"level{3}_train_idx_x_final_index"],
         data[f"level{3}_train_bert_idx_final_sep_index"],
         data[f"reference_level{3}_idx_train_x"],
         data[f"reference_level{3}_train_bert_idx_sentences"],
         data[f"reference_level{3}_train_bert_input_masks"],
         data[f"reference_level{3}_train_idx_x_final_index"],
         data[f"reference_level{3}_train_bert_idx_final_sep_index"],
         data[f"chosen_level{1}_train_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_idx_unique_titles"],
         data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
         data[f"chosen_level{2}_train_idx_unique_titles"],
         data[f"chosen_level{2}_train_bert_idx_unique_titles"],
         data[f"chosen_level{2}_train_bert_input_masks_unique_titles"],
         data[f"chosen_level{3}_train_idx_unique_titles"],
         data[f"chosen_level{3}_train_bert_idx_unique_titles"],
         data[f"chosen_level{3}_train_bert_input_masks_unique_titles"],
         data[f"neg_chosen_level{3}_train_idx_unique_titles"],
         data[f"neg_chosen_level{3}_train_bert_idx_unique_titles"],
         data[f"neg_chosen_level{3}_train_bert_input_masks_unique_titles"],
         data[f"neg2_chosen_level{3}_train_idx_unique_titles"],
         data[f"neg2_chosen_level{3}_train_bert_idx_unique_titles"],
         data[f"neg2_chosen_level{3}_train_bert_input_masks_unique_titles"],
         data[f"train_claims_to_chosen_title_ids"],
         data[f"train_claims_to_covered_titles_ids"],
         data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
         data[f"train_claims_to_true_title_ids_evidence_sets"],
         data[f"train_claims_to_true_titles_ids"],
         data[f"train_decision_labels"]) = \
            shuffle(train_debug_indexes,
                    data[f"level{1}_idx_train_x"],
                    data[f"level{1}_train_bert_idx_sentences"],
                    data[f"level{1}_train_bert_input_masks"],
                    data[f"level{1}_train_idx_x_final_index"],
                    data[f"level{1}_train_bert_idx_final_sep_index"],
                    data[f"level{2}_idx_train_x"],
                    data[f"level{2}_train_bert_idx_sentences"],
                    data[f"level{2}_train_bert_input_masks"],
                    data[f"level{2}_train_idx_x_final_index"],
                    data[f"level{2}_train_bert_idx_final_sep_index"],
                    data[f"level{3}_idx_train_x"],
                    data[f"level{3}_train_bert_idx_sentences"],
                    data[f"level{3}_train_bert_input_masks"],
                    data[f"level{3}_train_idx_x_final_index"],
                    data[f"level{3}_train_bert_idx_final_sep_index"],
                    data[f"reference_level{3}_idx_train_x"],
                    data[f"reference_level{3}_train_bert_idx_sentences"],
                    data[f"reference_level{3}_train_bert_input_masks"],
                    data[f"reference_level{3}_train_idx_x_final_index"],
                    data[f"reference_level{3}_train_bert_idx_final_sep_index"],
                    data[f"chosen_level{1}_train_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{1}_train_bert_input_masks_unique_titles"],
                    data[f"chosen_level{2}_train_idx_unique_titles"],
                    data[f"chosen_level{2}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{2}_train_bert_input_masks_unique_titles"],
                    data[f"chosen_level{3}_train_idx_unique_titles"],
                    data[f"chosen_level{3}_train_bert_idx_unique_titles"],
                    data[f"chosen_level{3}_train_bert_input_masks_unique_titles"],
                    data[f"neg_chosen_level{3}_train_idx_unique_titles"],
                    data[f"neg_chosen_level{3}_train_bert_idx_unique_titles"],
                    data[f"neg_chosen_level{3}_train_bert_input_masks_unique_titles"],
                    data[f"neg2_chosen_level{3}_train_idx_unique_titles"],
                    data[f"neg2_chosen_level{3}_train_bert_idx_unique_titles"],
                    data[f"neg2_chosen_level{3}_train_bert_input_masks_unique_titles"],
                    data[f"train_claims_to_chosen_title_ids"],
                    data[f"train_claims_to_covered_titles_ids"],
                    data[f"level{1}_train_claims_to_covered_titles_ids_tensors"],
                    data[f"train_claims_to_true_title_ids_evidence_sets"],
                    data[f"train_claims_to_true_titles_ids"],
                    data[f"train_decision_labels"], random_state=np_random_state)

    return data, train_debug_indexes


def update_grad_status_of_cnn_model(model, levels_to_consider, cnn_and_emb_requires_grad_bool=False, modify_ec=False, ec_requires_grad_bool=False):
    # Note that this is currently also used in utils_ft_only_ec
    for level_id in levels_to_consider:
        model.get_embedding(level_id).weight.requires_grad = cnn_and_emb_requires_grad_bool
        for i in range(len(model.FILTERS)):
            model.get_conv(level_id, i).weight.requires_grad = cnn_and_emb_requires_grad_bool
            model.get_conv(level_id, i).bias.requires_grad = cnn_and_emb_requires_grad_bool

    if modify_ec:
        level_id = "ec"
        model.get_embedding(level_id).weight.requires_grad = ec_requires_grad_bool
        for i in range(len(model.FILTERS)):
            model.get_conv(level_id, i).weight.requires_grad = ec_requires_grad_bool
            model.get_conv(level_id, i).bias.requires_grad = ec_requires_grad_bool
        model.fc.weight.requires_grad = ec_requires_grad_bool
        model.fc.bias.requires_grad = ec_requires_grad_bool
    return model


def init_weights_from_prev_layer_of_cnn_model(model, from_source_level=1, to_target_level=2):
    # In practice, we found that it was better to train all levels together, but this can be used in
    # conjunction with training one level at a time.
    assert from_source_level < to_target_level
    model.get_embedding(to_target_level).weight.data.copy_(model.get_embedding(from_source_level).weight.data)
    for i in range(len(model.FILTERS)):
        model.get_conv(to_target_level, i).weight.data.copy_(model.get_conv(from_source_level, i).weight.data)
        model.get_conv(to_target_level, i).bias.data.copy_(model.get_conv(from_source_level, i).bias.data)
    return model


def update_and_display_losses(cumulative_losses, all_epoch_cumulative_losses, label):
    print(f"Epoch average loss ({label}): {np.mean(cumulative_losses)}")
    all_epoch_cumulative_losses.extend(cumulative_losses)
    print(f"Average loss across all mini-batches (all epochs) ({label}): {np.mean(all_epoch_cumulative_losses)}")
    return all_epoch_cumulative_losses


def train_ft(data, params, np_random_state, bert_model, tokenizer, bert_device, only_save_best_models=False, model=None):
    # For the model in the the original memory match paper, we train with separate optimizers for the Transformer
    # and the memory layers, and iteratively freeze as noted in the paper.
    # When training ec (BLADE, etc.), BERT and the other level CNN's are always frozen.
    assert only_save_best_models

    start_train_time = time.time()

    # In the final version, we train all levels together. Allowing the option to train only some levels adds quite
    # a bit of complication. I may remove this option in the final version. In practice, training all levels together
    # yielded stronger results than training the lower levels first and continuing. Note that in the current version,
    # only fine-tuning level 3 is no longer a valid option, as I subsequently changed the handling of shuffling and
    # caching.
    levels_to_consider = [1, 2, 3]
    if data["only_level_1"]:
        print(f"Only training level 1")
        levels_to_consider = [1]
    elif data["only_levels_1_and_2"]:
        print(f"Only training levels 1 and 2")
        levels_to_consider = [1, 2]
    elif data["only_level_3"]:
        print(f"Only training level 3")
        levels_to_consider = [3]
        assert False, f"ERROR: fine-tuning only level 3 needs to be updated to use the current caching and shuffling " \
                      f"approach for levels 1 and 2."

    if model is None:
        if params["MODEL"] != "rand":
            # load word2vec
            print("loading word2vec...")
            word_vectors = KeyedVectors.load_word2vec_format(params['word_embeddings_file'], binary=not params["word_embeddings_file_in_plaintext"])

            wv_matrix = []

            # one for zero padding and one for unk
            wv_matrix.append(np.zeros(params["word_embedding_size"]).astype("float32"))
            wv_matrix.append(np.random.uniform(-0.01, 0.01, params["word_embedding_size"]).astype("float32"))

            for i in range(params["vocab_size"]-2):
                idx = i+2
                word = data["idx_to_word"][idx]
                if word in word_vectors.vocab:
                    wv_matrix.append(word_vectors.word_vec(word))
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, params["word_embedding_size"]).astype("float32"))

            wv_matrix = np.array(wv_matrix)
            params["WV_MATRIX"] = wv_matrix

        print("Initializing model")
        if params["GPU"] != -1:
            model = CNN(**params).cuda(params["GPU"])
        else:
            model = CNN(**params)
        print("Starting training")

    if data["init_level_2_with_level_1_weights"]:
        print(f"Initializing level 2 weights with level 1 weights.")
        model = init_weights_from_prev_layer_of_cnn_model(model, from_source_level=1, to_target_level=2)
    if data["init_level_3_with_level_2_weights"]:
        print(f"Initializing level 3 weights with level 2 weights.")
        model = init_weights_from_prev_layer_of_cnn_model(model, from_source_level=2, to_target_level=3)


    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    seq_labeling_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    pdist = nn.PairwiseDistance(p=2)

    # the BERT optimizer follows run_classifer.py in the transformers repo
    if params["fine_tune_bert"]:
        param_optimizer = list(bert_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        # len(data["idx_train_x"]*4: (claims + titles)_true + (claims + titles)_hard_negative
        # Each 'instance' is a claim and a title matched with a hard negative claim and title
        # For level 1, we have 1 hard negative for every claim + the number of true (which could consist of 1 or 2 titles)
        # note that data[f"train_total_admitted_chosen_sentences"] and data[f"train_total_admitted_chosen_evidence"]
        # only consider verifiable claims
        number_of_level1_instances = data[f"train_total_admitted_chosen_sentences"] * 2 + data[f"train_total_admitted_chosen_evidence"] * 2
        # In levels 2 and 3, evidence is all compressed into a single sequence, so we do not need to account for
        # varying numbers of evidence. The tricky thing to remember is that level 1 is a list to accomodate multiple
        # positive instances per claim.
        if not params["create_hard_negative_for_unverifiable_retrieval_in_level2"] or params["only_2_class"]:
            number_of_level2_instances = data[f"train_total_admitted_chosen_sentences"] * 4
        else:
            print(f"Creating a hard negative for unverifiable claims for level 2 from the top of the beam.")
            number_of_level2_instances = data[f"train_total_admitted_chosen_sentences"] * 4 + data[f"train_total_unverifiable_sentences"] * 2
        # In level 3, we have true and hard negatives for both the reference AND the predicted (claims + titles)
        if params["only_2_class"]:
            number_of_level3_instances = data[f"train_total_admitted_chosen_sentences"] * 4 + data[f"train_total_admitted_chosen_sentences"] * 4
        else:
            # with 3 class, in the current version, we consider: each claim has 2 hard negatives, and we also need to consider the unverifiable sentences
            """
            Reference:
                for verifiable:
                    Claim_true and corresponding title_true
                    Claim_false_other_class and corresponding title_false_other_class
                    Claim_false_unverifiable_class and corresponding title_false_unverifiable_class
                        == data[f"train_total_admitted_chosen_sentences"] * 6
                for unverifiable
                    Ignored (Note that we do not consider unverifiable sentences in these cases)
            Predicated
                for verifiable:
                    Claim_true and corresponding title_true
                    Claim_false_other_class and corresponding title_false_other_class
                    Claim_false_other(3d)_class_class and corresponding title_false_other(3d)_class_class
                        == data[f"train_total_admitted_chosen_sentences"] * 6
                for unverifiable            
                    Claim_true and corresponding title_true
                    Claim_false_other_class and corresponding title_false_other_class
                    Claim_false_other(3d)_class_class and corresponding title_false_other(3d)_class_class
                        == data[f"train_total_unverifiable_sentences"] * 6
            """
            number_of_level3_instances = data[f"train_total_admitted_chosen_sentences"] * 6 + data[
                f"train_total_admitted_chosen_sentences"] * 6 + data[f"train_total_unverifiable_sentences"] * 6

        if data["only_level_1"]:
            total_estimated_instances = number_of_level1_instances
        elif data["only_levels_1_and_2"]:
            total_estimated_instances = number_of_level1_instances + number_of_level2_instances
        elif data["only_level_3"]:
            total_estimated_instances = number_of_level3_instances
        else:
            total_estimated_instances = number_of_level1_instances + number_of_level2_instances + number_of_level3_instances

        print(f"Total estimated number of training instances per epoch: {total_estimated_instances}")
        num_train_optimization_steps = int(total_estimated_instances / params["BATCH_SIZE"]) * params[
            "bert_num_train_epochs"]
        bert_optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=params["bert_learning_rate"],
                                 warmup=params["bert_warmup_proportion"],
                                 t_total=num_train_optimization_steps)


    pre_dev_acc = 0
    max_dev_acc = 0
    max_dev_acc_epoch = -1

    max_dev_decision_acc = 0
    max_dev_decision_acc_epoch = -1

    total_filters = sum(params["FILTER_NUM"])  # This assumes the same number of filters across all CNNs.

    recorded_nvidia_smi = False
    recorded_nvidia_smi_for_bert = False

    freeze_bert_this_epoch = True
    total_bert_fine_tuning_epochs = 0

    all_epoch_cumulative_losses = []
    all_epoch_cumulative_losses_bert = []
    all_epoch_cumulative_losses_cnn = []
    #all_epoch_cumulative_losses_fc = []

    train_debug_indexes = [x for x in range(len(data[f"level{1}_idx_train_x"]))]

    # Initialize by setting all CNN's to not update. These are updated in turn, below.
    model = update_grad_status_of_cnn_model(model, [1, 2, 3], cnn_and_emb_requires_grad_bool=False, modify_ec=True,
                                    ec_requires_grad_bool=False)

    for e in range(params["EPOCH"]):
        if freeze_bert_this_epoch:
            print(f"TRAIN-TYPE: Freezing BERT, training CNN")
        else:
            print(f"TRAIN-TYPE: Freezing CNN, training BERT")

        # data[f"level{2 and 3}_train_claims_to_covered_titles_ids_tensors"] are created after the shuffle, so currently
        # they cannot be frozen across epochs; alternatively re-shuffle them using train_debug_indexes
        # Note that the base unique titles are not shuffled (and should never be shuffled), since they are always
        # indexed via mappings to the claims.
        if e == 0 or not data["only_level_3"]:
            data, train_debug_indexes = shuffle_training_data_structures(data, train_debug_indexes, np_random_state)
        else:
            assert False, f"ERROR: Only training level 3 is currently disabled. Update the shuffling convention first."

        if e == 0 and params["main_device"].type != "cpu":
            print(
                f'prior to first call to get_title_memory torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')
        if e == 0 and data["only_level_3"]:
            predicted_output = {}
            # cache level 1 and 2 dev memories to disk before updating BERT parameters
            for level_id in [1, 2]:
                utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title",
                                                      split_mode="dev", level_id=level_id)
                utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="retrieve",
                                                      split_mode="dev", level_id=level_id)
                data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist,
                                                                                                    data, model, params,
                                                                                                    save_eval_output=True,
                                                                                                    mode="dev",
                                                                                                    level_id=level_id)

        # train and dev title memories are now distinct structures, so they must be updated separately before each eval
        predicted_output = {}
        # above we can just ignore subsequent higher levels in utils_search if only training lower levels,
        # but here we have to be careful about not over-writing the lower levels if only training the top level
        if e == 0 and data["only_level_3"]:
            train_levels_to_consider = [1, 2, 3]
        else:
            train_levels_to_consider = levels_to_consider
        # coarse-to-fine search:
        for level_id in train_levels_to_consider:  # initialize all levels
            utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title", split_mode="train", level_id=level_id)
            utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="retrieve", split_mode="train", level_id=level_id)
            data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist, data, model, params,
                                                          save_eval_output=False, mode="train", level_id=level_id)

        if e == 0 and params["main_device"].type != "cpu":
            print(
                f'after first call to get_title_memory torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')

        if e == 0 and params["main_device"].type != "cpu":
            print(
                f'after first call to get_nearest_titles_from_memory torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')

        training_mini_batches = {}
        for level_id in levels_to_consider:
            # all of these are the same length
            training_mini_batches[f"level{level_id}_augmented_idx_train_x"], \
            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"], \
            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"], \
            training_mini_batches[f"level{level_id}_augmented_idx_train_titles"], \
            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"], \
            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"], \
            training_mini_batches[f"level{level_id}_autoencode_min_max_y"] = [], [], [], [], [], [], []
            # unlike the above, the following has the invariant of always equaling the number of claims; use this to
            # index into mini-batches using the above
            training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"] = []

        # To keep things simple in this version, we simply duplicate claims, when applicable, to match negative titles
        num_correctly_matched = 0
        num_with_true_covered = 0
        unk_title_id_warnings = 0

        # each level should have the same number of claims:
        total_number_of_claims = len(data[f"level{1}_idx_{'train'}_x"])
        for level_id in levels_to_consider:
            assert len(predicted_output[f"level{level_id}_{'train'}_nearest_wrong_level_title_ids"]) == total_number_of_claims
            assert len(data[f"level{level_id}_idx_{'train'}_x"]) == total_number_of_claims
            assert len(data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"]) == total_number_of_claims

        # Now, we iterate through all claims, using the results from the coarse-to-fine search to determine which
        # sequences will participate in this epoch
        for claim_index in range(total_number_of_claims):
            level_id = 1
            if data[f"{'train'}_decision_labels"][claim_index] != constants.MOREINFO_ID:
                chosen_title_id_tuples = data[f"{'train'}_claims_to_chosen_title_ids"][claim_index]
                if constants.UNK_TITLE_ID not in chosen_title_id_tuples:
                    assert constants.UNVERIFIABLE_TITLE_ID not in chosen_title_id_tuples
                    # add gold reference
                    # the loop in this case is to handle multiple reference wiki sentences (again, recall level 1 has
                    # an inner list, unlike levels 2 and 3)
                    for _ in data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index]:

                        training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                            data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                            data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                            data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])

                        # note the 0 -- we aim to minimize the distance to the reference
                        training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([0] * total_filters)

                    assert len(data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index]) == len(chosen_title_id_tuples)
                    # note the 'extend': level1 is a list of lists
                    training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].extend(
                        data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index])
                    training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].extend(
                        data[f"chosen_level{level_id}_{'train'}_bert_idx_unique_titles"][claim_index])
                    training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].extend(
                        data[f"chosen_level{level_id}_{'train'}_bert_input_masks_unique_titles"][claim_index])

                    # Note that we currently have 1 hard negative *per claim* -- i.e., if the claim has 2 ground-truth
                    # wiki sentences, there will still only be 1 hard negative.
                    nearest_wrong_relative_to_level_title_id = \
                    predicted_output[f"level{level_id}_{'train'}_nearest_wrong_level_title_ids"][claim_index]
                    if nearest_wrong_relative_to_level_title_id != constants.UNK_TITLE_ID:
                        # claim are the same as above
                        training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                            data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                            data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                            data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])
                        # title is now the predicted wrong title
                        training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                            data[f"{'train'}_idx_unique_titles"][
                                nearest_wrong_relative_to_level_title_id])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                            data[f"{'train'}_bert_idx_unique_titles"][
                                nearest_wrong_relative_to_level_title_id])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                            data[f"{'train'}_bert_input_masks_unique_titles"][
                                nearest_wrong_relative_to_level_title_id])

                        # note the 1 -- we aim to maximize the distance to the hard negative
                        training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([1] * total_filters)
            # be careful about unk cases--need to subsequently ignore null cases:
            training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"].append(len(training_mini_batches[f"level{level_id}_augmented_idx_train_x"]))

            remaining_levels_to_consider = [x for x in levels_to_consider if x != 1]
            # The following gets skipped if only level 1
            for level_id in remaining_levels_to_consider:  #for level_id in [2, 3]:
                chosen_title_id_tuples = data[f"{'train'}_claims_to_chosen_title_ids"][claim_index]
                if constants.UNK_TITLE_ID not in chosen_title_id_tuples:
                    # add gold reference
                    if level_id == 2 and data[f"{'train'}_decision_labels"][claim_index] != constants.MOREINFO_ID:
                        training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(data[f"chosen_level{level_id}_{'train'}_bert_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(data[f"chosen_level{level_id}_{'train'}_bert_input_masks_unique_titles"][claim_index])
                        # note the 0 -- we aim to minimize the distance to the reference
                        training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([0] * total_filters)

                    if level_id == 3 and data[f"{'train'}_decision_labels"][claim_index] != constants.MOREINFO_ID:
                        num_with_true_covered += 1
                        # Note that the claims string (left-side) is different than with the predicted
                        training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                            data[f"reference_level{level_id}_idx_{'train'}_x"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                            data[f"reference_level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                            data[f"reference_level{level_id}_{'train'}_bert_input_masks"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                            data[f"chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                            data[f"chosen_level{level_id}_{'train'}_bert_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                            data[f"chosen_level{level_id}_{'train'}_bert_input_masks_unique_titles"][claim_index])
                        # note the 0 -- we aim to minimize the distance to the reference
                        training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([0] * total_filters)

                        # also add 'true' negative (i.e., flipping the ground-truth binary label) for level 3
                        training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                            data[f"reference_level{level_id}_idx_{'train'}_x"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                            data[f"reference_level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                            data[f"reference_level{level_id}_{'train'}_bert_input_masks"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                            data[f"neg_chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                            data[f"neg_chosen_level{level_id}_{'train'}_bert_idx_unique_titles"][claim_index])
                        training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                            data[f"neg_chosen_level{level_id}_{'train'}_bert_input_masks_unique_titles"][claim_index])
                        # note the 1 -- we aim to maximize the distance to the artificial negative reference
                        training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([1] * total_filters)

                        if not params["only_2_class"]:  # also add 2nd negative (i.e., the unverifiable case)
                            training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                                data[f"reference_level{level_id}_idx_{'train'}_x"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                                data[f"reference_level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                                data[f"reference_level{level_id}_{'train'}_bert_input_masks"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                                data[f"neg2_chosen_level{level_id}_{'train'}_idx_unique_titles"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                                data[f"neg2_chosen_level{level_id}_{'train'}_bert_idx_unique_titles"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                                data[f"neg2_chosen_level{level_id}_{'train'}_bert_input_masks_unique_titles"][
                                    claim_index])
                            # note the 1 -- we aim to maximize the distance to the artificial negative reference
                            training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([1] * total_filters)

                    if (level_id == 2 and data[f"{'train'}_decision_labels"][claim_index] != constants.MOREINFO_ID) or \
                            (level_id == 2 and data[f"{'train'}_decision_labels"][claim_index] == constants.MOREINFO_ID
                             and params["create_hard_negative_for_unverifiable_retrieval_in_level2"]) \
                            or (level_id == 3):
                        nearest_wrong_relative_to_level_title_id = predicted_output[f"level{level_id}_{'train'}_nearest_wrong_level_title_ids"][claim_index]
                        if nearest_wrong_relative_to_level_title_id != constants.UNK_TITLE_ID:

                            # claim are the same as above
                            training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                                data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                                data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                                data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])
                            # title is now the predicted wrong title
                            training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_idx_unique_titles"][nearest_wrong_relative_to_level_title_id])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_bert_idx_unique_titles"][nearest_wrong_relative_to_level_title_id])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_bert_input_masks_unique_titles"][nearest_wrong_relative_to_level_title_id])

                            # note the 1 -- we aim to maximize the distance to the hard negative
                            training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([1] * total_filters)

                    if level_id == 3:  # also add the 'true' predicted for level 3 -- may want to make this an option, particularily for --do_not_marginalize_over_level3_evidence
                        nearest_true_level_title_ids = predicted_output[f"level{level_id}_{'train'}_nearest_true_level_title_ids"][claim_index]
                        if nearest_true_level_title_ids != constants.UNK_TITLE_ID:
                            training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                                data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                                data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                                data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])
                            # title is now the predicted titles at the top of the beam, which we 'marginalize'
                            training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_idx_unique_titles"][
                                    nearest_true_level_title_ids])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_bert_idx_unique_titles"][
                                    nearest_true_level_title_ids])
                            training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                                data[f"predicted_level{level_id}_{'train'}_bert_input_masks_unique_titles"][
                                    nearest_true_level_title_ids])
                            # note the 0 -- we aim to minimize the distance to the predicted 'true'
                            training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([0] * total_filters)
                        if not params["only_2_class"]:  # also add 2nd predicted negative
                            nearest_neg2_relative_to_level_title_id = \
                            predicted_output[f"level{level_id}_{'train'}_nearest_neg2_level_title_ids"][claim_index]
                            if nearest_neg2_relative_to_level_title_id != constants.UNK_TITLE_ID:
                                # claim are the same as above
                                training_mini_batches[f"level{level_id}_augmented_idx_train_x"].append(
                                    data[f"level{level_id}_idx_{'train'}_x"][claim_index])
                                training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"].append(
                                    data[f"level{level_id}_{'train'}_bert_idx_sentences"][claim_index])
                                training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"].append(
                                    data[f"level{level_id}_{'train'}_bert_input_masks"][claim_index])
                                # title is now the predicted wrong title
                                training_mini_batches[f"level{level_id}_augmented_idx_train_titles"].append(
                                    data[f"predicted_level{level_id}_{'train'}_idx_unique_titles"][
                                        nearest_neg2_relative_to_level_title_id])
                                training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"].append(
                                    data[f"predicted_level{level_id}_{'train'}_bert_idx_unique_titles"][
                                        nearest_neg2_relative_to_level_title_id])
                                training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"].append(
                                    data[f"predicted_level{level_id}_{'train'}_bert_input_masks_unique_titles"][
                                        nearest_neg2_relative_to_level_title_id])

                                # note the 1 -- we aim to maximize the distance to the hard negative
                                training_mini_batches[f"level{level_id}_autoencode_min_max_y"].append([1] * total_filters)

                # be careful about unk cases--need to subsequently ignore null cases:
                training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"].append(len(training_mini_batches[f"level{level_id}_augmented_idx_train_x"]))

        total_sequences = 0
        for level_id in levels_to_consider:
            assert len(training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"]) == \
                   total_number_of_claims, \
                f"{level_id}, {len(training_mini_batches[f'level{level_id}_augmented_end_indexes_by_claim'])}, " \
                f"{total_number_of_claims}"
            print(f"level_id: {level_id}, "
                  f"training_mini_batches[f'level{level_id}_augmented_end_indexes_by_claim'][-1]: "
                  f"{training_mini_batches[f'level{level_id}_augmented_end_indexes_by_claim'][-1]}; "
                  f"total_sequences (running): {total_sequences}; total_number_of_claims: {total_number_of_claims}")
            total_sequences += training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"][-1]

        # Note that each 'instance' (for comparison to the ground-truth/loss) consists of 2 sequences:
        # For FEVER: a claim + a title. These two sequences are batched together, and then the difference
        # vector (constructed by the absolute value of the maxpool of the filters of the CNN for each of the
        # 2 sequences) is calculated after the forward. Note that each 'sequence' itself consists of 3
        # sequences: a vector of word embedding indexes; a vector of wordpiece indexes; and a vector of masks.
        #
        # Note that that level 1 ground-truth is a list of lists, since in that case, claims are (optionally)
        # allowed to be matched to multiple title sequences. For all levels, the hard 'negative' always
        # consists of a single 'instance' (if found).

        # In this version, we simply drop training instances for which the true title is not covered AND not
        # appearing in some other claim, BUT uncovered id's appearing in another claim are forwarded in this version

        print(f"++Epoch {e+1}: Number of (paired) instances considered: {total_sequences}")
        start_epoch_time = time.time()
        num_batch_instances = math.ceil((total_number_of_claims / params["BATCH_SIZE"]))
        batch_num = 0
        cumulative_losses = []
        cumulative_losses_bert = []
        cumulative_losses_cnn = []
        #cumulative_losses_fc = []

        # Looping through the batches is complicated by the desire to keep claims together, and that each level
        # has a separate top-layer CNN (and embeddings) and length.

        for i in range(0, total_number_of_claims, params["BATCH_SIZE"]):  # note that the batch size is in terms of the claims; actual batch pushed through network is 2x this number (claims+titles)
            if batch_num % max(1, int(num_batch_instances * 0.25)) == 0:  # max(1,x) is for the edge case of very small augmented sets to avoid modulo by zero
                print(f"Epoch {e+1}, {batch_num/num_batch_instances}")
                if len(cumulative_losses) > 0:
                    print(f"\tCurrent epoch average loss: {np.mean(cumulative_losses)}")
            batch_num += 1
            batch_range = min(params["BATCH_SIZE"], total_number_of_claims - i)
            # need to check for and ignore empty cases (edge case where all claims in the batch are unk -- should rarely
            # (if ever) occur in Fever with the current batch sizes)
            # construct batch
            num_levels_considered = 0
            loss = None
            for level_id in levels_to_consider:
                start_and_end_indexes = training_mini_batches[f"level{level_id}_augmented_end_indexes_by_claim"][i:i + batch_range]
                if len(start_and_end_indexes) > 1:
                    if i == 0:
                        start_index = 0
                    else:
                        start_index = start_and_end_indexes[0]  # indexes into the augmented structures
                    end_index = start_and_end_indexes[-1]
                    if start_index != end_index:
                        num_levels_considered += 1
                        batch_x = training_mini_batches[f"level{level_id}_augmented_idx_train_x"][start_index:end_index]
                        number_of_claims = len(batch_x)
                        batch_x.extend(training_mini_batches[f"level{level_id}_augmented_idx_train_titles"][start_index:end_index])

                        # get BERT representations
                        if freeze_bert_this_epoch:
                            if loss is None:  # only need to update first time for each mini-batch
                                model.train()
                                bert_model.eval()
                                model = update_grad_status_of_cnn_model(model, levels_to_consider,
                                                                        cnn_and_emb_requires_grad_bool=True,
                                                                        modify_ec=False,
                                                                        ec_requires_grad_bool=False)

                            bert_output = run_main.get_bert_representations(
                                training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"][start_index:end_index] + training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"][start_index:end_index],
                                training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"][start_index:end_index] + training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"][start_index:end_index], bert_model, bert_device,
                                params["bert_layers"], len(batch_x[0]))
                            bert_output = torch.FloatTensor(bert_output).to(params["main_device"])
                            if loss is None:  # only need to update first time for each mini-batch
                                optimizer.zero_grad()
                        else:
                            if loss is None:  # only need to update first time for each mini-batch
                                model.eval()
                                bert_model.train()
                                model = update_grad_status_of_cnn_model(model, levels_to_consider,
                                                                        cnn_and_emb_requires_grad_bool=False,
                                                                        modify_ec=False,
                                                                        ec_requires_grad_bool=False)
                                bert_optimizer.zero_grad()
                            bert_output = get_bert_representations_ft(
                                training_mini_batches[f"level{level_id}_augmented_train_bert_idx_sentences"][start_index:end_index] + training_mini_batches[f"level{level_id}_augmented_train_bert_idx_titles"][start_index:end_index],
                                training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks"][start_index:end_index] + training_mini_batches[f"level{level_id}_augmented_train_bert_input_masks_titles"][start_index:end_index], bert_model, bert_device,
                                params["bert_layers"], len(batch_x[0]))

                        batch_x = torch.LongTensor(batch_x).to(params["main_device"])
                        batch_autoencode_min_max_y = torch.FloatTensor(training_mini_batches[f"level{level_id}_autoencode_min_max_y"][start_index:end_index]).to(
                            params["main_device"])

                        total_length = params["max_length"] * level_id + 2 * constants.PADDING_SIZE
                        model_output = model(batch_x, bert_output, level_id=level_id, total_length=total_length,
                                             forward_type_description="sentence_representation", main_device=None)

                        abs_filter_diff = torch.abs(
                            model_output[0:number_of_claims] - model_output[number_of_claims:])
                        level_loss = seq_labeling_criterion(abs_filter_diff, batch_autoencode_min_max_y)
                        if loss is None:
                            loss = level_loss.mean()
                        else:
                            loss += level_loss.mean()

            if loss is not None:
                # average over levels:
                loss = loss / num_levels_considered
                if (freeze_bert_this_epoch and not recorded_nvidia_smi and params["main_device"].type != "cpu") or \
                    (not freeze_bert_this_epoch and not recorded_nvidia_smi_for_bert and params["main_device"].type != "cpu"):

                    print(f"+++++++++++++++Before seq backward: Freeze BERT: {freeze_bert_this_epoch}+++++++++++++++")
                    nvidia_smi_out = subprocess.run(["nvidia-smi"], capture_output=True)
                    print(f'{nvidia_smi_out.stdout.decode("utf-8")}')

                cumulative_losses.append(loss.item())
                if freeze_bert_this_epoch:
                    cumulative_losses_cnn.append(loss.item())
                else:
                    cumulative_losses_bert.append(loss.item())
                loss.backward()

                if freeze_bert_this_epoch:  # for CNN
                    nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
                    optimizer.step()
                else:
                    bert_optimizer.step()

                if (freeze_bert_this_epoch and not recorded_nvidia_smi and params["main_device"].type != "cpu") or \
                        (not freeze_bert_this_epoch and not recorded_nvidia_smi_for_bert and params[
                            "main_device"].type != "cpu"):
                    print(f"+++++++++++++++After seq backward: Freeze BERT: {freeze_bert_this_epoch}+++++++++++++++")
                    nvidia_smi_out = subprocess.run(["nvidia-smi"], capture_output=True)
                    print(f'{nvidia_smi_out.stdout.decode("utf-8")}')
                    if freeze_bert_this_epoch:
                        recorded_nvidia_smi = True
                    else:
                        recorded_nvidia_smi_for_bert = True
                    print(f'torch.cuda.memory_allocated: {torch.cuda.memory_allocated(params["main_device"])}')
                    print(f'torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')

        update_and_display_losses(cumulative_losses, all_epoch_cumulative_losses, "combined")
        if freeze_bert_this_epoch:
            update_and_display_losses(cumulative_losses_cnn, all_epoch_cumulative_losses_cnn, "CNN")
        else:
            update_and_display_losses(cumulative_losses_bert, all_epoch_cumulative_losses_bert, "BERT")
        #update_and_display_losses(cumulative_losses_fc, all_epoch_cumulative_losses_fc, "EC_FC")

        end_epoch_time = time.time()
        print(f"Time to complete epoch: {(end_epoch_time - start_epoch_time) / 60} minutes")
        print(f"Cumulative overall training time: {(time.time() - start_train_time) / 60} minutes")

        # These might be retained on the GPU, so set to None before eval call. (TODO: actually, appears not to matter for allocation in current version)
        model_output, batch_x, bert_output, abs_filter_diff, batch_autoencode_min_max_y, level_loss = None, None, None, None, None, None

        print(f"------------------EVAL (dev) STARTING------------------")

        if e == 0 and data["only_level_3"]:
            dev_levels_to_consider = [1, 2, 3]
        else:
            dev_levels_to_consider = levels_to_consider

        predicted_output = {}
        for level_id in dev_levels_to_consider:
            if data["only_level_3"] and level_id in [1, 2]:
                # BERT will have drifted, so the memories from levels 1 and 2 are read
                # from disk from before epoch 0
                data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output,
                                                                                                    pdist, data, model,
                                                                                                    params,
                                                                                                    save_eval_output=True,
                                                                                                    mode="dev",
                                                                                                    level_id=level_id)
            else:
                utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title", split_mode="dev", level_id=level_id)
                utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="retrieve", split_mode="dev", level_id=level_id)
                data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist, data, model, params,
                                                              save_eval_output=True, mode="dev", level_id=level_id)

        print(f"------------------EVAL (dev) COMPLETE------------------")

        if data["only_level_1"]:
            # currently just duplicate for level 1 to keep it simple
            dev_acc = predicted_output[f"level{1}_{'dev'}_retrieval_acc"]
            dev_decision_acc = predicted_output[f"level{1}_{'dev'}_retrieval_acc"]
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), retrieval dev_acc (level 1): {dev_acc}")
            score_1_label = "level1_retrieval"
        elif data["only_levels_1_and_2"]:
            # overriding dec to be level 2 acc
            dev_acc = predicted_output[f"level{1}_{'dev'}_retrieval_acc"]
            dev_decision_acc = predicted_output[f"level{2}_{'dev'}_retrieval_acc"]
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), retrieval dev_acc (level 1): {dev_acc}")
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), retrieval dev_acc (level 2): {dev_decision_acc}")
            score_1_label = "level1_retrieval"
            score_2_label = "level2_retrieval"
        elif data["only_level_3"]:
            dev_acc = predicted_output[f"level{3}_{'dev'}_decision_acc"]
            dev_decision_acc = predicted_output[f"level{3}_{'dev'}_decision_acc"]
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), decision dev_acc (level 3): {dev_acc}")
            score_1_label = "level3_decision"
        else:
            # currently pulling retrieval acc from level 2
            dev_acc = predicted_output[f"level{2}_{'dev'}_retrieval_acc"]
            dev_decision_acc = predicted_output[f"level{3}_{'dev'}_decision_acc"]
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), retrieval dev_acc (level 2): {dev_acc}")
            print(f"epoch: {e + 1} (BERT frozen: {freeze_bert_this_epoch}), decision dev_acc (level 3): {dev_decision_acc}")
            score_1_label = "level2_retrieval"
            score_2_label = "level3_decision"

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            print("WARNING: Early stopping is not fully implemented in this version.")
            break
        else:
            pre_dev_acc = dev_acc

        already_saved_scores = False
        # retrieval:
        if dev_acc >= max_dev_acc:
            max_dev_acc = dev_acc
            max_dev_acc_epoch = e + 1
            if only_save_best_models:
                print(
                    f"Saving epoch {e + 1} as new best max_dev_{score_1_label}_acc_epoch model")
                utils.save_model_torch(model, params, f"max_dev_{score_1_label}_acc_epoch")

                print("Saving scores file")
                for level_id in levels_to_consider:
                    scores_file_name = params["score_vals_file"] + f".epoch{e + 1}.level{level_id}.compact.txt"
                    utils.save_lines(scores_file_name, predicted_output[f"level{level_id}_{'dev'}_score_vals_compact"])
                    print(f"Saved compact scores file: {scores_file_name}")
                already_saved_scores = True
                if params["fine_tune_bert"]:
                    # Note that we don't need to re-save bert_model in the non-update epochs, but to keep things simple,
                    # in this version we always save. We can change this in the future, but we have to be careful
                    # of the edge case in which the most recent BERT fine-tuning wasn't saved since it wasn't a max score
                    #if not freeze_bert_this_epoch:
                    save_bert_ft_model(bert_model, tokenizer, params["bert_ft_dir"])
        print(f"\tCurrent max dev {score_1_label} accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")

        # decision
        # To keep it simple and to avoid collisions (and for analysis), we simply save both models (and fine-tuned BERT,
        # saved to separate directories) even if the same epoch for max retrieval and decision accuracies.
        # However, the scores files are not overwritten if they are the same epoch, since they are identical.
        if not data["only_level_1"] and not data["only_level_3"]:
            if dev_decision_acc >= max_dev_decision_acc:
                max_dev_decision_acc = dev_decision_acc
                max_dev_decision_acc_epoch = e + 1
                if only_save_best_models:
                    print(
                        f"Saving epoch {e + 1} as new best max_dev_{score_2_label}_acc_epoch model")
                    utils.save_model_torch(model, params, f"max_dev_{score_2_label}_acc_epoch")
                    if not already_saved_scores:
                        print("Saving scores file")
                        for level_id in levels_to_consider:
                            scores_file_name = params["score_vals_file"] + f".epoch{e + 1}.level{level_id}.compact.txt"
                            utils.save_lines(scores_file_name,
                                             predicted_output[f"level{level_id}_{'dev'}_score_vals_compact"])
                            print(f"Saved compact scores file: {scores_file_name}")

                    if params["fine_tune_bert"]:
                        # Note that we don't need to re-save bert_model in the non-update epochs, but to keep things simple,
                        # in this version we always save. We can change this in the future, but we have to be careful
                        # of the edge case in which the most recent BERT fine-tuning wasn't saved since it wasn't a max score
                        #if not freeze_bert_this_epoch:
                        save_bert_ft_model(bert_model, tokenizer, params["bert_ft_aux_dir"])
            print(f"\tCurrent max dev {score_2_label} accuracy: {max_dev_decision_acc} at epoch {max_dev_decision_acc_epoch}")

        # save after *every* epoch, unless only_save_best_models
        if e + 1 > 0:
            if params["SAVE_MODEL"] and not only_save_best_models:
                assert False, f"Saving every epoch is not currently implemented. Need to update BERT saving approach."

        if params["fine_tune_bert"]:
            if freeze_bert_this_epoch:
                if total_bert_fine_tuning_epochs >= params["freeze_bert_after_epoch_num"]:
                    print(f"About to start epoch {e+1}, after fine-tuning BERT for {total_bert_fine_tuning_epochs} epochs, so training BERT has stopped based on --freeze_bert_after_epoch_num {params['freeze_bert_after_epoch_num']}")
                    freeze_bert_this_epoch = True
                else:
                    freeze_bert_this_epoch = False
            else:
                total_bert_fine_tuning_epochs += 1
                freeze_bert_this_epoch = True

    print(f"Final max dev {score_1_label} accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")
    if not data["only_level_1"] and not data["only_level_3"]:
        print(f"Final max dev {score_2_label} accuracy: {max_dev_decision_acc} at epoch {max_dev_decision_acc_epoch}")

    print(f"Cumulative overall training time: {(time.time() - start_train_time) / 60} minutes")


def save_bert_ft_model(model, tokenizer, bert_ft_dir):
    # Save a trained model, configuration and tokenizer -- this is adapted from README in the transformer repo and run_classifier.py

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(bert_ft_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(bert_ft_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(bert_ft_dir)

    print(f"Saved the fine-tuned BERT model and vocabulary to {bert_ft_dir}")


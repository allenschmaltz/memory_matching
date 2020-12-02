# This is used for an ec layer and is not used for the memory matching model/paper.

from model import CNN
import memory_match as run_main
import utils
import constants
import utils_eval
import utils_viz
import utils_search
import utils_ft_train as utils_ft

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


def train(data, params, np_random_state, bert_model, tokenizer, bert_device, only_save_best_models=False, model=None):
    assert only_save_best_models
    assert model is not None

    start_train_time = time.time()

    print("Starting training")
    print(f"Freezing the entire model except for the ec level.")
    model = utils_ft.update_grad_status_of_cnn_model(model, [1, 2, 3], cnn_and_emb_requires_grad_bool=False, modify_ec=True,
                                    ec_requires_grad_bool=True)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    # In this version, the level ec is only trained with one instance per claim
    number_of_levelec_instances = len(data[f"idx_{'train'}_x"])
    total_estimated_instances = number_of_levelec_instances
    print(f"Total estimated number of training instances per epoch: {total_estimated_instances}")

    # the BERT optimizer follows run_classifer.py in the transformers repo
    if params["fine_tune_bert"]:
        param_optimizer = list(bert_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        # # In this version, the level ec is only trained with one instance per claim
        # number_of_levelec_instances = len(data[f"idx_{'train'}_x"])
        # total_estimated_instances = number_of_levelec_instances
        # print(f"Total estimated number of training instances per epoch: {total_estimated_instances}")
        num_train_optimization_steps = int(total_estimated_instances / params["BATCH_SIZE"]) * params[
            "bert_num_train_epochs"]
        bert_optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=params["bert_learning_rate"],
                                 warmup=params["bert_warmup_proportion"],
                                 t_total=num_train_optimization_steps)


    pre_dev_acc = 0
    max_dev_acc = 0
    #max_test_acc = 0
    max_dev_acc_epoch = -1

    recorded_nvidia_smi = False
    recorded_nvidia_smi_for_bert = False

    freeze_bert_this_epoch = True
    total_bert_fine_tuning_epochs = 0

    all_epoch_cumulative_losses = []
    all_epoch_cumulative_losses_bert = []
    all_epoch_cumulative_losses_cnn = []


    num_batch_instances = math.ceil( ( total_estimated_instances / params["BATCH_SIZE"]) )

    for e in range(params["EPOCH"]):
        start_epoch_time = time.time()
        if freeze_bert_this_epoch:
            print(f"TRAIN-TYPE: Freezing BERT, training CNN")
        else:
            print(f"TRAIN-TYPE: Freezing CNN, training BERT")

        data["idx_train_x"], \
        data["train_bert_idx_sentences"], \
        data["train_bert_input_masks"], \
        data["train_y"], \
        data[f"{'train'}_decision_labels"], \
        data[f"{'train'}_predicted_level3_decision"] = \
            shuffle(data["idx_train_x"],
                    data["train_bert_idx_sentences"],
                    data["train_bert_input_masks"],
                    data["train_y"],
                    data[f"{'train'}_decision_labels"],
                    data[f"{'train'}_predicted_level3_decision"], random_state=np_random_state)

        print(f"Training accuracy at the beginning of epoch {e}:")
        test(data, model, params, bert_model, bert_device, mode="train")
        if e == 0:
            print(f"Dev accuracy at the beginning of epoch {e}:")
            test(data, model, params, bert_model, bert_device, mode="dev")
        batch_num = 0
        cumulative_losses = []
        cumulative_losses_bert = []
        cumulative_losses_cnn = []

        for i in range(0, total_estimated_instances, params["BATCH_SIZE"]):
            if batch_num % int(num_batch_instances * 0.25) == 0:
                print(f"Epoch {e+1}, {batch_num/num_batch_instances}")
                if len(cumulative_losses) > 0:
                    print(f"\tCurrent epoch average loss: {np.mean(cumulative_losses)}")
            batch_num += 1
            batch_range = min(params["BATCH_SIZE"], total_estimated_instances - i)

            batch_x = data["idx_train_x"][i:i + batch_range]
            batch_x = torch.LongTensor(batch_x).to(params["main_device"])
            batch_y = data["train_y"][i:i + batch_range]
            batch_y = torch.LongTensor(batch_y).to(params["main_device"])
            if freeze_bert_this_epoch:
                model.train()
                bert_model.eval()
                model = utils_ft.update_grad_status_of_cnn_model(model, [1,2,3],
                                                        cnn_and_emb_requires_grad_bool=False,
                                                        modify_ec=True,
                                                        ec_requires_grad_bool=True)
                # get BERT frozen representations
                bert_output = run_main.get_bert_representations(data["train_bert_idx_sentences"][i:i + batch_range],
                                                               data["train_bert_input_masks"][i:i + batch_range],
                                                               bert_model, bert_device, params["bert_layers"],
                                                               len(batch_x[0]))
                bert_output = torch.FloatTensor(bert_output).to(params["main_device"])
                optimizer.zero_grad()

            else:
                model.eval()
                bert_model.train()
                model = utils_ft.update_grad_status_of_cnn_model(model, [1,2,3],
                                                        cnn_and_emb_requires_grad_bool=False,
                                                        modify_ec=True,
                                                        ec_requires_grad_bool=False)
                bert_optimizer.zero_grad()
                # get BERT representations
                bert_output = utils_ft.get_bert_representations_ft(data["train_bert_idx_sentences"][i:i + batch_range],
                                                                data["train_bert_input_masks"][i:i + batch_range],
                                                                bert_model, bert_device, params["bert_layers"],
                                                                len(batch_x[0]))


            #total_length = params["max_length"] * 3 + 2 * constants.PADDING_SIZE
            total_length = params["ec_max_length"] + 2 * constants.PADDING_SIZE
            pred = model(batch_x, bert_output, level_id="ec", total_length=total_length,
                  forward_type_description="sentence_representation", main_device=None)
            loss = criterion(pred, batch_y)

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
                print(f"+++++++++++++++After backward: Freeze BERT: {freeze_bert_this_epoch}+++++++++++++++")
                nvidia_smi_out = subprocess.run(["nvidia-smi"], capture_output=True)
                print(f'{nvidia_smi_out.stdout.decode("utf-8")}')
                if freeze_bert_this_epoch:
                    recorded_nvidia_smi = True
                else:
                    recorded_nvidia_smi_for_bert = True
                print(f'torch.cuda.memory_allocated: {torch.cuda.memory_allocated(params["main_device"])}')
                print(f'torch.cuda.max_memory_allocated: {torch.cuda.max_memory_allocated(params["main_device"])}')

        utils_ft.update_and_display_losses(cumulative_losses, all_epoch_cumulative_losses, "combined")
        if freeze_bert_this_epoch:
            utils_ft.update_and_display_losses(cumulative_losses_cnn, all_epoch_cumulative_losses_cnn, "CNN")
        else:
            utils_ft.update_and_display_losses(cumulative_losses_bert, all_epoch_cumulative_losses_bert, "BERT")

        end_epoch_time = time.time()
        print(f"Time to complete epoch: {(end_epoch_time - start_epoch_time) / 60} minutes")
        print(f"Cumulative overall training time: {(time.time() - start_train_time) / 60} minutes")

        dev_acc, score_vals = test(data, model, params, bert_model, bert_device, mode="dev")

        #test_acc, _ = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc) #, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc >= max_dev_acc:
            max_dev_acc = dev_acc
            max_dev_acc_epoch = e + 1
            if only_save_best_models:
                print(
                    f"Saving epoch {e + 1} as new best max_dev_ec_level_acc_epoch_{data['ec_model_suffix']} model")
                utils.save_model_torch(model, params, f"max_dev_ec_level_acc_epoch_{data['ec_model_suffix']}")

                print("Saving scores file")
                scores_filename = params["score_vals_file"] + f".epoch{e+1}.max_dev_ec_{data['ec_model_suffix']}.txt"
                utils.save_lines(scores_filename, score_vals)
                print(f"Saved scores file: {scores_filename}")

                if params["fine_tune_bert"]:
                    # Note that we don't need to re-save bert_model in the non-update epochs, but to keep things simple,
                    # in this version we always save. We can change this in the future, but we have to be careful
                    # of the edge case in which the most recent BERT fine-tuning wasn't saved since it wasn't a max score
                    utils_ft.save_bert_ft_model(bert_model, tokenizer, params["bert_ec_ft_dir"])

        print(f"\tCurrent max dev accuracy: {max_dev_acc} at epoch {max_dev_acc_epoch}")

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

    print(f"Final max dev accuracy (from level 3 ec): {max_dev_acc} at epoch {max_dev_acc_epoch}")

    print(f"Cumulative overall training time: {(time.time() - start_train_time) / 60} minutes")


def test(data, model, params, bert_model, bert_device, mode="test"):
    if params["only_2_class"] or data["eval_symmetric_data"]:
        return test_2_class_ec(data, model, params, bert_model, bert_device, mode=mode)
    else:
        return test_3_class_ec(data, model, params, bert_model, bert_device, mode=mode)


def test_2_class_ec(data, model, params, bert_model, bert_device, mode="test"):
    """
    Calculate the sentence-level accuracy and return the output logits
    :param data:
    :param model:
    :param params:
    :param bert_model:
    :param bert_device:
    :param mode:
    :return:
    """
    model.eval()
    bert_model.eval()

    x, bert_idx_sentences, bert_input_masks, y, decision_labels, predicted_level3_decision = \
        data[f"idx_{mode}_x"], data[f"{mode}_bert_idx_sentences"], data[f"{mode}_bert_input_masks"], \
        data[f"{mode}_y"], data[f"{mode}_decision_labels"], data[f"{mode}_predicted_level3_decision"]

    score_vals = []
    predicted_ec = []
    predicted_updated_decision = []  # flip decision based on ec
    number_of_flipped_decisions = 0
    with torch.no_grad():
        for i in range(0, len(x), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(x) - i)

            batch_x = x[i:i + batch_range]

            # get BERT representations
            bert_output = run_main.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                            bert_input_masks[i:i + batch_range],
                                                            bert_model, bert_device,
                                                            params["bert_layers"], len(batch_x[0]))

            batch_y = y[i:i + batch_range]
            batch_decision_labels = decision_labels[i:i + batch_range]
            batch_predicted_level3_decision = predicted_level3_decision[i:i + batch_range]

            if params["GPU"] != -1:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output))

            total_length = params["ec_max_length"] + 2 * constants.PADDING_SIZE
            model_output = model(batch_x, bert_output, level_id="ec", total_length=total_length,
                         forward_type_description="sentence_representation", main_device=None)
            model_output = model_output.cpu().data.numpy()
            for j, gold in enumerate(batch_y):

                original_level3_decision = batch_predicted_level3_decision[j]
                flipped = False
                if model_output[j][1] > model_output[j][0]:
                    number_of_flipped_decisions += 1
                    flipped = True
                    if original_level3_decision == constants.SUPPORTS_ID:
                        original_level3_decision = constants.REFUTES_ID
                    elif original_level3_decision == constants.REFUTES_ID:
                        original_level3_decision = constants.SUPPORTS_ID
                score_vals.append(f"Hash:{gold}{original_level3_decision}{flipped}{original_level3_decision == batch_decision_labels[j]}\tWas error:{gold}\tNew label:{original_level3_decision}\tFlipped:{flipped}\tCorrect:{original_level3_decision == batch_decision_labels[j]}\t{model_output[j][0]}\t{model_output[j][1]}\n")
                predicted_updated_decision.append(int(original_level3_decision == batch_decision_labels[j]))

            predicted_ec.extend(np.argmax(model_output, axis=1))

    acc = sum([1 if p == y else 0 for p, y in zip(predicted_ec, y)]) / len(predicted_ec)
    print(f"EC acc {mode}: {acc}")
    print(f"EC Stats {mode}: Total flipped level 3 decisions: {number_of_flipped_decisions}")
    print(f"EC Stats {mode}: Updated decision accuracy: {np.mean(predicted_updated_decision)}: "
          f"{np.sum(predicted_updated_decision)} out of {len(predicted_updated_decision)}")
    if mode == "test":
        # random eval (as a check)
        pred_random = np.random.choice(2, len(predicted_ec))
        acc_random = sum([1 if p == y else 0 for p, y in zip(pred_random, y)]) / len(pred_random)
        print(f"\t(Accuracy from random prediction (only for debugging purposes): {acc_random})")
        # always predict 1
        pred_all_ones = np.ones(len(predicted_ec))
        acc_all_ones = sum([1 if p == y else 0 for p, y in zip(pred_all_ones, y)]) / len(pred_all_ones)
        print(f"\t(Accuracy from all 1's prediction (only for debugging purposes): {acc_all_ones})")
        print(f"\tGround-truth Stats: Number of instances with class 1: {np.sum(y)} out of {len(y)}")

    return np.mean(predicted_updated_decision), score_vals


def test_3_class_ec(data, model, params, bert_model, bert_device, mode="test"):
    """
    Calculate the sentence-level accuracy and return the output logits
    :param data:
    :param model:
    :param params:
    :param bert_model:
    :param bert_device:
    :param mode:
    :return:
    """
    model.eval()
    bert_model.eval()

    x, bert_idx_sentences, bert_input_masks, y, decision_labels, predicted_level3_decision = \
        data[f"idx_{mode}_x"], data[f"{mode}_bert_idx_sentences"], data[f"{mode}_bert_input_masks"], \
        data[f"{mode}_y"], data[f"{mode}_decision_labels"], data[f"{mode}_predicted_level3_decision"]

    score_vals = []
    predicted_ec = []
    predicted_updated_decision = []  # flip decision based on ec
    number_of_flipped_decisions = 0
    with torch.no_grad():
        for i in range(0, len(x), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(x) - i)

            batch_x = x[i:i + batch_range]

            # get BERT representations
            bert_output = run_main.get_bert_representations(bert_idx_sentences[i:i + batch_range],
                                                            bert_input_masks[i:i + batch_range],
                                                            bert_model,
                                                            bert_device,
                                                            params["bert_layers"],
                                                            len(batch_x[0]))

            batch_y = y[i:i + batch_range]
            batch_decision_labels = decision_labels[i:i + batch_range]
            batch_predicted_level3_decision = predicted_level3_decision[i:i + batch_range]

            if params["GPU"] != -1:
                batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output)).cuda(params["GPU"])
            else:
                batch_x = Variable(torch.LongTensor(batch_x))
                if bert_model is not None:
                    bert_output = Variable(torch.FloatTensor(bert_output))

            total_length = params["ec_max_length"] + 2 * constants.PADDING_SIZE
            model_output = model(batch_x, bert_output, level_id="ec", total_length=total_length,
                         forward_type_description="sentence_representation", main_device=None)
            model_output = model_output.cpu().data.numpy()
            for j, gold in enumerate(batch_y):
                # This is the original prediction of level 3 from the base model
                updated_level3_decision = batch_predicted_level3_decision[j]
                flipped = False
                class_prediction = np.argmax(model_output[j])
                # If the EC predicts as wrong, then we update the label
                if class_prediction != constants.EC_CORRECT_ID:
                    number_of_flipped_decisions += 1
                    flipped = True
                    updated_level3_decision = class_prediction
                score_vals.append(f"Hash:{gold}{batch_decision_labels[j]}{updated_level3_decision}{flipped}{updated_level3_decision == batch_decision_labels[j]}\tOriginal EC:{gold}\tGold Decision:{batch_decision_labels[j]}\tNew label:{updated_level3_decision}\tFlipped:{flipped}\tCorrect:{updated_level3_decision == batch_decision_labels[j]}\t{model_output[j][0]}\t{model_output[j][1]}\t{model_output[j][2]}\t{model_output[j][3]}\n")
                predicted_updated_decision.append(int(updated_level3_decision == batch_decision_labels[j]))

            predicted_ec.extend(np.argmax(model_output, axis=1))

    acc = sum([1 if p == y else 0 for p, y in zip(predicted_ec, y)]) / len(predicted_ec)
    print(f"EC acc {mode}: {acc}")
    print(f"EC Stats {mode}: Total flipped level 3 decisions: {number_of_flipped_decisions}")
    print(f"EC Stats {mode}: Updated decision accuracy: {np.mean(predicted_updated_decision)}: "
          f"{np.sum(predicted_updated_decision)} out of {len(predicted_updated_decision)}")
    if mode == "test":
        # random eval (as a check)
        pred_random = np.random.choice(3, len(predicted_ec))
        acc_random = sum([1 if p == y else 0 for p, y in zip(pred_random, y)]) / len(pred_random)
        print(f"\t(Accuracy from random prediction (only for debugging purposes): {acc_random})")
        # always predict 1
        pred_all_ones = np.ones(len(predicted_ec))
        acc_all_ones = sum([1 if p == y else 0 for p, y in zip(pred_all_ones, y)]) / len(pred_all_ones)
        print(f"\t(Accuracy from all 1's prediction (only for debugging purposes): {acc_all_ones})")
        print(f"\tGround-truth Stats: Number of instances with class 1: {np.sum(y)} out of {len(y)}")

    return np.mean(predicted_updated_decision), score_vals

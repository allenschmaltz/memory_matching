"""
Main entry point. This is internal research code. The full release will be in the main directory of the repo.
Most of the heavy lifting for:
    Search is in utils_search.py, serving as the entry point for batch forwards through the search graph,
        caching memory vectors, and dynamically creating the input sequences
    Training is in utils_ft_train.py
    Exemplar auditing is in utils_exemplar.py
    Alignment visualization (of level 1) is in utils_alignment_visualization.py (Note that viz here is only to standard
        out and is largely unused with FEVER given the nature of the task, and the current implementation is
        only expected to be used with the 2-class symmetric data for illustrative purposes.
        The BLADE implementation has more sophisticated viz for token-level tasks which I'll merge in the
        final version.)

The file utils.py handles the rather complicated data processing. I aim to simplify the file formats in the final
version.

Naming conventions: 'title' refers to support sequence; 'retrieve' is used for query sequences

Note that there are some passing references to an 'ec' ('error correction') layer. This is for BLADE/mulitBLADE -- i.e.,
incorporating an additional classification layer when freezing the full retrieval system, which is intended as a
drop-in replacement for use cases otherwise using BLADE/mulitBLADE for classification. These are situations where
token-level analysis is important, and by combining with the model here, we can also incorporate explicit retrieval.
That will appear in the released codebase, and will also unify the BLADE/mulitBLADE repos.

Note that by incorporating BLADE/mulitBLADE with retrieval-classification memory matching, we will have a
unified implementation of 'Full Resolution Language/Sequence Modeling':
 1. Updatability via the retrieval datastore
 2. Updatability via exemplar auditing: Both in terms of difference vectors and token-level memories in the case
    of BLADE/mulitBLADE
 3. Visualization (and associated analysis use cases) of level alignments and token-level contributions
 4. Ability to constrain and analyze the model via level distances and exemplar distances
 5. Ability to analyze corpora via the above and the sequence feature weighting as demonstrated in the BLADE paper
    (i.e., defacto extractive, comparative summarization).

"""

from model import CNN
import utils
import constants
import utils_eval
import utils_viz
import utils_search
import utils_ec
import utils_ft_only_ec
import utils_ft_train
import utils_exemplar
import utils_alignment_visualization


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

from collections import defaultdict

import torch.nn.functional as F

import subprocess
import time


from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME


# move these to the constants file:
ID_CORRECT = 0 # "negative" class (correct token)
ID_WRONG = 1  # "positive class" (token with error)
HOLDER_SYM = "$$$HOLDER$$$"  # every sentence has a final place holder to account for insertions at the end of the sentence -- this is only for sequence-labeling
INS_START_SYM = "<ins>"
INS_END_SYM = "</ins>"
DEL_START_SYM = "<del>"
DEL_END_SYM = "</del>"


def get_bert_representations(train_bert_idx_sentences, train_bert_input_masks, bert_model, bert_device, bert_layers, total_length):
    """
    Get BERT output representations and reshape them to match the padding of the CNN output (in preparation for concat).
    This is only for use when BERT is frozen. Otherwise use get_bert_representations_ft() in utils_ft_train.py.
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
    with torch.no_grad():
        all_encoder_layers, _ = bert_model(train_bert_idx_sentences, token_type_ids=None, attention_mask=train_bert_input_masks)
        top_layers = []
        for one_layer_index in bert_layers:
            top_layers.append(all_encoder_layers[one_layer_index])
        layer_output = torch.cat(top_layers, 2)

        # zero trailing padding
        masks = torch.unsqueeze(train_bert_input_masks, 2).float()
        layer_output = layer_output*masks

        bert_output_reshaped = torch.zeros((layer_output.shape[0], total_length, layer_output.shape[2]))
        bert_output_reshaped[:, constants.PADDING_SIZE-1:(constants.PADDING_SIZE-1)+layer_output.shape[1], :] = layer_output

    return bert_output_reshaped.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="-----[retrieval-classification]-----")
    parser.add_argument("--mode", default="train", help="train: train (with eval on dev_file) a model / test: test saved models; zero performs zero shot labeling; seq_labeling_fine_tune performs fine tuning with token-level labels, which must be provided")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="aesw", help="available datasets: aesw")
    parser.add_argument("--word_embeddings_file", default="", help="word_embeddings_file")
    parser.add_argument("--training_file", default="", help="training_file")
    parser.add_argument("--dev_file", default="", help="dev_file")
    parser.add_argument("--test_file", default="", help="test_file")
    parser.add_argument("--max_length", default=100, type=int, help="max sentence length (set for training); eval sentences are truncated to this length at inference time")
    parser.add_argument("--max_vocab_size", default=100000, type=int, help="max vocab size (set for training)")
    parser.add_argument("--vocab_file", default="", help="Vocab file")
    parser.add_argument("--use_existing_vocab_file", default=False, action='store_true', help="Use existing vocab for training. Always true for test.")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--dropout_probability", default=0.5, type=float, help="dropout_probability for training; default is 0.5")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used; -1 for cpu")
    parser.add_argument("--save_dir", default="saved_models", help="save_dir")
    parser.add_argument("--saved_model_file", default="", help="path to existing model (for test mode only)")
    parser.add_argument("--score_vals_file", default="", help="score_vals_file")
    parser.add_argument("--seed_value", default=1776, type=int, help="seed_value")
    parser.add_argument("--data_formatter", default="", help="use 'fce' for fce replication; use 'lowercase' for uncased BERT models")
    parser.add_argument("--word_embeddings_file_in_plaintext", default=False, action='store_true', help="embeddings file is in plain text format")

    parser.add_argument("--filter_widths", default="3,4,5", type=str)
    parser.add_argument("--number_of_filter_maps", default="100,100,100", type=str)

    # for BERT:
    parser.add_argument("--bert_cache_dir", default="", type=str)
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--bert_layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--bert_gpu", default=1, type=int, help="the number of gpu to be used for the BERT model; -1 for cpu")

    # for zero-shot labeling:
    parser.add_argument("--color_gradients_file", default="", help="color_gradients_file")
    parser.add_argument("--visualization_out_file", default="", help="visualization_out_file")
    parser.add_argument("--correction_target_comparison_file", default="", help="correction_target_comparison_file")
    parser.add_argument("--output_generated_detection_file", default="", help="output_generated_detection_file")
    parser.add_argument("--detection_offset", type=int, help="detection_offset")
    parser.add_argument("--fce_eval", default=False, action='store_true', help="fce_eval")
    parser.add_argument("--test_seq_labels_file", default="", help="test_seq_labels_file")

    # for fine-tuning labeling
    parser.add_argument("--training_seq_labels_file", default="", help="training_seq_labels_file")
    parser.add_argument("--dev_seq_labels_file", default="", help="dev_seq_labels_file")

    parser.add_argument("--forward_type", default=2, type=int, help="forward_type for sequence training")

    # for topics
    parser.add_argument("--output_topics_file", default="", help="output_topics_file")

    # for input data without tokenized negations: i.e., normal text: can't is can't and not can n't
    parser.add_argument("--input_is_untokenized", default=False, action='store_true', help="for input data without tokenized negations: i.e., normal text: 'can't' is 'can't' and not 'can n't'")

    # additional (diff from bert_cnnv2_finetune_rf/)
    parser.add_argument("--use_sentence_prediction_for_labeling", default=False, action='store_true', help="If provided, positive token label are only considered for sentences classified as positive.")
    parser.add_argument("--output_neg_features_file", default="", help="output_neg_features_file")
    parser.add_argument("--output_pos_features_file", default="", help="output_pos_features_file")
    parser.add_argument("--output_neg_sentence_features_file", default="", help="output_neg_sentence_features_file")
    parser.add_argument("--output_pos_sentence_features_file", default="", help="output_pos_sentence_features_file")

    parser.add_argument("--only_save_best_models", default=False, action='store_true',
                        help="only save model with best acc")

    # viz options
    parser.add_argument("--only_visualize_missed_predictions", default=False, action='store_true',
                        help="only_visualize_missed_predictions")
    parser.add_argument("--only_visualize_correct_predictions", default=False, action='store_true',
                        help="only_visualize_correct_predictions")
    parser.add_argument("--do_not_tune_offset", default=False, action='store_true',
                        help="do_not_tune_offset")

    # metric
    parser.add_argument("--forward_type_description", default="maxpool_no_relu_no_dropout", help="forward_type_description")
    parser.add_argument("--retrieval_forward_type_description", default="maxpool_no_relu_no_dropout",
                        help="Note that dropout is never used for retrieval, so matching --forward_type_description is analogous to the typical train/eval dropout differences.")
    parser.add_argument("--only_consider_negative_pairs", default=False, action='store_true',
                        help="only add pairs with missed predictions")
    parser.add_argument("--error_correction_forward_type_description", default="error_correction_no_relu_dropout",
                        help="error_correction_forward_type_description")
    parser.add_argument("--batch_size", default=50, type=int, help="batch_size")
    parser.add_argument("--use_auto_encoder_loss", default=False, action='store_true',
                        help="use_auto_encoder_loss")
    parser.add_argument("--unique_titles_file", default="", help="unique_titles_file")
    parser.add_argument("--do_not_save_test_eval_output", default=False, action='store_true',
                        help="do_not_save_test_eval_output")

    parser.add_argument("--train_true_titles_file", default="", help="train_true_titles_file")
    parser.add_argument("--dev_true_titles_file", default="", help="dev_true_titles_file")
    parser.add_argument("--test_true_titles_file", default="", help="test_true_titles_file")

    parser.add_argument("--train_covered_titles_file", default="", help="train_covered_titles_file")
    parser.add_argument("--dev_covered_titles_file", default="", help="dev_covered_titles_file")
    parser.add_argument("--test_covered_titles_file", default="", help="test_covered_titles_file")

    # BERT fine-tuning (some of these are adapted from run_classifier.py from the transformers repo):
    parser.add_argument("--fine_tune_bert", default=False, action='store_true', help="fine_tune_bert")
    parser.add_argument("--bert_learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--bert_num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--bert_warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--freeze_bert_after_epoch_num", default=3, type=int, help="Stop BERT fine-tuning and freeze after this many epochs. In the current version, this should typically match --bert_num_train_epochs.")
    parser.add_argument("--word_embedding_size", default=300, type=int,
                        help="word_embedding_size")
    parser.add_argument("--bert_ft_dir", default="", help="bert_ft_dir")
    parser.add_argument("--bert_ft_aux_dir", default="", help="bert_ft_aux_dir used in fine-tuning to save alternative epochs when saving epochs based on metrics; use --bert_ft_dir for test")
    parser.add_argument("--load_ft_bert", default=False, action='store_true', help="If testing a fine-tuned BERT model, use this and provide --bert_ft_dir. Otherwise, the pre-trained model will be loaded.")
    parser.add_argument("--do_not_save_detailed_scores_output", default=False, action='store_true',
                        help="do_not_save_detailed_scores_output")

    parser.add_argument("--train_decision_labels_file", default="", help="train_decision_labels_file")
    parser.add_argument("--dev_decision_labels_file", default="", help="dev_decision_labels_file")
    parser.add_argument("--test_decision_labels_file", default="", help="test_decision_labels_file")

    parser.add_argument("--train_chosen_sentences_only_evidence_file", default="", help="train_chosen_sentences_only_evidence_file")
    parser.add_argument("--dev_chosen_sentences_only_evidence_file", default="", help="dev_chosen_sentences_only_evidence_file")
    parser.add_argument("--test_chosen_sentences_only_evidence_file", default="", help="test_chosen_sentences_only_evidence_file")

    parser.add_argument("--train_covered_sentences_dictionary_file", default="", help="train_covered_sentences_dictionary_file")
    parser.add_argument("--dev_covered_sentences_dictionary_file", default="", help="dev_covered_sentences_dictionary_file")
    parser.add_argument("--test_covered_sentences_dictionary_file", default="", help="test_covered_sentences_dictionary_file")

    parser.add_argument("--level1_memory_batch_size", default=150, type=int, help="memory_batch_size for level 1")
    parser.add_argument("--level1_retrieval_batch_size", default=150, type=int, help="retrieval_batch_size for level 1")
    parser.add_argument("--level1_titles_chunk_size", default=50000, type=int, help="titles_chunk_size for level 1")
    parser.add_argument("--level1_retrieval_chunk_size", default=1500, type=int, help="retrieval_chunk_size for level 1")

    parser.add_argument("--level2_memory_batch_size", default=150, type=int, help="memory_batch_size for level 2")
    parser.add_argument("--level2_retrieval_batch_size", default=150, type=int, help="retrieval_batch_size for level 2")
    parser.add_argument("--level2_titles_chunk_size", default=50000, type=int, help="titles_chunk_size for level 2")
    parser.add_argument("--level2_retrieval_chunk_size", default=1500, type=int, help="retrieval_chunk_size for level 2")

    parser.add_argument("--level3_memory_batch_size", default=150, type=int, help="memory_batch_size for level 3")
    parser.add_argument("--level3_retrieval_batch_size", default=150, type=int, help="retrieval_batch_size for level 3")
    parser.add_argument("--level3_titles_chunk_size", default=50000, type=int, help="titles_chunk_size for level 3")
    parser.add_argument("--level3_retrieval_chunk_size", default=1500, type=int, help="retrieval_chunk_size for level 3")

    parser.add_argument("--titles_memory_dir", default="", help="titles_memory_dir")
    parser.add_argument("--retrieval_memory_dir", default="", help="retrieval_memory_dir (for saving intermediate train/dev/etc. memories)")

    parser.add_argument("--level1_top_k_nearest_memories", default=3, type=int, help="level1_top_k_nearest_memories")
    parser.add_argument("--level2_top_k_nearest_memories", default=3, type=int, help="level2_top_k_nearest_memories")
    parser.add_argument("--level3_top_k_nearest_memories", default=3, type=int, help="level3_top_k_nearest_memories")

    parser.add_argument("--level3_top_k_stratifications", default=3, type=int, help="level3_top_k_stratifications")
    parser.add_argument("--only_2_class", default=False, action='store_true',
                        help="only consider supports or refutes")
    parser.add_argument("--level3_max_1_evidence_constructions", default=1, type=int,
                        help="Number of size 1 evidence constructions for each claim generated for level 3. For "
                             "example, 1 (the min), means that only the top of the beam from level 2 is taken, "
                             "onto which all classification labels are pre-pended.")
    parser.add_argument("--level3_max_2_evidence_constructions", default=0, type=int,
                        help="If 0, size 2 evidence constructions are never generated for level 3.")
    parser.add_argument("--do_not_marginalize_over_level3_evidence", default=False, action='store_true',
                        help="do_not_marginalize_over_level3_evidence")
    parser.add_argument("--level3_top_k_evidence_predictions", default=3, type=int, help="level3_top_k_evidence_predictions")

    parser.add_argument("--only_levels_1_and_2", default=False, action='store_true',
                        help="only train and eval levels 1 and 2 (i.e., retrieval only)")
    parser.add_argument("--only_level_1", default=False, action='store_true',
                        help="only train and eval level 1 (i.e., initial coarse retrieval only)")

    parser.add_argument("--init_level_2_with_level_1_weights", default=False, action='store_true',
                        help="init_level_2_with_level_1_weights")
    parser.add_argument("--init_level_3_with_level_2_weights", default=False, action='store_true',
                        help="init_level_3_with_level_2_weights")
    parser.add_argument("--continue_training", default=False, action='store_true',
                        help="continue_training")

    parser.add_argument("--only_level_3", default=False, action='store_true',
                        help="only train and eval level 3 (i.e., final decision only)")

    parser.add_argument("--save_output_for_ec", default=False, action='store_true',
                        help="Save training/eval output for the ec layer to --score_vals_file.")

    parser.add_argument("--eval_constrained", default=False, action='store_true',
                        help="Constrain based on --level3_constrained_mean and --level3_constrained_std")
    parser.add_argument("--level2_constrained_mean", default=0.0, type=float,
                        help="level2_constrained_mean")
    parser.add_argument("--level2_constrained_std", default=0.0, type=float,
                        help="level2_constrained_std")
    parser.add_argument("--level3_constrained_mean", default=0.0, type=float,
                        help="level3_constrained_mean")
    parser.add_argument("--level3_constrained_std", default=0.0, type=float,
                        help="level3_constrained_std")

    parser.add_argument("--create_hard_negative_for_unverifiable_retrieval_in_level2", default=False, action='store_true',
                        help="In the 3 class case, this adds a hard negative for level 2 for all unverifiable claims"
                             "in level2 for training. The hard negative is the top of the beam. Note this corresponds "
                             "to an unlabeled (negative) retrieval prediction, since there are no reference retrieval "
                             "sentences for unverifiable claims.")

    # for training ec:
    parser.add_argument("--only_ec", default=False, action='store_true',
                        help="For training or evaluating the ec level.")
    parser.add_argument("--training_ec_file", default="", help="training_ec_file")
    parser.add_argument("--dev_ec_file", default="", help="dev_ec_file")
    parser.add_argument("--test_ec_file", default="", help="test_ec_file")
    parser.add_argument("--bert_ec_ft_dir", default="", help="bert_ft_dir; Note that by design, when training ec,"
                                                             "it is necessary to copy one of the retrieval or decision "
                                                             "directories over before fine-tuning if using "
                                                             "--fine_tune_bert")
    parser.add_argument("--ec_max_length", default=150, type=int, help="ec_max_length")
    parser.add_argument("--ec_model_suffix", default="top_3_evidence", help="ec_model_suffix")
    parser.add_argument("--ec_model_update_key", default=1, type=int, help="controls updates to fc (only used for initial development)")

    # for evaluating symmetric data:
    parser.add_argument("--eval_symmetric_data", default=False, action='store_true',
                        help="Be careful in using this, as it has the effect of turning off some checks (e.g., "
                             "it no longer enforces the requirement of (wiki title, sentence ids) being unique. "
                             "This is only intended to be used with the symmetric eval dataset format. "
                             "In the current version, this is expected to be used with "
                             "--constrain_to_2_class_at_inference")
    parser.add_argument("--constrain_to_2_class_at_inference", default=False, action='store_true',
                        help="If provided, only 2 class decision labels (i.e., supports and refutes, but not"
                             "unverifiable) are considered in constructing the level 3 beam.")

    # for exemplar auditing:
    parser.add_argument("--create_exemplar_database", default=False, action='store_true',
                        help="Mode should otherwise be test")
    parser.add_argument("--create_exemplar_query", default=False, action='store_true',
                        help="Mode should otherwise be test")
    parser.add_argument("--exemplar_memory_batch_size", default=150, type=int, help="exemplar_memory_batch_size")
    parser.add_argument("--exemplar_chunk_size", default=4800, type=int, help="exemplar_chunk_size")
    parser.add_argument("--exemplar_database_memory_dir", default="", help="exemplar_database_memory_dir")
    parser.add_argument("--exemplar_query_memory_dir", default="", help="exemplar_query_memory_dir")
    parser.add_argument("--save_exemplar_output", default=False, action='store_true',
                        help="Save exemplar inference output to --score_vals_file.")
    parser.add_argument("--add_full_beam_to_exemplar_database", default=False, action='store_true',
                        help="Only considered with the --create_exemplar_database option. If provided, the full "
                             "level 3 decision beam is added to the exemplar database. If not, only the top of the "
                             "level 3 decision beam is added to the exemplar database.")
    parser.add_argument("--exemplar_match_type", default="all",
                        help="This only affects --save_exemplar_output; all levels are currently "
                             "saved when creating the database and query. "
                             "all: levels123 plus diff for each level, "
                             "level123_diff: level 1 diff AND level 2 diff AND level 3 diff, "
                             "level3_diff: only level 3 diff, "
                             "level23_diff: only level 2 diff AND level 3 diff, "
                             "level1_diff: only level 1 diff: use this with caution (and typically, only for analysis),"
                             " as --add_full_beam_to_exemplar_database means that the level 1 diff no longer fully "
                             "describes a database instance (and more generally, only the decision labels in level 3 "
                             "make a level12 instance unique).")

    # for visualization
    parser.add_argument("--visualize_alignment", default=False, action='store_true',
                        help="Save visualize alignment output to --score_vals_file.")

    options = parser.parse_args()

    seed_value = options.seed_value
    max_length = options.max_length
    max_vocab_size = options.max_vocab_size
    vocab_file = options.vocab_file
    use_existing_vocab_file = options.use_existing_vocab_file
    training_file = options.training_file.strip()
    dev_file = options.dev_file.strip()
    test_file = options.test_file.strip()
    data_formatter = options.data_formatter.strip()
    word_embeddings_file_in_plaintext = options.word_embeddings_file_in_plaintext

    torch.manual_seed(seed_value)
    np_random_state = np.random.RandomState(seed_value)

    if options.gpu != -1 or options.bert_gpu != -1:
        torch.cuda.manual_seed_all(seed_value)

    main_device = torch.device(f"cuda:{options.gpu}" if options.gpu > -1 else "cpu")

    # NOTE: Here are elsewhere, there are some references to 'zero-shot' labeling and sequence labeling. Those are from
    # the original BLADE codebase. I've removed some of those files to make this easier to follow. Those will get
    # re-added in the final version once I standardize the input formats.

    # for zero-shot labeling:
    color_gradients_file = options.color_gradients_file
    visualization_out_file = options.visualization_out_file
    correction_target_comparison_file = options.correction_target_comparison_file.strip()
    output_generated_detection_file = options.output_generated_detection_file.strip()
    detection_offset = options.detection_offset
    fce_eval = options.fce_eval

    assert options.dataset == "aesw"  # internal label for particular format; doesn't correspond to a real-world dataset
    assert options.score_vals_file.strip() != ""

    filter_widths = [int(x) for x in options.filter_widths.split(",")]
    number_of_filter_maps = [int(x) for x in options.number_of_filter_maps.split(",")]
    print(f"CNN: Filter widths: {filter_widths}")
    print(f"CNN: Number of filter maps: {number_of_filter_maps}")

    # sequence-level labels
    training_seq_labels_file = options.training_seq_labels_file.strip()
    if training_seq_labels_file == "":
        training_seq_labels_file = None
    dev_seq_labels_file = options.dev_seq_labels_file.strip()
    if dev_seq_labels_file == "":
        dev_seq_labels_file = None
    test_seq_labels_file = options.test_seq_labels_file.strip()
    if test_seq_labels_file == "":
        test_seq_labels_file = None

    if options.bert_model.strip() != "":
        bert_device = torch.device(f"cuda:{options.bert_gpu}" if options.bert_gpu > -1 else "cpu")

        if options.load_ft_bert:
            if options.only_ec:
                assert options.bert_ec_ft_dir != "", f"ERROR: --bert_ec_ft_dir must be provided when loading fine-tuned BERT for ec."
                print(f"Loading a fine-tuned BERT model from {options.bert_ec_ft_dir}")
                # Load the fine-tuned model
                tokenizer = BertTokenizer.from_pretrained(options.bert_ec_ft_dir, do_lower_case=options.do_lower_case,
                                                          cache_dir=options.bert_cache_dir)
                bert_model = BertModel.from_pretrained(options.bert_ec_ft_dir, cache_dir=options.bert_cache_dir)
            else:
                assert options.bert_ft_dir != "", f"ERROR: --bert_ft_dir must be provided when loading fine-tuned BERT."
                print(f"Loading a fine-tuned BERT model from {options.bert_ft_dir}")
                # Load the fine-tuned model
                tokenizer = BertTokenizer.from_pretrained(options.bert_ft_dir, do_lower_case=options.do_lower_case,
                                                          cache_dir=options.bert_cache_dir)
                bert_model = BertModel.from_pretrained(options.bert_ft_dir, cache_dir=options.bert_cache_dir)

        else:
            tokenizer = BertTokenizer.from_pretrained(options.bert_model, do_lower_case=options.do_lower_case,
                                                      cache_dir=options.bert_cache_dir)
            bert_model = BertModel.from_pretrained(options.bert_model, cache_dir=options.bert_cache_dir)
        print(f"Placing BERT on {bert_device} and setting to eval")
        bert_model.to(bert_device)
        bert_model.eval()

        bert_layers = [int(x) for x in options.bert_layers.split(",")]
        if options.bert_model == "bert-large-cased":
            bert_emb_size = 1024*len(bert_layers)
        elif options.bert_model == "bert-base-cased" or options.bert_model == "bert-base-uncased":
            bert_emb_size = 768*len(bert_layers)
        else:
            assert False, "Not implemented"
    else:
        print("Not using BERT model")
        bert_device = None
        tokenizer = None
        bert_model = None
        bert_layers = None
        bert_emb_size = 0

    if options.mode.strip() == "train" or options.mode.strip() == "fine_tune_error_correction":
        print(f"Training mode")
        # training_file, dev_file contain the final, level 3 gold
        # The level 3 gold training set is used to build the vocabulary. This includes all applicable prefix labels.
        if options.only_ec:
            data = utils_ec.read_ec_data_bert(max_length, options.training_ec_file, options.dev_ec_file, "", None, None, None, data_formatter, tokenizer, options.input_is_untokenized)
        else:
            data = utils.read_metric_data(training_file, dev_file, "", None, None, None, data_formatter, tokenizer, options.input_is_untokenized)
        if use_existing_vocab_file or options.mode.strip() == "fine_tune_error_correction" or options.continue_training or options.only_ec:
            print(f"Using the existing vocab at {vocab_file}")
            vocab, word_to_idx, idx_to_word = utils.load_vocab(vocab_file)
        else:
            # the vocabulary is constructed from both the claims and the level 3 titles (appearing in ground-truth training)
            # and all of the prefix strings
            prefix_tokens_for_vocab = utils.init_prefix_tokens_for_vocab(len(data["train_x"]+data["train_y"]), tokenizer)
            vocab, word_to_idx, idx_to_word = utils.get_vocab(data["train_x"]+data["train_y"]+prefix_tokens_for_vocab, max_vocab_size)
            utils.save_vocab(vocab_file, vocab, word_to_idx)

        if options.only_ec:
            # note the max_length*3 for this level can now be modified via options.ec_max_length
            data = utils_ec.init_ec_data_structure_for_split(options.ec_max_length, data, "train", word_to_idx, max_length, tokenizer,
                                             options.train_decision_labels_file, np_random_state, options.only_2_class or options.eval_symmetric_data)
            data = utils_ec.init_ec_data_structure_for_split(options.ec_max_length, data, "dev", word_to_idx, max_length,
                                                             tokenizer,
                                                             options.dev_decision_labels_file, np_random_state, options.only_2_class or options.eval_symmetric_data)

        else:
            data = utils.init_data_structure_for_split(data, "train", word_to_idx, max_length, tokenizer,
                                                       options.train_covered_titles_file, options.train_true_titles_file,
                                                       options.train_decision_labels_file,
                                                       options.train_chosen_sentences_only_evidence_file,
                                                       options.train_covered_sentences_dictionary_file, options)
            data = utils.init_data_structure_for_split(data, "dev", word_to_idx, max_length, tokenizer,
                                                       options.dev_covered_titles_file, options.dev_true_titles_file,
                                                       options.dev_decision_labels_file,
                                                       options.dev_chosen_sentences_only_evidence_file,
                                                       options.dev_covered_sentences_dictionary_file, options)

        data = utils.update_memory_parameters(data, options)

        if not options.only_ec:
            split_mode = "train"
            print(f'Number of instances in {split_mode}: {len(data[f"level{3}_idx_{split_mode}_x"])}')
            print(f'Number of unique chosen titles in {split_mode}: {len(data[f"{split_mode}_all_titles2freq"])}')
            print(f'Total number of unique covered titles in {split_mode}: {len(data[f"{split_mode}_idx_unique_titles"])}')

            split_mode = "dev"
            print(f'Number of instances in {split_mode}: {len(data[f"level{3}_idx_{split_mode}_x"])}')
            print(f'Number of unique chosen titles in {split_mode}: {len(data[f"{split_mode}_all_titles2freq"])}')
            print(f'Total number of unique covered titles in {split_mode}: {len(data[f"{split_mode}_idx_unique_titles"])}')

    else:
        print(f"Test mode")
        if options.only_ec:
            data = utils_ec.read_ec_data_bert(max_length, "", "", options.test_ec_file, None, None, None, data_formatter, tokenizer, options.input_is_untokenized)
        else:
            data = utils.read_metric_data("", "", test_file, None, None, None, data_formatter, tokenizer, options.input_is_untokenized)
        vocab, word_to_idx, idx_to_word = utils.load_vocab(vocab_file)

        split_mode = "test"
        if options.only_ec:
            # note the max_length*3 for this level
            data = utils_ec.init_ec_data_structure_for_split(options.ec_max_length, data, split_mode, word_to_idx, max_length, tokenizer,
                                             options.test_decision_labels_file, np_random_state, options.only_2_class or options.eval_symmetric_data)
        else:
            data = utils.init_data_structure_for_split(data, split_mode, word_to_idx, max_length, tokenizer,
                                                       options.test_covered_titles_file, options.test_true_titles_file,
                                                       options.test_decision_labels_file,
                                                       options.test_chosen_sentences_only_evidence_file,
                                                       options.test_covered_sentences_dictionary_file, options)

        data = utils.update_memory_parameters(data, options)

        #assert data["top_k_nearest_memories"] > 1, f"ERROR: In this version, at least k==2 is needed to retrieve nearest wrong title in all cases."
        if not options.only_ec:
            print(f'Number of instances in {split_mode}: {len(data[f"level{3}_idx_{split_mode}_x"])}')
            print(f'Number of unique chosen titles in {split_mode}: {len(data[f"{split_mode}_all_titles2freq"])}')
            print(f'Total number of unique covered titles in {split_mode}: {len(data[f"{split_mode}_idx_unique_titles"])}')


    data["vocab"] = vocab
    if options.only_2_class:
        data["classes"] = [0, 1]
    else:
        data["classes"] = [0, 1, 2]
    data["word_to_idx"] = word_to_idx
    data["idx_to_word"] = idx_to_word

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "max_length": max_length,
        "total_length": max_length + 2*constants.PADDING_SIZE,
        "BATCH_SIZE": options.batch_size,
        "WORD_DIM": options.word_embedding_size,
        "vocab_size": len(data["vocab"]),  # note this includes padding and unk
        "CLASS_SIZE": len(data["classes"]),
        "padding_idx": constants.PAD_SYM_ID,
        "FILTERS": filter_widths, #[3, 4, 5],
        "FILTER_NUM": number_of_filter_maps, #[100, 100, 100],
        "DROPOUT_PROB": options.dropout_probability, #0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "main_device": main_device,
        "word_embeddings_file": options.word_embeddings_file,
        "save_dir": options.save_dir,
        "score_vals_file": options.score_vals_file.strip(),
        "word_embeddings_file_in_plaintext": word_embeddings_file_in_plaintext,
        "bert_layers": bert_layers,
        "bert_emb_size": bert_emb_size,
        "forward_type_description": options.forward_type_description,
        "retrieval_forward_type_description": options.retrieval_forward_type_description,
        "only_consider_negative_pairs": options.only_consider_negative_pairs,
        "error_correction_forward_type_description": options.error_correction_forward_type_description,
        "use_auto_encoder_loss": options.use_auto_encoder_loss,
        "fine_tune_bert": options.fine_tune_bert,
        "bert_learning_rate": options.bert_learning_rate,
        "bert_num_train_epochs": options.bert_num_train_epochs,
        "bert_warmup_proportion": options.bert_warmup_proportion,
        "freeze_bert_after_epoch_num": options.freeze_bert_after_epoch_num,
        "bert_ft_dir": options.bert_ft_dir.strip(),
        "bert_ft_aux_dir": options.bert_ft_aux_dir.strip(),
        "bert_ec_ft_dir": options.bert_ec_ft_dir.strip(),
        "do_not_save_detailed_scores_output": options.do_not_save_detailed_scores_output,
        "only_2_class": options.only_2_class,
        "ec_max_length": options.ec_max_length,
        "create_hard_negative_for_unverifiable_retrieval_in_level2": options.create_hard_negative_for_unverifiable_retrieval_in_level2
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("vocab_size:", params["vocab_size"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print(f"Dropout probability: {params['DROPOUT_PROB']}")
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode.strip() == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        if options.fine_tune_bert:
            print(f"Fine-tuning BERT")
            if data["only_ec"]:
                assert params["bert_ec_ft_dir"] != "", f"ERROR: --bert_ec_ft_dir must be provided when fine-tuning BERT for ec."
            else:
                assert params["bert_ft_dir"] != "", f"ERROR: --bert_ft_dir must be provided when fine-tuning BERT."
                assert params["bert_ft_aux_dir"] != "", f"ERROR: --bert_ft_aux_dir must be provided when fine-tuning BERT."

        else:
            print(f"Freezing BERT across all epochs")
            # Note that utils_ft.train_ft() now handles non-ft training, as well

        model = None
        if data["continue_training"]:
            if params["GPU"] != -1:
                model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))  # .cuda(params["GPU"])
            else:
                model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))
        if data["only_ec"]:

            if model.DROPOUT_PROB != params['DROPOUT_PROB']:
                print(
                    f"Note: The dropout probability differs from the original model. Switching from {model.DROPOUT_PROB} (of the original model) to {params['DROPOUT_PROB']} for fine-tuning.")
                model.DROPOUT_PROB = params['DROPOUT_PROB']
            if options.ec_model_update_key == 1:
                print(f"Updating the EC fc to size 4 and placing on {params['main_device']}.")
                model.fc = nn.Linear(sum(model.FILTER_NUM), 4).to(params["main_device"])
            elif options.ec_model_update_key == 2:
                print(f"Updating the EC fc to size 2 and placing on {params['main_device']}.")
                model.fc = nn.Linear(sum(model.FILTER_NUM), 2).to(params["main_device"])
            elif options.ec_model_update_key == 3:
                assert False

            print(f"EC max length of {params['ec_max_length']}")

            utils_ft_only_ec.train(data, params, np_random_state, bert_model, tokenizer, bert_device, only_save_best_models=options.only_save_best_models, model=model)

        else:
            print(f"Marginalizing over evidence in level 3, so no level 3 search is performed during training.")
            utils_ft_train.train_ft(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                    options.only_save_best_models, model=model)

        print("=" * 20 + "TRAINING COMPLETED" + "=" * 20)

    elif options.mode.strip() == "test":
        if params["GPU"] != -1:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))  # .cuda(params["GPU"])
        else:
            model = utils.load_model_torch(options.saved_model_file, int(params["GPU"]))

        pdist = nn.PairwiseDistance(p=2)

        print(f"------------------EVAL (test) STARTING------------------")
        if data["only_ec"]:
            assert False, f"NOT IMPLEMENTED"
        levels_to_consider = [1, 2, 3]
        if data["only_levels_1_and_2"]:
            levels_to_consider = [1, 2]
        elif data["only_level_1"]:
            levels_to_consider = [1]

        predicted_output = {}
        for level_id in levels_to_consider:
            utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title", split_mode="test", level_id=level_id)
            utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="retrieve", split_mode="test", level_id=level_id)
            data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist, data, model, params,
                                                          save_eval_output=True, mode="test", level_id=level_id)
        print(f"------------------EVAL (test) COMPLETE------------------")

        if data["create_exemplar_database"] or data["create_exemplar_query"] or data["save_exemplar_output"]:
            print(f"------------------Starting exemplar processing------------------")
            exemplar_output = utils_exemplar.exemplars_main(data, params, np_random_state, bert_model, tokenizer, bert_device, model,
                                          predicted_output)
            if exemplar_output is not None:
                print("Saving exemplar format file for level 3")
                scores_file_name = f"{options.score_vals_file}.eval.level{3}.exemplar_format.txt"
                utils.save_lines(scores_file_name, exemplar_output[f"level{3}_{'test'}_score_vals_compact"])
                print(f"Saved exemplar format file for level 3: {scores_file_name}")
            print(f"------------------Completed exemplar processing------------------")
        elif data["visualize_alignment"]:
            print(f"------------------Starting visualization------------------")
            # TODO: We can detokenize and save to HTML as with BLADE
            viz_output = utils_alignment_visualization.visualization_main(data, params, np_random_state, bert_model, tokenizer, bert_device, model,
                                          predicted_output)

            print(f"------------------Completed visualization------------------")
        else:
            if options.do_not_save_test_eval_output:
                print(f"Not saving test eval output due to --do_not_save_test_eval_output.")
            else:
                if data["eval_constrained"]:
                    assert False, f"ERROR: --eval_constrained option is not expected to be used in this version when" \
                                  f" saving output."
                if data["save_output_for_ec"]:
                    print("Saving ec format file for level 3")
                    scores_file_name = f"{options.score_vals_file}.eval.level{3}.ec_format.txt"
                    utils.save_lines(scores_file_name, predicted_output[f"level{3}_{'test'}_score_vals_compact"])
                    print(f"Saved ec format file for level 3: {scores_file_name}")
                else:
                    print("Saving scores file")
                    for level_id in levels_to_consider:
                        scores_file_name = f"{options.score_vals_file}.eval.level{level_id}.compact.txt"
                        utils.save_lines(scores_file_name, predicted_output[f"level{level_id}_{'test'}_score_vals_compact"])
                        print(f"Saved compact scores file: {scores_file_name}")


if __name__ == "__main__":
    main()

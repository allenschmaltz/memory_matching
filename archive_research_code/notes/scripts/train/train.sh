################################################################################
#### 1. Start training with k_1=10.
#### 2. Continue training with k_1=30.
################################################################################

################################################################################
#### 1. Start training with k_1=10.
#### I've added some in-line comments below with additional information.
################################################################################

# trained with 1 V100 (32GB); max 42GB system memory; for around 3 days

SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}/archive_research_code/main

GPU_IDS=0

VOCAB_SIZE=7500  # this is only for the randomly initialized embeddings as part of input to the memory layers; BERT otherwise has access to its full WordPiece vocab
FILTER_NUMS=1000  # each CNN has 1000 filters of kernel width 1
DROPOUT_PROB=0.0
# max len is the max len for level 1; max length for level 2 is MAX_SENT_LEN*2; max length for level 3 is MAX_SENT_LEN*3
MAX_SENT_LEN=50

#PREPROCESS_VERSION="version8_3way_debug_set_of_size2000"  # uncomment for the smaller debugging datasets
#PREPROCESS_VERSION="version8_3way_debug_set_of_size100"
PREPROCESS_VERSION="version8_3way"  # this is our internal label for this version of the preprocessing
EXPERIMENT_LABEL=fever_${PREPROCESS_VERSION}_unicnnbert_v${VOCAB_SIZE}_bertbasecased_top1layer_fn${FILTER_NUMS}_dout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}_lr2e5_6ftepochs_max2evidence_levels123

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/fever/${PREPROCESS_VERSION}/${EXPERIMENT_LABEL}"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/fever/${PREPROCESS_VERSION}/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

# the following 2 directories should have at least 50GB of free space and need to be fast scratch space:
# These will be used to cache the memory vectors every traversal through the search graph.
MODEL_MEMORY_DIR=${MODEL_DIR}/memory  # this directory stores the memory vectors for the support sequences
MODEL_RETRIEVAL_MEMORY_DIR=${MODEL_DIR}/memory_retrieval  # this directory stores the memory vectors for the query sequences

# These directories are used to save the Transformer parameters: The current running best retrieval parameters in BERT_FT_DIR
# and the current best classification parameters in BERT_AUX_FT_DIR. In practice, we only care about BERT_AUX_FT_DIR; the other is only for analysis.
# In practice, in our training, the final epochs are the same, but some care is needed at inference to choose the right directory if they diverge.
# Note that for convenience, we save the memory (CNN) parameters separately in MODEL_DIR.
# Note that with those parameters, as well, we make the distinction of the best retrieval and best classification parameters.
# The highest retreival accuracy memory (CNN) parameters are saved to aesw_rand_max_dev_level2_retrieval_acc_epoch.pt and the highest
# accuracy classification parameters are saved to aesw_rand_max_dev_level3_decision_acc_epoch.pt.
# At inference, it's important to choose the matching Transformer and memory parameters.
BERT_FT_DIR=${MODEL_DIR}/bert_retrieval_ft
BERT_AUX_FT_DIR=${MODEL_DIR}/bert_decision_ft

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
#NUM_LAYERS='-1,-2,-3,-4'
NUM_LAYERS='-1'  # only using the top layer of BERT

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/fever"  # this is the main data directory; adjust accordingly

INPUT_TRAIN_FILE="train.jsonl"
INPUT_DEV_FILE="shared_task_dev.jsonl"

#DEBUG_SET_SIZE=100
#DEBUG_SET_SIZE=2000
#DEBUG_SUFFIX=".head${DEBUG_SET_SIZE}.txt"  # uncomment the suffixes to use the debug subsets
DEBUG_SUFFIX=""
TRAIN_DATA_DIR=${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat
TRAIN_DATA_INPUT_NAME="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt"${DEBUG_SUFFIX}
DEV_DATA_DIR=${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat
DEV_DATA_INPUT_NAME="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt"${DEBUG_SUFFIX}

TRAIN_TRUE_SENTS_FILE="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl"${DEBUG_SUFFIX}
TRAIN_COVERED_SENTS_FILE="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt"${DEBUG_SUFFIX}

DEV_TRUE_SENTS_FILE="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl"${DEBUG_SUFFIX}
DEV_COVERED_SENTS_FILE="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt"${DEBUG_SUFFIX}

TRAIN_DECISION_LABELS_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.control.txt"${DEBUG_SUFFIX}
DEV_DECISION_LABELS_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.control.txt"${DEBUG_SUFFIX}

TRAIN_CHOSEN_SENTS_ONLY_EVIDENCE_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt"${DEBUG_SUFFIX}
DEV_CHOSEN_SENTS_ONLY_EVIDENCE_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt"${DEBUG_SUFFIX}

TRAIN_COVERED_SENTS_DICT_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt"${DEBUG_SUFFIX}
DEV_COVERED_SENTS_DICT_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt"${DEBUG_SUFFIX}

# INIT the following before running:
mkdir -p "${MODEL_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir "${OUTPUT_LOG_DIR}"
mkdir "${MODEL_MEMORY_DIR}"
mkdir "${MODEL_RETRIEVAL_MEMORY_DIR}"
mkdir "${BERT_FT_DIR}"
mkdir "${BERT_AUX_FT_DIR}"
# echo "${OUTPUT_LOG_DIR}"


# TODO: UPDATE --bert_cache_dir with a suitable PATH, if you already have a local copy of BERT_base
# Aside: As noted in the README, this is from my research codebase and there are now some zombie args (e.g., --error_correction_forward_type_description); I'll remove those in the final version to appear in the main directory.
CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u memory_match.py \
--mode "train" \
--model "rand" \
--dataset "aesw" \
--word_embeddings_file "not_currently_used_since_embeddings_are_randomly_initialized" \
--training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
--dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \
--train_true_titles_file ${TRAIN_DATA_DIR}/"${TRAIN_TRUE_SENTS_FILE}" \
--dev_true_titles_file ${DEV_DATA_DIR}/"${DEV_TRUE_SENTS_FILE}" \
--train_covered_titles_file ${TRAIN_DATA_DIR}/"${TRAIN_COVERED_SENTS_FILE}" \
--dev_covered_titles_file ${DEV_DATA_DIR}/"${DEV_COVERED_SENTS_FILE}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}.txt \
--save_model \
--epoch 15 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${MODEL_DIR}" \
--score_vals_file "${OUTPUT_DIR}"/train.dev_score_vals.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/bert_cache_dir/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB}  \
--input_is_untokenized \
--only_save_best_models \
--forward_type_description "maxpool_no_relu_no_dropout" \
--retrieval_forward_type_description "maxpool_no_relu_no_dropout" \
--error_correction_forward_type_description "error_correction_no_relu_dropout" \
--word_embedding_size 300 \
--use_auto_encoder_loss \
--titles_memory_dir ${MODEL_MEMORY_DIR} \
--retrieval_memory_dir ${MODEL_RETRIEVAL_MEMORY_DIR} \
--fine_tune_bert \
--bert_learning_rate 2e-5 \
--bert_num_train_epochs 6.0 \
--bert_warmup_proportion 0.1 \
--freeze_bert_after_epoch_num 6 \
--bert_ft_dir "${BERT_FT_DIR}" \
--bert_ft_aux_dir "${BERT_AUX_FT_DIR}" \
--do_not_save_detailed_scores_output \
--train_decision_labels_file ${TRAIN_DECISION_LABELS_FILE} \
--dev_decision_labels_file ${DEV_DECISION_LABELS_FILE} \
--train_chosen_sentences_only_evidence_file ${TRAIN_CHOSEN_SENTS_ONLY_EVIDENCE_FILE} \
--dev_chosen_sentences_only_evidence_file ${DEV_CHOSEN_SENTS_ONLY_EVIDENCE_FILE} \
--train_covered_sentences_dictionary_file ${TRAIN_COVERED_SENTS_DICT_FILE} \
--dev_covered_sentences_dictionary_file ${DEV_COVERED_SENTS_DICT_FILE} \
--batch_size 9 \
--level1_top_k_nearest_memories 10 \
--level2_top_k_nearest_memories 3 \
--level3_top_k_nearest_memories 3 \
--level1_memory_batch_size 2500 \
--level1_retrieval_batch_size 2500 \
--level1_titles_chunk_size 50000 \
--level1_retrieval_chunk_size 25000 \
--level2_memory_batch_size 1200 \
--level2_retrieval_batch_size 1200 \
--level2_titles_chunk_size 24000 \
--level2_retrieval_chunk_size 24000 \
--level3_memory_batch_size 1200 \
--level3_retrieval_batch_size 1200 \
--level3_titles_chunk_size 4800 \
--level3_retrieval_chunk_size 4800 \
--level3_top_k_evidence_predictions 3


################################################################################
#### 2. Continue training with k_1=30.
#### In our submitted model, this starts from epoch 13. Note that due to the
#### larger beam size, we also need to allocate additional system memory. In
#### practice, we found that 56GB was sufficient, as opposed to 42 GB for
#### k_1=10. In practice, the highest epoch occurs relatively early, at
#### epoch 2. Note that in the example here, some of the saved model files
#### from the initial run will be overwritten. If you want to save them, either
#### make a copy or change the paths.
################################################################################

# # Main changes from initial run:
# --level1_top_k_nearest_memories 30 \
# --continue_training \
# --load_ft_bert \
# --saved_model_file="${MODEL_DIR}/aesw_rand_max_dev_level2_retrieval_acc_epoch.pt"
#
# In particular, note that --saved_model_file appears to be the best retrieval epoch (as opposed to aesw_rand_max_dev_level3_decision_acc_epoch.pt), but in training in our submitted model, the best retrieval and classification accuracies occurred in the same epoch, so the parameters are the same in both files. Importantly, note that the Transformer parameters are read from --bert_ft_dir in this re-training setting of levels 1-3. I.e., $BERT_FT_DIR needs to contain the Transformer parameters from which to continue training.

SERVER_DRIVE_PATH_PREFIX="UPDATE_WITH_YOUR_PATH"

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}/archive_research_code/main

GPU_IDS=0

VOCAB_SIZE=7500
FILTER_NUMS=1000
DROPOUT_PROB=0.0

MAX_SENT_LEN=50


#PREPROCESS_VERSION="version8_3way_debug_set_of_size2000"
#PREPROCESS_VERSION="version8_3way_debug_set_of_size100"
PREPROCESS_VERSION="version8_3way"
EXPERIMENT_LABEL=fever_${PREPROCESS_VERSION}_unicnnbert_v${VOCAB_SIZE}_bertbasecased_top1layer_fn${FILTER_NUMS}_dout${DROPOUT_PROB}_maxlen${MAX_SENT_LEN}_lr2e5_6ftepochs_max2evidence_levels123

MODEL_DIR="${SERVER_DRIVE_PATH_PREFIX}/models/fever/${PREPROCESS_VERSION}/${EXPERIMENT_LABEL}"
OUTPUT_DIR="${SERVER_DRIVE_PATH_PREFIX}/output/fever/${PREPROCESS_VERSION}/${EXPERIMENT_LABEL}"
OUTPUT_LOG_DIR="${OUTPUT_DIR}"/logs

MODEL_MEMORY_DIR=${MODEL_DIR}/memory
MODEL_RETRIEVAL_MEMORY_DIR=${MODEL_DIR}/memory_retrieval

BERT_FT_DIR=${MODEL_DIR}/bert_retrieval_ft
BERT_AUX_FT_DIR=${MODEL_DIR}/bert_decision_ft

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
#NUM_LAYERS='-1,-2,-3,-4'
NUM_LAYERS='-1'

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/fever"

INPUT_TRAIN_FILE="train.jsonl"
INPUT_DEV_FILE="shared_task_dev.jsonl"

#DEBUG_SET_SIZE=100
#DEBUG_SET_SIZE=2000
#DEBUG_SUFFIX=".head${DEBUG_SET_SIZE}.txt"
DEBUG_SUFFIX=""
TRAIN_DATA_DIR=${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat
TRAIN_DATA_INPUT_NAME="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt"${DEBUG_SUFFIX}
DEV_DATA_DIR=${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat
DEV_DATA_INPUT_NAME="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt"${DEBUG_SUFFIX}

TRAIN_TRUE_SENTS_FILE="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl"${DEBUG_SUFFIX}
TRAIN_COVERED_SENTS_FILE="${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt"${DEBUG_SUFFIX}

DEV_TRUE_SENTS_FILE="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl"${DEBUG_SUFFIX}
DEV_COVERED_SENTS_FILE="${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt"${DEBUG_SUFFIX}

TRAIN_DECISION_LABELS_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.control.txt"${DEBUG_SUFFIX}
DEV_DECISION_LABELS_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.control.txt"${DEBUG_SUFFIX}

TRAIN_CHOSEN_SENTS_ONLY_EVIDENCE_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt"${DEBUG_SUFFIX}
DEV_CHOSEN_SENTS_ONLY_EVIDENCE_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt"${DEBUG_SUFFIX}

TRAIN_COVERED_SENTS_DICT_FILE=${TRAIN_DATA_DIR}/"${INPUT_TRAIN_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt"${DEBUG_SUFFIX}
DEV_COVERED_SENTS_DICT_FILE=${DEV_DATA_DIR}/"${INPUT_DEV_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt"${DEBUG_SUFFIX}

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u memory_match.py \
--mode "train" \
--model "rand" \
--dataset "aesw" \
--word_embeddings_file "not_currently_used_since_embeddings_are_randomly_initialized_and_here_we_are_continuing_training" \
--training_file ${TRAIN_DATA_DIR}/"${TRAIN_DATA_INPUT_NAME}" \
--dev_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \
--train_true_titles_file ${TRAIN_DATA_DIR}/"${TRAIN_TRUE_SENTS_FILE}" \
--dev_true_titles_file ${DEV_DATA_DIR}/"${DEV_TRUE_SENTS_FILE}" \
--train_covered_titles_file ${TRAIN_DATA_DIR}/"${TRAIN_COVERED_SENTS_FILE}" \
--dev_covered_titles_file ${DEV_DATA_DIR}/"${DEV_COVERED_SENTS_FILE}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}.txt \
--save_model \
--epoch 15 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${MODEL_DIR}" \
--score_vals_file "${OUTPUT_DIR}"/train.dev_score_vals.txt \
--bert_cache_dir "${SERVER_DRIVE_PATH_PREFIX}/bert_cache_dir/pytorch_pretrained_bert" \
--bert_model=${BERT_MODEL} \
--bert_layers=${NUM_LAYERS} \
--bert_gpu 0 \
--filter_widths="1" \
--number_of_filter_maps="${FILTER_NUMS}" \
--dropout_probability ${DROPOUT_PROB}  \
--input_is_untokenized \
--only_save_best_models \
--forward_type_description "maxpool_no_relu_no_dropout" \
--retrieval_forward_type_description "maxpool_no_relu_no_dropout" \
--error_correction_forward_type_description "error_correction_no_relu_dropout" \
--word_embedding_size 300 \
--use_auto_encoder_loss \
--titles_memory_dir ${MODEL_MEMORY_DIR} \
--retrieval_memory_dir ${MODEL_RETRIEVAL_MEMORY_DIR} \
--fine_tune_bert \
--bert_learning_rate 2e-5 \
--bert_num_train_epochs 6.0 \
--bert_warmup_proportion 0.1 \
--freeze_bert_after_epoch_num 6 \
--bert_ft_dir "${BERT_FT_DIR}" \
--bert_ft_aux_dir "${BERT_AUX_FT_DIR}" \
--do_not_save_detailed_scores_output \
--train_decision_labels_file ${TRAIN_DECISION_LABELS_FILE} \
--dev_decision_labels_file ${DEV_DECISION_LABELS_FILE} \
--train_chosen_sentences_only_evidence_file ${TRAIN_CHOSEN_SENTS_ONLY_EVIDENCE_FILE} \
--dev_chosen_sentences_only_evidence_file ${DEV_CHOSEN_SENTS_ONLY_EVIDENCE_FILE} \
--train_covered_sentences_dictionary_file ${TRAIN_COVERED_SENTS_DICT_FILE} \
--dev_covered_sentences_dictionary_file ${DEV_COVERED_SENTS_DICT_FILE} \
--batch_size 9 \
--level1_top_k_nearest_memories 30 \
--level2_top_k_nearest_memories 3 \
--level3_top_k_nearest_memories 3 \
--level1_memory_batch_size 2500 \
--level1_retrieval_batch_size 2500 \
--level1_titles_chunk_size 50000 \
--level1_retrieval_chunk_size 25000 \
--level2_memory_batch_size 1200 \
--level2_retrieval_batch_size 1200 \
--level2_titles_chunk_size 24000 \
--level2_retrieval_chunk_size 24000 \
--level3_memory_batch_size 1200 \
--level3_retrieval_batch_size 1200 \
--level3_titles_chunk_size 4800 \
--level3_retrieval_chunk_size 4800 \
--level3_top_k_evidence_predictions 3 \
--continue_training \
--load_ft_bert \
--saved_model_file="${MODEL_DIR}/aesw_rand_max_dev_level2_retrieval_acc_epoch.pt"



####### as with the initial run, these need to exist before running:
# mkdir -p "${MODEL_DIR}"
# mkdir -p "${OUTPUT_DIR}"
# mkdir "${OUTPUT_LOG_DIR}"
# mkdir "${MODEL_MEMORY_DIR}"
# mkdir "${MODEL_RETRIEVAL_MEMORY_DIR}"
# mkdir "${BERT_FT_DIR}"
# mkdir "${BERT_AUX_FT_DIR}"
# echo "${OUTPUT_LOG_DIR}"

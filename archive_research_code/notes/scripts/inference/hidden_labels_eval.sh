################################################################################
#### 1. Inference on dev/test with hidden labels, with z=3, k_1=100.
#### 2. Convert to CodaLab submission format.
################################################################################

################################################################################
#### 1. Inference on dev/test with hidden labels, with z=3, k_1=100.
################################################################################

# run with 1 V100 (32GB); beam size k_1=100 takes less than an hour

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

# The following should typically already exist from training:
# mkdir -p "${MODEL_DIR}"
# mkdir -p "${OUTPUT_DIR}"
# mkdir "${OUTPUT_LOG_DIR}"
# mkdir "${MODEL_MEMORY_DIR}"
# mkdir "${MODEL_RETRIEVAL_MEMORY_DIR}"
# mkdir "${BERT_FT_DIR}"
# mkdir "${BERT_AUX_FT_DIR}"
# echo "${OUTPUT_LOG_DIR}"

BERT_MODEL="bert-base-cased"
#BERT_MODEL="bert-large-cased"
#NUM_LAYERS='-1,-2,-3,-4'
NUM_LAYERS='-1'

DATA_DIR="${SERVER_DRIVE_PATH_PREFIX}/data/fever"  # TODO: UPDATE DATA PATH AS NECESSARY

INPUT_TRAIN_FILE="train.jsonl"

echo "NOTE, here we are evaluating on the blind dev or test"
#INPUT_DEV_FILE="shared_task_dev_public.jsonl"  # uncomment this, and comment the next line, to run on the dev set -- this is only for checking the format (otherwise, we have the labels for dev, as with the examples in notes/scripts/inference/create_k1_z_vs_acc_table.sh)
INPUT_DEV_FILE="shared_task_test.jsonl"


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

# Note: --bert_ft_dir must correspond to --saved_model_file
# See additional comments in notes/scripts/inference/create_k1_z_vs_acc_table.sh regarding the model file names

K_SIZE=100
EVAL_LABEL="k${K_SIZE}k3_topk_evidence3_print5_include_retrieval_distances_blind_inference"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -u memory_match.py \
--mode "test" \
--model "rand" \
--dataset "aesw" \
--word_embeddings_file "not_used" \
--test_file ${DEV_DATA_DIR}/"${DEV_DATA_INPUT_NAME}" \
--test_true_titles_file ${DEV_DATA_DIR}/"${DEV_TRUE_SENTS_FILE}" \
--test_covered_titles_file ${DEV_DATA_DIR}/"${DEV_COVERED_SENTS_FILE}" \
--max_length ${MAX_SENT_LEN} \
--max_vocab_size ${VOCAB_SIZE} \
--vocab_file ${MODEL_DIR}/vocab${VOCAB_SIZE}.txt \
--epoch 60 \
--learning_rate 1.0 \
--gpu 0 \
--save_dir="${MODEL_DIR}" \
--score_vals_file "${OUTPUT_DIR}"/${INPUT_DEV_FILE}.ec_format.${EVAL_LABEL} \
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
--do_not_save_detailed_scores_output \
--test_decision_labels_file ${DEV_DECISION_LABELS_FILE} \
--test_chosen_sentences_only_evidence_file ${DEV_CHOSEN_SENTS_ONLY_EVIDENCE_FILE} \
--test_covered_sentences_dictionary_file ${DEV_COVERED_SENTS_DICT_FILE} \
--batch_size 9 \
--level1_top_k_nearest_memories ${K_SIZE} \
--level2_top_k_nearest_memories 5 \
--level3_top_k_nearest_memories 5 \
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
--level3_top_k_evidence_predictions 5 \
--load_ft_bert \
--bert_ft_dir "${BERT_FT_DIR}" \
--saved_model_file="${MODEL_DIR}/aesw_rand_max_dev_level2_retrieval_acc_epoch.pt" \
--save_output_for_ec >"${OUTPUT_LOG_DIR}"/${INPUT_DEV_FILE}.ec_format.log.r1.${EVAL_LABEL}.txt


################################################################################
#### 2. Convert to CodaLab submission format.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way"
INPUT_DATA_DIR="${DATA_DIR}"  # TODO: UPDATE, as necessary. This assumes the file directory structure used in the provided data preprocessing scripts.
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"

#INPUT_FILE="shared_task_dev_public.jsonl"
INPUT_FILE="shared_task_test.jsonl"

OUTPUT_DIR=UPDATE_WITH_YOUR_PATH  # this should be "${OUTPUT_DIR}" above given to the --score_vals_file arg

# dev, z=5, k_1=100
#PREDICTION_FILE=${OUTPUT_DIR}/shared_task_dev_public.jsonl.ec_format.k100k3_topk_evidence3_print5_include_retrieval_distances_blind_inference.eval.level3.ec_format.txt

# test, z=5, k_1=100
PREDICTION_FILE=${OUTPUT_DIR}/shared_task_test.jsonl.ec_format.k100k3_topk_evidence3_print5_include_retrieval_distances_blind_inference.eval.level3.ec_format.txt  # TODO: adjust, accordingly, if you used a different file name

OUTPUT_CODALAB_DIR=${OUTPUT_DIR}/${INPUT_FILE}
mkdir ${OUTPUT_CODALAB_DIR}

python -u ${REPO_DIR}/archive_research_code/data/eval/fever_data_version8_3class_convert_to_codalab_eval_format.py \
--input_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--input_ec_file ${PREDICTION_FILE} \
--control_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.control.txt" \
--filtered_titles_to_title_hashes_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.title_hashes.txt" \
--output_codalab_jsonl_dir ${OUTPUT_CODALAB_DIR}

# For submission to CodaLab, the file should be called "predictions.jsonl". Additionally, compress (with zip) the file before uploading.

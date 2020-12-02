################################################################################
#### 1. Format symmetric data -- symmetric_v0.2 format: new dev, test split
#### 2. Format symmetric data -- symmetric_v0.1 format: From original paper
#### 3. Format single evidence train and dev
####
#### In all cases, DATA_DIR should be the original FEVER data and
#### INPUT_DATA_DIR includes the applicable split to the Symmetric data.
####
#### Importantly, note that the symmetric data now breaks our original
#### invariant of each evidence sentence being uniquely described by its
#### title and sentence number. The hack to get around this amounts to
#### setting the true evidence to null for eval, which is fine here, since we do
#### not need to produce stats on retrieval effetiveness since retrieval is
#### given. See the special args for memory_match.py. Retrieval distances
#### are unaffected.
####
#### Note that standard out output can be largely ignored here. That was
#### mostly for my reference vis-a-vis checking the data relative to the
#### original FEVER data. I may remove that output in a future version.
####
#### The heuristic to re-match the titles appears not to miss anything
#### substantive -- All of the <1 difference ratios appear to be due to
#### slightly different tokenization (e.g., R&B vs. R & B).
################################################################################

################################################################################
#### 1. Format symmetric data -- symmetric_v0.2 format: new dev, test split
################################################################################


REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever" # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way_symmetric"
INPUT_DATA_DIR="/Users/a/Documents/data/fever_bias_data/FeverSymmetric-master/symmetric_v0.2" # TODO: UPDATE WITH YOUR PATH
OUTPUT_DATA_DIR="${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
mkdir -p "${OUTPUT_DATA_DIR}"

#INPUT_FILE="fever_symmetric_dev.jsonl"
#INPUT_FILE="fever_symmetric_test.jsonl"

for INPUT_FILE in "fever_symmetric_dev.jsonl" "fever_symmetric_test.jsonl";
do
  # Keep in mind for all analysis that the symmetric test set is actually derived
  # from the original dev set. This is fine for the purposes here (since the
  # symmetric instances would then, in principle, be harder in relative terms),
  # but note that we have to be careful in how we use exemplars from the
  # original dev set to avoid contamination (i.e., verbatim/exact matches).
  # Here, we need to re-associate the title and sentence id with the evidence.
  # We also need to detokenize/tokenize to match our training tokenization.
  # We parallel the data structures used for the main data to use the same
  # code, but note that we can no longer use the --output_true_sentences_file
  # .jsonl structures since the title+sent_id is no longer a unique hash, since
  # the symmetric re-annotations modify the body of the sentence, but we
  # re-associate the original title and sentence ids. However, we can just
  # ignore the true sentence .jsonl (and set to [], like unverifiable claims)
  # since we don't need to assess retrieval accuracy -- it is always true here
  # since we are given the correct sentence to match for levels 1 and 2.
  # (Actually, we don't need to run retrieval for the symmetric dataset at all,
  # but we do anyway so that we can analyze the level 1 and level 2 distances.)
  ORIGINAL_JSON_FILE=/Users/a/Documents/data/fever/shared_task_dev.jsonl  # TODO: UPDATE WITH YOUR PATH
  # TODO: UPDATE --bert_cache_dir WITH YOUR PATH
  python -u ${REPO_DIR}/archive_research_code/data/symmetric_preprocessing/fever_symmetric_data_preprocess.py \
  --input_symmetric_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
  --input_jsonl_file "${ORIGINAL_JSON_FILE}" \
  --input_wiki_file UPDATE_WITH_YOUR_PATH/wiki-pages/wiki-pages/processed/wiki_v2/detokenized/all.wiki_v2.txt \
  --output_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.txt" \
  --output_control_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.control.txt" \
  --output_true_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_titles.txt" \
  --output_covered_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_titles.txt" \
  --output_chosen_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt" \
  --output_true_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl" \
  --output_covered_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt" \
  --output_filtered_titles_to_title_hashes_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.title_hashes.txt" \
  --output_chosen_sentences_only_evidence_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt" \
  --output_covered_sentences_dictionary_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt" \
  --bert_cache_dir="/Volumes/Extreme SSD/main/models/bert_cache/started_2020_03_10/" \
  --bert_model=${BERT_MODEL} \
  --size_of_debug_set -1
done

################################################################################
#### 2. Format symmetric data -- symmetric_v0.1 format: From original paper
#### This is the version of the symmetric dataset used in the original
#### "Towards Debiasing Fact Verification Models" paper.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way_symmetric"
INPUT_DATA_DIR="/Users/a/Documents/data/fever_bias_data/FeverSymmetric-master/symmetric_v0.1"  # TODO: UPDATE WITH YOUR PATH
OUTPUT_DATA_DIR="${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
mkdir -p "${OUTPUT_DATA_DIR}"

# This serves as the eval set here. Their original paper does not split
# dev and test. Note that means we cannot use exemplars from the original
# dev, as there may be duplicates.
INPUT_FILE="fever_symmetric_generated.jsonl"

# Also see the comment above for fever_symmetric_data_preprocess.py.
# This script is very similar. As with symmetric_v0.2, we need to re-associate
# the wikipedia title. Note that unlike symmetric_v0.2, symmetric_v0.1 only
# contains (as far as I can tell), new claim-evidence pairs (i.e., 3 new
# instances, and the originals are not in this file). As with symmetric_v0.2,
# it's important to keep in mind that as a result of re-associating the titles
# to article sentences that have now changed, we can no longer use
# the title+sentence as a unique hash to the evidence. This
# breaks some invariants in our train/test code (particularily with regard to
# true evidence sets used for training and eval), but we can get around this
# by using the --eval_symmetric_data in memory_match.py for inference.
# Note that these can't currently be used for training and the retrieval
# eval output should be ignored, but is known (it's always 1).

ORIGINAL_JSON_FILE=/Users/a/Documents/data/fever/shared_task_dev.jsonl  # TODO: UPDATE WITH YOUR PATH
# TODO: UPDATE --bert_cache_dir WITH YOUR PATH
python -u ${REPO_DIR}/archive_research_code/data/symmetric_preprocessing/fever_symmetric_data_preprocess_original_format.py \
--input_symmetric_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--input_jsonl_file "${ORIGINAL_JSON_FILE}" \
--input_wiki_file UPDATE_WITH_YOUR_PATH/wiki-pages/wiki-pages/processed/wiki_v2/detokenized/all.wiki_v2.txt \
--output_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.txt" \
--output_control_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.control.txt" \
--output_true_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_titles.txt" \
--output_covered_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_titles.txt" \
--output_chosen_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt" \
--output_true_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl" \
--output_covered_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt" \
--output_filtered_titles_to_title_hashes_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.title_hashes.txt" \
--output_chosen_sentences_only_evidence_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt" \
--output_covered_sentences_dictionary_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt" \
--bert_cache_dir="/Volumes/Extreme SSD/main/models/bert_cache/started_2020_03_10/" \
--bert_model=${BERT_MODEL} \
--size_of_debug_set -1

################################################################################
#### 3. Format single evidence train and dev
#### Run this for each of dev and train:
# (A.)
# INPUT_FILE="fever.dev.jsonl"
# ORIGINAL_JSON_FILE=UPDATE_WITH_YOUR_PATH/shared_task_dev.jsonl # original
# (B.)
# INPUT_FILE="fever.train.jsonl"
# ORIGINAL_JSON_FILE=UPDATE_WITH_YOUR_PATH/train.jsonl # original
################################################################################

# We also need to process the train and dev from the symmetric paper. These
# are only instances with single evidence sentences. It seems that they
# duplicated claims (creating separate claim-evidence instances) when a claim
# was associated with multiple single evidence sentence sets (e.g., id
# 111897 in fever.dev.jsonl), so we need to be careful when using the id as
# a unique key in these cases. To get around this, we create new hashes for
# the ids by appending the line id to the end of the string
# representation of the original id, separated by '.' with a final 1 digit.

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever" # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way_symmetric"
INPUT_DATA_DIR="/Users/a/Documents/data/fever_bias_data"  # TODO: UPDATE WITH YOUR PATH
OUTPUT_DATA_DIR="${DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
OUTPUT_LOG_DIR="${OUTPUT_DATA_DIR}/logs"
mkdir -p "${OUTPUT_DATA_DIR}"
mkdir -p "${OUTPUT_LOG_DIR}"

# for processing dev, uncomment the next two lines
# INPUT_FILE="fever.dev.jsonl"
# ORIGINAL_JSON_FILE=/Users/a/Documents/data/fever/shared_task_dev.jsonl  # TODO: UPDATE WITH YOUR PATH

# for processing dev, comment the next two lines
INPUT_FILE="fever.train.jsonl"
ORIGINAL_JSON_FILE=/Users/a/Documents/data/fever/train.jsonl  # TODO: UPDATE WITH YOUR PATH

# TODO: UPDATE --bert_cache_dir WITH YOUR PATH
python -u ${REPO_DIR}/archive_research_code/data/symmetric_preprocessing/fever_symmetric_data_preprocess_original_format_traindev.py \
--input_symmetric_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
--input_jsonl_file "${ORIGINAL_JSON_FILE}" \
--input_wiki_file UPDATE_WITH_YOUR_PATH/wiki-pages/wiki-pages/processed/wiki_v2/detokenized/all.wiki_v2.txt \
--output_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.txt" \
--output_control_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.control.txt" \
--output_true_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_titles.txt" \
--output_covered_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_titles.txt" \
--output_chosen_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt" \
--output_true_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl" \
--output_covered_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt" \
--output_filtered_titles_to_title_hashes_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.title_hashes.txt" \
--output_chosen_sentences_only_evidence_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt" \
--output_covered_sentences_dictionary_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt" \
--bert_cache_dir="/Volumes/Extreme SSD/main/models/bert_cache/started_2020_03_10/" \
--bert_model=${BERT_MODEL} \
--size_of_debug_set -1 > "${OUTPUT_LOG_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.data_preprocess.log.txt"

# INPUT_FILE="fever.dev.jsonl"
#     # 7983 SUPPORTS
#     # 8681 REFUTES
#     # 16664 total
#
# INPUT_FILE="fever.train.jsonl"
#     # 100570 SUPPORTS
#     # 41850 REFUTES
#     # 142420 total

################################################################################
#### 1. Format wiki data
#### 2. Format FEVER data -- train and dev with labels
#### 3. (Optional) Format FEVER data -- small debugging sets with labels
#### 4. Format FEVER data -- dev and test without labels
####
#### In some cases, applicable output appears in comments after the
#### command, which you can use as a check. I am also releasing the
#### preprocessed data. (The code here is primarily only of interest to see
#### further details of the preprocessing decisions.)
################################################################################


################################################################################
#### 1. Format wiki data
#### Note that this uses Bert_large cased to filter the data. This is
#### relatively inefficient, taking several hours to complete over all of
#### FEVER Wikipedia, but it only needs to be run once.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="wiki_v2"
INPUT_DATA_DIR="${DATA_DIR}/wiki-pages/wiki-pages"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/processed/${PREPROCESS_VERSION}/detokenized"
mkdir -p "${OUTPUT_DATA_DIR}"
mkdir "${OUTPUT_DATA_DIR}/logs"

# TODO: UPDATE --bert_cache_dir WITH YOUR PATH
python -u ${REPO_DIR}/archive_research_code/data/wikipedia_preprocessing/fever_process_wiki_data_by_sent.py \
--input_dir "${INPUT_DATA_DIR}" \
--output_file "${OUTPUT_DATA_DIR}/all.${PREPROCESS_VERSION}.txt" \
--bert_cache_dir="/Volumes/Extreme SSD/main/models/bert_cache/started_2020_03_10/" \
--bert_model=${BERT_MODEL} > "${OUTPUT_DATA_DIR}/logs/all.${PREPROCESS_VERSION}.log.txt"

# Number of empty lines: 20431
# Number of BERT-filtered tokens: 0
# Wiki: By wordpiece: Mean length: 115.62836497281558; std: 155.11339289967228; min: 1, max: 121866
# Number of sentences per wiki document: Mean: 6.791003920234332; std: 25.77864907682789; min: 1, max: 27120
# Total lines: 5396106

# extra line break:
# /Users/a/Documents/data/fever/wiki-pages/wiki-pages/wiki-031.jsonl
# Electoral_district_of_Talbot_and_Avoca
# Note that there were a number of mis-formats in the underlying wiki data which
# trigger a number of warnings. These are handled by ignoring those segments.
# This is fine, since the line numbers are retained.
# There are 518 WARNING cases in the log file (due to these various inconsistencies in the raw file),
# which is relatively small compared to the total number of articles and sentences.

################################################################################
#### 2. Format FEVER data -- train and dev with labels
#### We need to run the preprocessing script twice, which is done in the loop
#### below for each of:
####  INPUT_FILE="train.jsonl"
####  INPUT_FILE="shared_task_dev.jsonl"
#### Note: Completely uncovered claims very rarely occur, and are either
#### misspellings, or have a suffix (plural/etc.) that precludes a verbatim
#### match. Our simple model-free lexical cover does not handle that,
#### which is sufficient for FEVER.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way"  # this is our internal label for this version of the preprocessing
INPUT_DATA_DIR="${DATA_DIR}"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
mkdir -p "${OUTPUT_DATA_DIR}"

#INPUT_FILE="train.jsonl"
#INPUT_FILE="shared_task_dev.jsonl"

# TODO: UPDATE --bert_cache_dir WITH YOUR PATH
for INPUT_FILE in "train.jsonl" "shared_task_dev.jsonl";
do
  python -u ${REPO_DIR}/archive_research_code/data/fever_preprocessing/fever_data_version8_3class.py \
  --input_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
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
# INPUT_FILE="train.jsonl"
#
# Total number of titles: 5396106
# number_of_duplicates_due_to_filtering: 551826
#
# WARNING: 0 covered claims in Newspapers exclude listings.
# WARNING: 0 covered claims in Newspapers exclude reviews.
#
# Proportion of verifiable claims with at least one lexically covered title: 0.9786722520717603
# Proportion of verifiable claims for which the chosen title is the first title in the chosen evidence set: 0.9780894271924233
# Total unique covered titles: 107503
# Size of covered titles sets: mean: 51.832332982694965; min: 0, max: 559
# Size of covered sentences sets: mean: 307.22197471278594; min: 0, max: 2777
# Size of non-zero (i.e., verifiable) true sentences sets: mean: 1.5732902285766324; min: 1, max: 38
# Number of blank wiki sentences dropped in the covered titles sets: 28517226
# Total verifiable claims: 109810; number supported: 80035; number refuted: 29775
# Total unverifiable claims: 35639 out of 145449: proportion: 0.24502746667216688
# Total number of covered sentences (across all claims, including duplicates): 44685129
# Size of evidence set for the chosen sentences: mean: 1.1181768509243237; min: 1, max: 2, len: 109810
# Size of evidence set for the chosen sentence == 1: len: 96833,proportion of total: 0.8818231490756762
# Size of evidence set for the chosen sentence == 2: len: 12977,proportion of total: 0.11817685092432383
# Number of chosen evidence sentences not in the covered sentences (including for other claims): 2296
# Number of reformatted titles mapped to multiple title hashes: 0

# INPUT_FILE="shared_task_dev.jsonl"
#
# Total number of titles: 5396106
# number_of_duplicates_due_to_filtering: 551826
#
# WARNING: 0 covered claims in Cleopatre premiered posthumously.
# WARNING: 0 covered claims in Kojol received nominations.
# WARNING: 0 covered claims in Cleopatre premiered violently.
#
# Proportion of verifiable claims with at least one lexically covered title: 0.9710471047104711
# Proportion of verifiable claims for which the chosen title is the first title in the chosen evidence set: 0.9706720672067207
# Total unique covered titles: 51337
# Size of covered titles sets: mean: 52.1009100910091; min: 0, max: 318
# Size of covered sentences sets: mean: 309.5532053205321; min: 0, max: 1821
# Size of non-zero (i.e., verifiable) true sentences sets: mean: 1.4562706270627064; min: 1, max: 32
# Number of blank wiki sentences dropped in the covered titles sets: 3988236
# Total verifiable claims: 13332; number supported: 6666; number refuted: 6666
# Total unverifiable claims: 6666 out of 19998: proportion: 0.3333333333333333
# Total number of covered sentences (across all claims, including duplicates): 6190445
# Size of evidence set for the chosen sentences: mean: 1.0905340534053405; min: 1, max: 2, len: 13332
# Size of evidence set for the chosen sentence == 1: len: 12125,proportion of total: 0.9094659465946595
# Size of evidence set for the chosen sentence == 2: len: 1207,proportion of total: 0.09053405340534053
# Number of chosen evidence sentences not in the covered sentences (including for other claims): 535
# Number of reformatted titles mapped to multiple title hashes: 0

################################################################################
#### 3. (Optional) Format FEVER data -- small debugging sets with labels
#### Optionally, use this to create smaller subsets for debugging. (Note
#### that due to the mappings in the dictionary files, it's not possible to just
#### take head -n on the full input files to get smaller versions.)
#### Note that in the dev set, in particular, 'unverifiable' claims are
#### prominent due to the sort. Run with the preferred INPUT_FILE.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

for DEBUG_SET_SIZE in 100 2000;
do
  DEBUG_SUFFIX=".head${DEBUG_SET_SIZE}.txt"
  PREPROCESS_VERSION="version8_3way_debug_set_of_size${DEBUG_SET_SIZE}"
  INPUT_DATA_DIR="${DATA_DIR}"
  OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
  mkdir -p "${OUTPUT_DATA_DIR}"

  echo "${OUTPUT_DATA_DIR}"
  # TODO: Choose desired input file:
  INPUT_FILE="train.jsonl"
  #INPUT_FILE="shared_task_dev.jsonl"

  # TODO: UPDATE --bert_cache_dir WITH YOUR PATH
  python -u ${REPO_DIR}/archive_research_code/data/fever_preprocessing/fever_data_version8_3class.py \
  --input_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
  --input_wiki_file UPDATE_WITH_YOUR_PATH/wiki-pages/wiki-pages/processed/wiki_v2/detokenized/all.wiki_v2.txt \
  --output_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.txt"${DEBUG_SUFFIX} \
  --output_control_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.control.txt"${DEBUG_SUFFIX} \
  --output_true_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_titles.txt"${DEBUG_SUFFIX} \
  --output_covered_titles_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_titles.txt"${DEBUG_SUFFIX} \
  --output_chosen_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.txt"${DEBUG_SUFFIX} \
  --output_true_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.true_sentences.jsonl"${DEBUG_SUFFIX} \
  --output_covered_sentences_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.txt"${DEBUG_SUFFIX} \
  --output_filtered_titles_to_title_hashes_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.title_hashes.txt"${DEBUG_SUFFIX} \
  --output_chosen_sentences_only_evidence_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.pairedformat.chosen_sentences.evidence_only.txt"${DEBUG_SUFFIX} \
  --output_covered_sentences_dictionary_file "${OUTPUT_DATA_DIR}/${INPUT_FILE}.${PREPROCESS_VERSION}.covered_sentences.dictionary.txt"${DEBUG_SUFFIX} \
  --bert_cache_dir="/Volumes/Extreme SSD/main/models/bert_cache/started_2020_03_10/" \
  --bert_model=${BERT_MODEL} \
  --size_of_debug_set ${DEBUG_SET_SIZE}
done


################################################################################
#### 4. Format FEVER data -- dev and test without labels
####
#### Separately, we also need to preprocess the versions that lack
#### ground-truth labels. This includes the final test set, as well as a version
#### of the dev set with labels removed. Note that the output from
#### the coarse-to-fine search (memory_match.py) needs to be subsequently
#### resorted to match the order of the original claims before posting to
#### CodaLab/etc. The order is recoverable from the original claim ids (in
#### the control file), and the original Wiki title hash is available from the
#### --output_filtered_titles_to_title_hashes_file dictionary file.
#### See post-processing.
################################################################################

REPO_DIR=UPDATE_WITH_YOUR_PATH_TO_THE_REPO  # TODO: UPDATE WITH YOUR PATH
cd ${REPO_DIR}

BERT_MODEL="bert-large-cased"

DATA_DIR="/Users/a/Documents/data/fever"  # TODO: UPDATE WITH YOUR PATH

PREPROCESS_VERSION="version8_3way"
INPUT_DATA_DIR="${DATA_DIR}"
OUTPUT_DATA_DIR="${INPUT_DATA_DIR}/${PREPROCESS_VERSION}/pairedformat"
mkdir -p "${OUTPUT_DATA_DIR}"

#INPUT_FILE="shared_task_dev_public.jsonl"
#INPUT_FILE="shared_task_test.jsonl"

# TODO: UPDATE --bert_cache_dir WITH YOUR PATH
for INPUT_FILE in "shared_task_dev_public.jsonl" "shared_task_test.jsonl";
do
  python -u ${REPO_DIR}/archive_research_code/data/fever_preprocessing/fever_data_version8_3class.py \
  --input_jsonl_file "${INPUT_DATA_DIR}/${INPUT_FILE}" \
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
  --size_of_debug_set -1 \
  --input_is_blind_test
done

#INPUT_FILE="shared_task_dev_public.jsonl"
# Total unique covered titles: 51337
# Size of covered titles sets: mean: 52.1009100910091; min: 0, max: 318
# Size of covered sentences sets: mean: 309.5532053205321; min: 0, max: 1821
# Number of blank wiki sentences dropped in the covered titles sets: 3988236
# Total verifiable claims: 0; number supported: 0; number refuted: 0
# Total unverifiable claims: 19998 out of 19998: proportion: 1.0
# Total number of covered sentences (across all claims, including duplicates): 6190445
# Number of chosen evidence sentences not in the covered sentences (including for other claims): 0
# Number of reformatted titles mapped to multiple title hashes: 0



#INPUT_FILE="shared_task_test.jsonl"
# WARNING: 0 covered claims in Bacteriophages reproduce.
# Total unique covered titles: 56596
# Size of covered titles sets: mean: 54.04675467546755; min: 0, max: 444
# Size of covered sentences sets: mean: 318.9016901690169; min: 0, max: 2028
# Number of blank wiki sentences dropped in the covered titles sets: 4103178
# Total verifiable claims: 0; number supported: 0; number refuted: 0
# Total unverifiable claims: 19998 out of 19998: proportion: 1.0
# Total number of covered sentences (across all claims, including duplicates): 6377396
# Number of chosen evidence sentences not in the covered sentences (including for other claims): 0
# Number of reformatted titles mapped to multiple title hashes: 0

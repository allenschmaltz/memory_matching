# Note: currently assumes 2-class.
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

# for saving exemplar metadata
import pickle


def save_memory_metadata_pickle(memory_dir, file_identifier_prefix, memory_metadata):
    path = f"{memory_dir}/{file_identifier_prefix}_memory.pkl"
    with open(path, 'wb') as output:
        pickle.dump(memory_metadata, output, protocol=3)


def load_memory_metadata_pickle(memory_dir, file_identifier_prefix):
    path = f"{memory_dir}/{file_identifier_prefix}_memory.pkl"
    try:
        with open(path, 'rb') as input:
            memory_metadata = pickle.load(input)
        return memory_metadata
    except:
        print(f"No available memory metadata at {path}.")
        exit()


def exemplars_main(data, params, np_random_state, bert_model, tokenizer, bert_device, model, predicted_output):
    assert int(data["create_exemplar_database"]) + int(data["create_exemplar_query"]) <= 1, f"ERROR: The exemplar" \
                                                                                            f"database and query must" \
                                                                                            f"be constructed " \
                                                                                            f"separately."
    if data["create_exemplar_database"]:
        retrieve_and_save_exemplar_cross_sequences(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                                   model, predicted_output, exemplar_mode="database")
        return None
    elif data["create_exemplar_query"]:
        retrieve_and_save_exemplar_cross_sequences(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                                   model, predicted_output, exemplar_mode="query")
        return None
    if data["save_exemplar_output"]:
        # run exemplar inference; this assumes that the database and query have already been created
        pdist = nn.PairwiseDistance(p=2)
        # we only consider 2 class for the symmetric experiments
        database_decision_labels_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID]
        exemplar_id_to_title_id_and_dist_and_ec_dict_by_class = \
            get_top_k_nearest_exemplar_cross_sequences_from_memory(pdist, data, model, params,
                                                                   database_decision_labels_to_consider)
        # save the output (along the lines of
        # utils_search_eval_levels_formats.eval_nearest_titles_from_memory_for_level3_ec_format)

        predicted_output = eval_nearest_titles_from_memory_for_level3_exemplar_format(database_decision_labels_to_consider,
                                                                   exemplar_id_to_title_id_and_dist_and_ec_dict_by_class,
                                                                   predicted_output,
                                                                   predicted_output[f"level{3}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"],
                                                                   data, params, save_eval_output=True, mode="test",
                                                                   level_id=3)
        return predicted_output


def retrieve_and_save_exemplar_cross_sequences(data, params, np_random_state, bert_model, tokenizer, bert_device,
                                               model, predicted_output, exemplar_mode="query"):

    # options: exemplar_mode="query" or exemplar_mode="database"
    # Note: If the command line option data["add_full_beam_to_exemplar_database"] is present, then all level 3
    # decision labels are added to the database. This is only applicable for the database in the current version (since
    # we don't currently consider flips in the query).

    if exemplar_mode == "database":
        print(f"Creating the exemplar database.")
        if data["add_full_beam_to_exemplar_database"]:
            print(f"Adding the full level 3 decision beam to the exemplar database.")
    else:
        assert exemplar_mode == "query"
        print(f"Creating the exemplar query.")
        # ensure only the top of the beam is considered for the query in this version:
        data["add_full_beam_to_exemplar_database"] = False

    # Note that there is some extraneous computation, since we make a second pass after generating predicted_output.
    # However, it is considerably more straightforward to just make the second pass here, then to try to build up the
    # exemplar vectors in chunks across levels as with utils_search.retrieve_and_save_memory. This is only run once
    # for each of database/query, so the extra cost is acceptable.

    exemplars_levels_to_consider = [1, 2, 3]

    start_time = time.time()
    model.eval()
    bert_model.eval()

    chunk_size = data[f"exemplar_chunk_size"]
    memory_batch_size = data[f"exemplar_memory_batch_size"]

    if exemplar_mode == "database":
        memory_dir = data[f"exemplar_database_memory_dir"]
    else:
        assert exemplar_mode == "query"
        memory_dir = data[f"exemplar_query_memory_dir"]

    decision_labels = data[f"{'test'}_decision_labels"]
    # each level should have the same number of claims:
    total_number_of_claims = len(data[f"level{1}_idx_{'test'}_x"])
    for level_id in exemplars_levels_to_consider:
        assert len(data[f"level{level_id}_idx_{'test'}_x"]) == total_number_of_claims
        assert len(predicted_output[f"level{level_id}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"]) == \
               total_number_of_claims
        assert len(decision_labels) == total_number_of_claims

    # Here, we split by ["tp", "fn", "fp", "tn"] as in multi-blade, but note that some of these subsets will be very
    # small due to the small size of the symmetric set.

    # Note that we only consider "tp", "fn", "fp", "tn" in terms of the final decision (i.e., not with respect
    # to retrieval in levels 1 or 2), as the symmetric data does not evaluate retrieval.
    masks = {}
    # important, note we only consider 2 class for the symmetric experiments:
    decision_labels_to_consider = [constants.SUPPORTS_ID, constants.REFUTES_ID]
    for class_type in decision_labels_to_consider:
        for v_type in ["tp", "fn", "fp", "tn"]:
            # These are indexes into the claims-titles cross-sequence. Note that there may be more than 1 cross-seq.
            # per claim (due to flipping the decision label).
            masks[f"mask_true_class{class_type}_{v_type}"] = []

    # initialize data structures:
    mini_batches = {}
    for level_id in exemplars_levels_to_consider:
        mini_batches[f"level{level_id}_augmented_idx_x"], \
        mini_batches[f"level{level_id}_augmented_bert_idx_sentences"], \
        mini_batches[f"level{level_id}_augmented_bert_input_masks"], \
        mini_batches[f"level{level_id}_augmented_idx_titles"], \
        mini_batches[f"level{level_id}_augmented_bert_idx_titles"], \
        mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"] = [], [], [], [], [], []

    # These cross sequences (claim+title) no longer necessarily match the number of claims, since we consider
    # all applicable labels in level 3. Depending on the setting, it may make sense to instead only consider
    # the top of the beam, or consider the full beam. Here, we consider the full beam if
    # data["add_full_beam_to_exemplar_database"] is True.
    cross_sequence_id = 0
    cross_sequence_id_to_claim_id = {}
    unique_titles_to_decision_labels = data[f"predicted_level{3}_{'test'}_unique_titles_to_decision_labels"]
    for claim_index in range(total_number_of_claims):
        # For levels 1 and 2, we only consider the top of the beam, so it stays constant as level 3 decision labels
        # are flipped
        # For level 3, we record all labels for the database if data["add_full_beam_to_exemplar_database"],
        # but only the predicted for query

        # We concurrently build cross-sequences for levels 1 and 2, simply duplicating the sequences
        # to push through the forward pass if data["add_full_beam_to_exemplar_database"]
        # to avoid complicated re-aligning

        level3_title_id_and_dist_and_ec_relative_to_level = \
            predicted_output[f"level{3}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]

        assert len(level3_title_id_and_dist_and_ec_relative_to_level) == 2, f"ERROR: Only 2 labels expected for level 3 " \
                                                                     f"in the symmetric data"
        if not data["add_full_beam_to_exemplar_database"] or exemplar_mode == "query":
            # only take the top of the beam for the query at level 3
            level3_title_id_and_dist_and_ec_relative_to_level = level3_title_id_and_dist_and_ec_relative_to_level[0:1]

        assert len(predicted_output[f"level{1}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]) == 1, \
            f"ERROR: Only 1 title expected for levels " \
            f"1 and 2 in the symmetric data (i.e., " \
            f"evidence is given and it consists of only" \
            f"a single sentence)."
        assert len(predicted_output[f"level{2}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][claim_index]) == 1, \
            f"ERROR: Only 1 title expected for levels " \
            f"1 and 2 in the symmetric data (i.e., " \
            f"evidence is given and it consists of only" \
            f"a single sentence)."
        for level3_predicted_i in range(len(level3_title_id_and_dist_and_ec_relative_to_level)):
            level3_predicted_title_id = level3_title_id_and_dist_and_ec_relative_to_level[level3_predicted_i][0]
            predicted_decision_id = unique_titles_to_decision_labels[level3_predicted_title_id]
            true_decision_id = decision_labels[claim_index]
            if predicted_decision_id in decision_labels_to_consider:
                # There's duplication (via symmetry) here with just the binary case, but we keep this template
                # for eventually moving back to multi-label/class, as with multi-BLADE. I.e., at inference in this case,
                # we could (equivalently) simply always pivot, for example, on mask_true_class{constants.SUPPORTS_ID}.
                if true_decision_id == constants.SUPPORTS_ID:
                    if predicted_decision_id == constants.SUPPORTS_ID:
                        # Note the semantics here: Since we are considering the full beam, 'tp' cases will also
                        # include cases that are deeper in the beam (i.e., wouldn't be predicted at inference). As
                        # noted above, we do this here in order to create additional instances for comparison
                        # since the symmetric set is very small. We could also view this as a type of data
                        # augmentation.
                        masks[f"mask_true_class{constants.SUPPORTS_ID}_{'tp'}"].append(cross_sequence_id)
                        masks[f"mask_true_class{constants.REFUTES_ID}_{'tn'}"].append(cross_sequence_id)
                    elif predicted_decision_id == constants.REFUTES_ID:
                        masks[f"mask_true_class{constants.SUPPORTS_ID}_{'fn'}"].append(cross_sequence_id)
                        masks[f"mask_true_class{constants.REFUTES_ID}_{'fp'}"].append(cross_sequence_id)
                elif true_decision_id == constants.REFUTES_ID:
                    if predicted_decision_id == constants.SUPPORTS_ID:
                        masks[f"mask_true_class{constants.REFUTES_ID}_{'fn'}"].append(cross_sequence_id)
                        masks[f"mask_true_class{constants.SUPPORTS_ID}_{'fp'}"].append(cross_sequence_id)
                    elif predicted_decision_id == constants.REFUTES_ID:
                        masks[f"mask_true_class{constants.REFUTES_ID}_{'tp'}"].append(cross_sequence_id)
                        masks[f"mask_true_class{constants.SUPPORTS_ID}_{'tn'}"].append(cross_sequence_id)
                assert cross_sequence_id not in cross_sequence_id_to_claim_id
                cross_sequence_id_to_claim_id[cross_sequence_id] = claim_index
                cross_sequence_id += 1
                for level_id in exemplars_levels_to_consider:
                    if level_id == 3:
                        predicted_title_id = level3_predicted_title_id
                    else:
                        # for levels 1 and 2, we just take the top of the beam, since retrieval is given in this case
                        predicted_title_id = \
                        predicted_output[f"level{level_id}_{'test'}_retrieval_id_to_title_id_and_dist_and_ec"][
                            claim_index][0][0]

                    # add 'left' sequences
                    mini_batches[f"level{level_id}_augmented_idx_x"].append(
                        data[f"level{level_id}_idx_{'test'}_x"][claim_index]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_idx_sentences"].append(
                        data[f"level{level_id}_{'test'}_bert_idx_sentences"][claim_index]
                    )
                    mini_batches[f"level{level_id}_augmented_bert_input_masks"].append(
                        data[f"level{level_id}_{'test'}_bert_input_masks"][claim_index]
                    )
                    # add 'right' sequences
                    if level_id == 1:
                        mini_batches[f"level{level_id}_augmented_idx_titles"].append(
                            data[f"{'test'}_idx_unique_titles"][predicted_title_id]
                        )
                        mini_batches[f"level{level_id}_augmented_bert_idx_titles"].append(
                            data[f"{'test'}_bert_idx_unique_titles"][predicted_title_id]
                        )
                        mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"].append(
                            data[f"{'test'}_bert_input_masks_unique_titles"][predicted_title_id]
                        )
                    else:
                        mini_batches[f"level{level_id}_augmented_idx_titles"].append(
                            data[f"predicted_level{level_id}_{'test'}_idx_unique_titles"][predicted_title_id]
                        )
                        mini_batches[f"level{level_id}_augmented_bert_idx_titles"].append(
                            data[f"predicted_level{level_id}_{'test'}_bert_idx_unique_titles"][predicted_title_id]
                        )
                        mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"].append(
                            data[f"predicted_level{level_id}_{'test'}_bert_input_masks_unique_titles"][predicted_title_id]
                        )
    if data["add_full_beam_to_exemplar_database"]:
        assert cross_sequence_id == total_number_of_claims*2, f"{cross_sequence_id}, {total_number_of_claims}"
    else:
        assert cross_sequence_id == total_number_of_claims, f"{cross_sequence_id}, {total_number_of_claims}"

    total_sequences = len(mini_batches[f"level{1}_augmented_idx_x"])
    for level_id in exemplars_levels_to_consider:
        assert len(mini_batches[f"level{level_id}_augmented_idx_x"]) == total_sequences
    chunk_ids = []
    with torch.no_grad():
        chunk_id = 0
        for chunk_i in range(0, total_sequences, chunk_size):
            chunk_range = min(chunk_size, total_sequences - chunk_i)
            chunk_mini_batches = {}
            for level_id in exemplars_levels_to_consider:
                chunk_mini_batches[f"level{level_id}_augmented_idx_x"] = \
                    mini_batches[f"level{level_id}_augmented_idx_x"][chunk_i:chunk_i + chunk_range]
                chunk_mini_batches[f"level{level_id}_augmented_bert_idx_sentences"] = \
                    mini_batches[f"level{level_id}_augmented_bert_idx_sentences"][chunk_i:chunk_i + chunk_range]
                chunk_mini_batches[f"level{level_id}_augmented_bert_input_masks"] = \
                    mini_batches[f"level{level_id}_augmented_bert_input_masks"][chunk_i:chunk_i + chunk_range]
                chunk_mini_batches[f"level{level_id}_augmented_idx_titles"] = \
                    mini_batches[f"level{level_id}_augmented_idx_titles"][chunk_i:chunk_i + chunk_range]
                chunk_mini_batches[f"level{level_id}_augmented_bert_idx_titles"] = \
                    mini_batches[f"level{level_id}_augmented_bert_idx_titles"][chunk_i:chunk_i + chunk_range]
                chunk_mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"] = \
                    mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"][chunk_i:chunk_i + chunk_range]
            total_chunk_sequences = len(chunk_mini_batches[f"level{level_id}_augmented_idx_x"])

            print(f"Processing {exemplar_mode} memory structure chunk_id {chunk_id}: starting at {chunk_i}, with size {total_chunk_sequences}, out of {total_sequences}")
            memory_structure = []
            num_batch_instances = math.ceil((total_chunk_sequences / memory_batch_size))
            batch_num = 0
            for i in range(0, total_chunk_sequences, memory_batch_size):
                # if batch_num % max(1, int(num_batch_instances * 0.25)) == 0:
                #     print(f"\tProcessing {mode} memory, {batch_num/num_batch_instances}")
                batch_num += 1
                batch_range = min(memory_batch_size, total_chunk_sequences - i)
                batch_memory_structure = []
                for level_id in exemplars_levels_to_consider:
                    batch_x = chunk_mini_batches[f"level{level_id}_augmented_idx_x"][i:i + batch_range]
                    number_of_claims = len(batch_x)
                    batch_x.extend(chunk_mini_batches[f"level{level_id}_augmented_idx_titles"][i:i + batch_range])

                    bert_output = run_main.get_bert_representations(
                        chunk_mini_batches[f"level{level_id}_augmented_bert_idx_sentences"][i:i + batch_range] +
                        chunk_mini_batches[f"level{level_id}_augmented_bert_idx_titles"][i:i + batch_range],
                        chunk_mini_batches[f"level{level_id}_augmented_bert_input_masks"][i:i + batch_range] +
                        chunk_mini_batches[f"level{level_id}_augmented_bert_input_masks_titles"][i:i + batch_range],
                        bert_model, bert_device, params["bert_layers"], len(batch_x[0]))
                    bert_output = torch.FloatTensor(bert_output).to(params["main_device"])

                    batch_x = torch.LongTensor(batch_x).to(params["main_device"])

                    total_length = params["max_length"] * level_id + 2 * constants.PADDING_SIZE
                    model_output = model(batch_x, bert_output, level_id=level_id, total_length=total_length,
                                         forward_type_description="sentence_representation", main_device=None)

                    # left seq
                    claims_output = model_output[0:number_of_claims]
                    # right seq
                    titles_output = model_output[number_of_claims:]
                    abs_filter_diff = torch.abs(claims_output - titles_output)

                    # left :: right :: diff
                    batch_memory_structure.append(torch.cat([claims_output, titles_output, abs_filter_diff], 1))
                # (left :: right :: diff)_level1 :: (left :: right :: diff)_level2 :: (left :: right :: diff)_level3
                memory_structure.append(torch.cat(batch_memory_structure, 1))
                if chunk_i == 0 and i == 0 and params["main_device"].type != "cpu":
                    print(
                        f'i==0 retrieve_and_save_memory torch.cuda.max_memory_allocated: '
                        f'{torch.cuda.max_memory_allocated(params["main_device"])}')

            utils.save_memory_structure_torch(memory_dir, f"levels123diff_exemplar_{exemplar_mode}", torch.cat(memory_structure, 0), chunk_id)
            chunk_ids.append(chunk_id)
            chunk_id += 1
        # save chunk ids -- These are the keys for recovering the files
        utils.save_memory_structure_torch(memory_dir, f"levels123diff_exemplar_{exemplar_mode}_chunk_ids", torch.LongTensor(chunk_ids), 0)
        # save masks
        save_memory_metadata_pickle(memory_dir, f"levels123diff_exemplar_{exemplar_mode}_masks", masks)
        # save mapping from sequence id back to the original claim id -- this important when there is not necessarily
        # one exemplar sequence per claim
        save_memory_metadata_pickle(memory_dir, f"levels123diff_exemplar_{exemplar_mode}_cross_sequence_id_to_claim_id",
                                    cross_sequence_id_to_claim_id)
        # For convenience, we also save the list of decision labels: these are indexed by the claim index.
        # When querying the database, we know tp/tn from the masks (i.e., they are sufficient for recovering binary
        # labels), but we also save the decision_labels as a check
        # that the mapping is correct. Go through cross_sequence_id_to_claim_id to get the claim index (from the
        # retrieved exemplar sequence index), which can then be used to index this list.
        save_memory_metadata_pickle(memory_dir,
                                    f"levels123diff_exemplar_{exemplar_mode}_decision_labels_by_claim_indexes",
                                    decision_labels)

    for class_type in decision_labels_to_consider:
        for v_type in ["tp", "fn", "fp", "tn"]:
            # These are indexes into the claims-titles cross-sequence. Note that there may be more than 1 cross-seq.
            # per claim (due to flipping the decision label).
            size_of_mask = len(masks[f"mask_true_class{class_type}_{v_type}"])
            print(f"Size of mask_true_class{class_type}_{v_type} for {exemplar_mode}: {size_of_mask}")

    end_time = time.time()
    print(f"retrieve_and_save_exemplar_cross_sequences time: {(end_time - start_time) / 60} minutes")


def get_top_k_nearest_exemplar_cross_sequences_from_memory(pdist, data, model, params,
                                                           database_decision_labels_to_consider):
    # This mirrors utils_search.get_top_k_nearest_titles_from_memory, with the database now having a similar role
    # as the titles store, and the query having a similar role as the retrieval instances.

    #assert level_id != -1
    start_time = time.time()
    model.eval()

    top_k_nearest_memories = data[f"level{3}_top_k_nearest_memories"]
    memory_dir = data[f"exemplar_database_memory_dir"]

    memory_chunk_ids = utils.load_memory_structure_torch(memory_dir, f"levels123diff_exemplar_{'database'}_chunk_ids", 0, -1).numpy()

    retrieval_memory_dir = data[f"exemplar_query_memory_dir"]
    retrieval_chunk_ids = utils.load_memory_structure_torch(retrieval_memory_dir, f"levels123diff_exemplar_{'query'}_chunk_ids", 0, -1).numpy()

    # These masks are in terms of the database. Note that currently masks are actually saved for the query, as well,
    # but they are only used for debugging.
    masks = load_memory_metadata_pickle(memory_dir, f"levels123diff_exemplar_{'database'}_masks")

    total_filters = sum(model.FILTER_NUM)
    if data["exemplar_match_type"] == "level3_diff":
        print(f"Exemplars are constructed from the level 3 diff.")
    elif data["exemplar_match_type"] == "level1_diff":
        print(f"Exemplars are constructed from the level 1 diff.")
    elif data["exemplar_match_type"] == "level123_diff":
        print(f"Exemplars are constructed from the level 1 diff AND level 2 diff AND level 3 diff.")
        # (left :: right :: diff)_level1 :: (left :: right :: diff)_level2 :: (left :: right :: diff)_level3
        exemplar_indexes = []
        for possible_index in range(0, total_filters * 9):
            if (possible_index >= total_filters * 2 and possible_index < total_filters * 3) or \
                    (possible_index >= total_filters * 5 and possible_index < total_filters * 6) or \
                    (possible_index >= total_filters * 8):
                exemplar_indexes.append(possible_index)
        exemplar_indexes = torch.LongTensor(exemplar_indexes).to(params["main_device"])
        assert exemplar_indexes.shape[0] == total_filters * 3
    elif data["exemplar_match_type"] == "level23_diff":
        print(f"Exemplars are constructed from the level 2 AND level 3 diff.")
        # (left :: right :: diff)_level1 :: (left :: right :: diff)_level2 :: (left :: right :: diff)_level3
        exemplar_indexes = []
        for possible_index in range(0, total_filters*9):
            if (possible_index >= total_filters*5 and possible_index < total_filters*6) or \
                    (possible_index >= total_filters * 8):
                exemplar_indexes.append(possible_index)
        exemplar_indexes = torch.LongTensor(exemplar_indexes).to(params["main_device"])
        assert exemplar_indexes.shape[0] == total_filters*2
    else:
        print(f"Exemplars are constructed from the level 1, 2, and 3 claim and title vectors, AND the"
              f"level 1, 2, and 3 diff.")

    #claims_to_covered_titles_ids_tensors = data[f"level{level_id}_{mode}_claims_to_covered_titles_ids_tensors"]

    #retrieval_id_to_title_id_and_dist_and_ec = defaultdict(list)  # sorted (closest to farthest) list of [title id, dist]

    exemplar_id_to_title_id_and_dist_and_ec_dict_by_class = {}
    for class_type in database_decision_labels_to_consider:
        for v_type in ["tp", "fn", "fp", "tn"]:
            # This mirrors retrieval_id_to_title_id_and_dist_and_ec in
            # utils_search.get_top_k_nearest_titles_from_memory,
            # but now we keep a separate structure for each of "tp", "fn", "fp", "tn", for each label. This means that
            # each query claim will have 4 exemplar vectors (or unk) corresponding to each of those database instances,
            # FOR EACH LABEL (i.e., a total of 8 distances in the binary label case). Note that in the strictly binary
            # label case, we don't actually need to keep structures for
            # both labels (due to the symmetry), but we keep this setup for the eventual merge with the multi-label code
            # base.
            exemplar_id_to_title_id_and_dist_and_ec_dict_by_class[f"retrieval_id_to_title_id_and_dist_and_ec_true_class{class_type}_{v_type}"] = \
                defaultdict(list)
            # Convert masks to tensors and send to gpu (if applicable). Note that unlike
            # claims_to_covered_titles_ids_tensors in utils_search.get_top_k_nearest_titles_from_memory, these masks
            # stay constant for all query instances (i.e., every query sees the same database instances) in the current
            # version
            masks[f"mask_true_class{class_type}_{v_type}"] = torch.LongTensor(masks[f"mask_true_class{class_type}_{v_type}"]).to(params["main_device"])

    with torch.no_grad():
        running_memory_line_index = 0
        for memory_chunk_id in memory_chunk_ids:
            running_retrieval_line_index = 0  # Note that we reload the entire retrieval set for EACH memory chunk
            print(f"TopK: Processing database memory chunk id {memory_chunk_id} of {len(memory_chunk_ids)}.")
            title_memory = utils.load_memory_structure_torch(memory_dir, f"levels123diff_exemplar_{'database'}", memory_chunk_id, int(params["GPU"]))
            for retrieval_chunk_id in retrieval_chunk_ids:
                retrieval_memory = utils.load_memory_structure_torch(retrieval_memory_dir, f"levels123diff_exemplar_{'query'}", retrieval_chunk_id, int(params["GPU"]))
                # Now, match each retrieval entry with a title entry (in this chunk) for each of
                # "tp", "fn", "fp", "tn", for each label
                for sentence_i in range(retrieval_memory.shape[0]):
                    retrieval_id = running_retrieval_line_index + sentence_i
                    for class_type in database_decision_labels_to_consider:
                        for v_type in ["tp", "fn", "fp", "tn"]:
                            current_title_id_and_dist_and_ec = exemplar_id_to_title_id_and_dist_and_ec_dict_by_class[
                                f"retrieval_id_to_title_id_and_dist_and_ec_true_class{class_type}_{v_type}"][retrieval_id]
                            #current_title_id_and_dist_and_ec = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]

                            #if claims_to_covered_titles_ids_tensors[retrieval_id].shape[0] > 0:
                            if masks[f"mask_true_class{class_type}_{v_type}"].shape[0] > 0:
                                # offset by current chunk
                                admitted_ids = masks[f"mask_true_class{class_type}_{v_type}"]
                                #admitted_ids = claims_to_covered_titles_ids_tensors[retrieval_id]
                                # need to restrict the ids to those that appear in this chunk
                                admitted_ids = admitted_ids[admitted_ids.ge(running_memory_line_index) & admitted_ids.lt(running_memory_line_index+title_memory.shape[0])]
                                if admitted_ids.shape[0] > 0:
                                    # if any remaining ids, next, need to renormalize the indexes to this chunk:
                                    admitted_ids -= running_memory_line_index
                                    # and then get the applicable subset
                                    #title_memory_subset = title_memory[admitted_ids.to(params["main_device"]), :]
                                    # in this case, admitted_ids are already on the gpu, if applicable
                                    title_memory_subset = title_memory[admitted_ids, :]
                                    if data["exemplar_match_type"] == "level3_diff":
                                        l2_pairwise_distances = pdist(retrieval_memory[sentence_i, -total_filters:].expand_as(title_memory_subset[:, -total_filters:]), title_memory_subset[:, -total_filters:])
                                    elif data["exemplar_match_type"] == "level1_diff":
                                        l2_pairwise_distances = pdist(retrieval_memory[sentence_i, total_filters*2:total_filters*3].expand_as(title_memory_subset[:, total_filters*2:total_filters*3]), title_memory_subset[:, total_filters*2:total_filters*3])
                                    elif data["exemplar_match_type"] == "level123_diff" or data["exemplar_match_type"] == "level23_diff":
                                        # in these cases, the desired indexes are discontinuous, so we use a LongTensor
                                        # (exemplar_indexes) to index, which is constructed above
                                        l2_pairwise_distances = pdist(retrieval_memory[sentence_i, exemplar_indexes].expand_as(title_memory_subset[:, exemplar_indexes]), title_memory_subset[:, exemplar_indexes])
                                    else:
                                        l2_pairwise_distances = pdist(retrieval_memory[sentence_i, :].expand_as(title_memory_subset), title_memory_subset)
                                    smallest_k_distances, smallest_k_distances_idx = torch.topk(l2_pairwise_distances, min(top_k_nearest_memories, title_memory_subset.shape[0]), largest=False, sorted=True)
                                    for dist, dist_idx in zip(smallest_k_distances, smallest_k_distances_idx):
                                        # dist_idx needs to be recast to the admitted_ids and then renormed overall
                                        memory_id = running_memory_line_index + admitted_ids[dist_idx.item()].item()
                                        error_correction_score = 0.0  # currently not using
                                        current_title_id_and_dist_and_ec.append(
                                            [memory_id, dist.item(), error_correction_score])

                                        # # currently, not using level-specific fc layers:
                                        # abs_filter_diff = torch.abs(retrieval_memory[sentence_i, :] - title_memory_subset[dist_idx.item(), :])
                                        # fc_out = model.layer1_fc(abs_filter_diff)
                                        # error_correction_score = fc_out[1] - fc_out[0]
                                        # current_title_id_and_dist_and_ec.append(
                                        #     [memory_id, dist.item(), error_correction_score.item()])

                                #current_title_id_and_dist_and_ec.append([memory_id, dist.item(), error_correction_score.item()])
                            # Sort and keep top_k_nearest_memories
                            # Note that this is always necessary, even if top_k_nearest_memories=1, since we process
                            # in chunks.
                            exemplar_id_to_title_id_and_dist_and_ec_dict_by_class[
                                f"retrieval_id_to_title_id_and_dist_and_ec_true_class{class_type}_{v_type}"][
                                retrieval_id] = sorted(current_title_id_and_dist_and_ec, key=lambda x: x[1])[0:top_k_nearest_memories]
                # update retrieval line index:
                running_retrieval_line_index += retrieval_memory.shape[0]
            # update memory line index:
            running_memory_line_index += title_memory.shape[0]

    end_time = time.time()
    print(f"get_top_k_nearest_exemplar_cross_sequences_from_memory time: {(end_time - start_time) / 60} minutes")
    return exemplar_id_to_title_id_and_dist_and_ec_dict_by_class


def eval_nearest_titles_from_memory_for_level3_exemplar_format(database_decision_labels_to_consider, exemplar_id_to_title_id_and_dist_and_ec_dict_by_class, predicted_output, retrieval_id_to_title_id_and_dist_and_ec, data, params, save_eval_output=False, mode="train", level_id=-1):
    # class,type,id,dist
    # We keep this relatively simple, only saving the output, and then subsequent analysis is done off the gpu's.

    assert level_id == 3
    assert len(predicted_output[f"level{level_id-1}_{mode}_retrieval_distances_at_top_of_beam"]) == \
           len(exemplar_id_to_title_id_and_dist_and_ec_dict_by_class[f"retrieval_id_to_title_id_and_dist_and_ec_true_class{0}_{'tp'}"])

    start_time = time.time()

    original_claims = data[f"{mode}_original_sentences"]
    unique_original_titles = data[f"{mode}_unique_original_titles"]
    top_k_nearest_memories = data[f"level{level_id}_top_k_nearest_memories"]
    claims_to_chosen_title_ids = data[f"{mode}_claims_to_chosen_title_ids"]
    claims_to_true_titles_ids = data[f"{mode}_claims_to_true_titles_ids"]
    claims_to_true_title_ids_evidence_sets = data[f"{mode}_claims_to_true_title_ids_evidence_sets"]
    decision_labels = data[f"{mode}_decision_labels"]  # ground-truth decision labels for each claim
    #unique_title_ids_to_decision_labels = data[f"{mode}_unique_title_ids_to_decision_labels"]  # ground-truth decision labels for each unique title


    predicted_original_evidence_ids = data[f"predicted_level{level_id}_{mode}_original_evidence_ids"]
    unique_titles_to_decision_labels = data[f"predicted_level{level_id}_{mode}_unique_titles_to_decision_labels"]
    predicted_original_evidence_ids_marginalized = data[f"predicted_level{level_id}_{mode}_original_evidence_ids_marginalized"]

    # for fever, the following provides a mapping from wiki sentences to wiki documents; this is used for eval purposes
    # (e.g., to check whether the ground truth documents are found)
    unique_title_ids_to_document_ids = data[f"{mode}_unique_title_ids_to_document_ids"]

    assert len(claims_to_chosen_title_ids) == len(retrieval_id_to_title_id_and_dist_and_ec), \
        f"{len(claims_to_chosen_title_ids)}, {len(retrieval_id_to_title_id_and_dist_and_ec)}"
    retrieval_id_check = 0
    for _ in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        assert retrieval_id_check in retrieval_id_to_title_id_and_dist_and_ec
        retrieval_id_check += 1

    eval_output_compact = []

    true_top_decision_predictions = []
    distances = []
    distances_true_predictions = []
    distances_false_predictions = []
    retrieval_distances_true_predictions = []
    retrieval_distances_false_predictions = []
    retrieval_distance_unk_for_nonunk_prediction = 0
    num_unpredicted_lines = 0
    for retrieval_id in range(len(retrieval_id_to_title_id_and_dist_and_ec)):
        output_line = []
        decision_is_correct = False
        # in this case, need to convert relative ids back to the original ids
        title_id_and_dist_and_ec_relative_to_level = retrieval_id_to_title_id_and_dist_and_ec[retrieval_id]
        if len(title_id_and_dist_and_ec_relative_to_level) >= 1:
            predicted_title_id = title_id_and_dist_and_ec_relative_to_level[0][0]
            predicted_retrieval_original_evidence_ids = predicted_original_evidence_ids_marginalized[predicted_title_id]
            predicted_title_strings = []
            for predicted_retrieval_original_evidence_id in predicted_retrieval_original_evidence_ids:
                predicted_title_strings.append(' '.join(unique_original_titles[predicted_retrieval_original_evidence_id]))

            predicted_title_distance = title_id_and_dist_and_ec_relative_to_level[0][1]
            distances.append(predicted_title_distance)

            retrieval_distance_at_top_of_beam = \
                predicted_output[f"level{level_id - 1}_{mode}_retrieval_distances_at_top_of_beam"][retrieval_id]
            predicted_decision_id = unique_titles_to_decision_labels[predicted_title_id]
            if predicted_decision_id == decision_labels[retrieval_id]:
                decision_is_correct = True
                distances_true_predictions.append(predicted_title_distance)
                if retrieval_distance_at_top_of_beam != -1.0:
                    retrieval_distances_true_predictions.append(retrieval_distance_at_top_of_beam)
                else:
                    retrieval_distance_unk_for_nonunk_prediction += 1
            else:
                distances_false_predictions.append(predicted_title_distance)
                if retrieval_distance_at_top_of_beam != -1.0:
                    retrieval_distances_false_predictions.append(retrieval_distance_at_top_of_beam)
                else:
                    retrieval_distance_unk_for_nonunk_prediction += 1

            #db_class,type,id,dist
            for class_type in database_decision_labels_to_consider:
                for v_type in ["tp", "fn", "fp", "tn"]:
                    current_title_id_and_dist_and_ec = exemplar_id_to_title_id_and_dist_and_ec_dict_by_class[
                        f"retrieval_id_to_title_id_and_dist_and_ec_true_class{class_type}_{v_type}"][retrieval_id]
                    # only consider the top of the beam
                    if len(current_title_id_and_dist_and_ec) >= 1:
                        top_of_beam_db_id = current_title_id_and_dist_and_ec[0][0]
                        top_of_beam_dist = current_title_id_and_dist_and_ec[0][1]
                    else:
                        top_of_beam_db_id = constants.UNK_TITLE_ID
                        top_of_beam_dist = -1
                    output_line.append(f"db_class{class_type},{v_type},{top_of_beam_db_id},{top_of_beam_dist}")

            output_line.append(f"Retrieval (k=0) Distance:")
            output_line.append(f"{retrieval_distance_at_top_of_beam}")
            output_line.append(f"Decision Distance:")
            output_line.append(f"{predicted_title_distance}")
            output_line.append(f"Prediction: {decision_is_correct}")
            output_line.append(f"{predicted_decision_id}")

            if predicted_decision_id == constants.SUPPORTS_ID:
                output_line.append(constants.SUPPORTS_STRING)
            elif predicted_decision_id == constants.REFUTES_ID:
                output_line.append(constants.REFUTES_STRING)
            elif predicted_decision_id == constants.MOREINFO_ID:
                output_line.append(constants.MOREINFO_STRING)
            else:
                assert False

            output_line.append(' '.join(original_claims[retrieval_id]))
            output_line.extend(predicted_title_strings)
        else:
            for class_type in database_decision_labels_to_consider:
                for v_type in ["tp", "fn", "fp", "tn"]:
                    output_line.append(f"db_class{class_type},{v_type},{constants.UNK_TITLE_ID},{-1}")
            output_line.append(f"Retrieval (k=0) Distance:")
            output_line.append(f"{-1}")
            output_line.append(f"Decision Distance:")
            output_line.append(f"{-1}")
            output_line.append(f"Prediction: {decision_is_correct}")
            output_line.append(f"{constants.UNK_TITLE_ID}")
            output_line.append("")
            output_line.append(' '.join(original_claims[retrieval_id]))
            num_unpredicted_lines += 1

        # output line is: Distance\tDistance val\tPrediction: True/False\tDecision id\tDecision string\tclaim\evidence (if any)
        eval_output_compact.append("\t".join(output_line)+"\n")
        true_top_decision_predictions.append(int(decision_is_correct))

    print(f"Total unpredicted lines: {num_unpredicted_lines} out of {len(true_top_decision_predictions)}")

    print(
        f"\tEval ({mode}) level{level_id}: Number of correct decision predictions (regardless of retrieval): "
        f"{np.sum(true_top_decision_predictions)} out of {len(true_top_decision_predictions)}; "
        f"proportion: {np.mean(true_top_decision_predictions)}")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered predicted: Distance "
            f"mean: {np.mean(distances)}; std: {np.std(distances)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered correctly predicted: Distance "
            f"mean: {np.mean(distances_true_predictions)}; std: {np.std(distances_true_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered wrongly predicted: Distance "
            f"mean: {np.mean(distances_false_predictions)}; std: {np.std(distances_false_predictions)}")
    except:
        print(f"Unexpected distance output")

    print_distance_stats(mode, level_id, retrieval_distance_unk_for_nonunk_prediction,
                         retrieval_distances_true_predictions,
                         retrieval_distances_false_predictions)

    predicted_output[f"level{level_id}_{mode}_nearest_true_level_title_ids"] = -1
    predicted_output[f"level{level_id}_{mode}_nearest_wrong_level_title_ids"] = -1
    predicted_output[f"level{level_id}_{mode}_retrieval_acc"] = -1
    predicted_output[f"level{level_id}_{mode}_decision_acc"] = np.mean(true_top_decision_predictions)
    predicted_output[f"level{level_id}_{mode}_score_vals"] = []
    predicted_output[f"level{level_id}_{mode}_score_vals_compact"] = eval_output_compact

    end_time = time.time()
    print(f"eval_nearest_titles_from_memory_for_level3_exemplar_format time: {(end_time - start_time) / 60} minutes")

    return predicted_output


def print_distance_stats(mode, level_id, retrieval_distance_unk_for_nonunk_prediction,
                         retrieval_distances_true_predictions,
                         retrieval_distances_false_predictions):

    print(f"Number of retrieval distances that are -1 for non-unk decision predictions: "
          f"{retrieval_distance_unk_for_nonunk_prediction}")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered correctly predicted decisions: Retrieval distances "
            f"mean: {np.mean(retrieval_distances_true_predictions)}; "
            f"std: {np.std(retrieval_distances_true_predictions)}")
    except:
        print(f"Unexpected distance output")

    try:
        print(
            f"\tEval ({mode}) level{level_id}: Among covered wrongly predicted decisions: Retrieval distances "
            f"mean: {np.mean(retrieval_distances_false_predictions)}; "
            f"std: {np.std(retrieval_distances_false_predictions)}")
    except:
        print(f"Unexpected distance output")
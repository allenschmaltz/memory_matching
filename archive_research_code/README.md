# memory_matching -- limited support reference code

This is a limited support (pre-publication) release of some of my raw research code. This is primarily intended for researchers already familiar with non-parametric neural structures to provide complementary details to the arXiv paper. (Rather than one-off sharing, I'm just releasing it here.) A cleaned up, refactored version of the code aimed at a more general CS audience will be released in the main directory no later than publication. (Those caveats aside, this is a fully working codebase and can be used to replicate the results in the paper. We also provide the trained model and the preprocessed data below.)

To get started, you will need to install a previous version of the HuggingFace transformer repo. I will (probably) update to the more recent HuggingFace transformer repo in the final refactored version, but at the moment, you'll need to use the older version due to minor changes in the function definitions. I've included a copy [here](transformer_versions/bert_hsph/pytorch-pretrained-BERT-master.zip). Unpack the directory and install, as with

```bash
pip install --editable .
```

This has been tested on a single V100 GPU with 32GB of memory. Training on a lower memory GPU is probably not tenable, at least for the full FEVER dataset, as the large memory card allows very large batch sizes in the forward pass through the coarse-to-fine search, as well as batching all relevant sequences from all levels for multiple claims together during training. A multi-GPU version is in the queue, but will likely not be released until some time after the publication code release.

This has most recently been tested with Python 3.7.7, PyTorch 1.6.0, and CUDA 10.2. A few additional dependencies are noted in [notes/additional_dependencies.md](notes/additional_dependencies.md).

The code heavily relies on caching the memory vectors to disk, so fast scratch space for the memory directories is important. 50 GB should be sufficient for up to k_1=100 for FEVER.

In principle, this code can be used as-is on other datasets by simply matching the FEVER data format used here; however, in the interest of time, I'm going to hold off on providing support/advice for that until the refactored codebase (to appear in the main directory) is released, which will simplify running on additional datasets.

Note that I have removed some code associated with certain command line options (this is mostly my raw research scaffolding, with limited changes) to make it easier to follow, so use the provided scripts to know which options to use for replication, rather than the argparse.ArgumentParser help, some of the documentation of which is now stale.

1. Details on the preprocessing of FEVER (along with Wikipedia) and the symmetric sets is described in [notes/data_preprocessing.md](notes/data_preprocessing.md). (Note that here and elsewhere, the associated bash scripts will need updates for file directories/etc.) This is a somewhat involved process, requiring a pass through all of the (introductory sections of the) Wikipedia articles, creating the covered sets, and generating the various input and debugging sets. You can alternatively [download the preprocessed data here (approx. 4GB, including the raw wiki pages)](https://drive.google.com/file/d/1AYPJEg2vhAd_dPmjliCgGD1tLQYsUakj/view?usp=sharing).

2. An example of training the model is in [notes/scripts/train/train.sh](notes/scripts/train/train.sh). The trained model used to submit to CodaLab is [available for download here (approx. 861MB)](https://drive.google.com/file/d/1DCaN3il34nrTvxpz_-YW8-of1lquG2WA/view?usp=sharing).

3. An example of running inference, as well as converting to the CodaLab format and evaluating the FEVER score is in [notes/scripts/inference/create_k1_z_vs_acc_table.sh](notes/scripts/inference/create_k1_z_vs_acc_table.sh). This can be used to re-create Table 2 in the paper. For reference, these dev set output files are [available for download here (approx. 32MB)](https://drive.google.com/file/d/1QaP-HzKmxYm1oubqtwjg7o2f6uatldvP/view?usp=sharing).

4. Similarly, to run on the data files with hidden labels see [notes/scripts/inference/hidden_labels_eval.sh](notes/scripts/inference/hidden_labels_eval.sh).

A few points of particular interest/focus:

1. The core of the coarse-to-fine search is handled by an iteration over the 'levels' (to use the terminology of the paper), as in the example for the test split below. This involves two main tasks for each level. First, calls to ``utils_search.retrieve_and_save_memory()`` for the titles (a.k.a., support sequences, to use the convention of the paper) and for 'retrieve' mode (a.k.a., for the query sequences) process the memory vectors. The query and support sequences are first chunked, batched, and forwarded separately, with the resulting memory vectors saved to disk. Second, ``utils_search.get_nearest_titles_from_memory_for_all_levels()`` will get the top-k support sequences matched to every query sequence (via ``utils_search.get_top_k_nearest_titles_from_memory()``) by iteratively reading and comparing the chunked memory vectors from disk, and will also update the data structures: predicted_output contains effectiveness metrics and the predicted ids, and the data dictionary will get updated with the dynamically created sequences for use in the subsequent level. Note that the level 1 support sequences are bi-encoded, and the levels 2 and 3 support sequences are cross-encoded, with the latter dependent on the search results of the earlier levels.

```python
predicted_output = {}
for level_id in levels_to_consider:  # level_id in [1,2,3], for example
    utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="title", split_mode="test", level_id=level_id)
    utils_search.retrieve_and_save_memory(data, model, params, bert_model, bert_device, mode="retrieve", split_mode="test", level_id=level_id)
    data, predicted_output = utils_search.get_nearest_titles_from_memory_for_all_levels(predicted_output, pdist, data, model, params,
                                                  save_eval_output=True, mode="test", level_id=level_id)
```

* A variation on this general theme is taken to generate and match to the exemplar database, with the difference that the query and support sequences are chunked and batched together in order to produce the difference vectors (since in that case an initial search has already occurred).

2. For training, we first run the coarse-to-fine search to get the hard-negatives and the retrieval from which to construct the level 3 prediction sequences.
Construction of the sequences for each epoch is rather involved; Appendix E in the paper provides details on specifically which sequences are used for positive and negative instances for each level. For each level, we forward the query and support sequences together, and then subsequently calculate the difference vector. We forward each level separately, since each has a separate max length, but the backward pass is not calculated until the loss of all sequences, of all levels, of a given mini-batch is determined; hence, the joint learning. Also of note with training is that we iteratively freeze either the Transformer parameters or the CNN parameters, as with ``utils_ft_train.update_grad_status_of_cnn_model()`` and getting the BERT hidden layers from ``utils_ft_train.get_bert_representations_ft()`` vs. ``memory_match.get_bert_representations()``.

3. Setting aside search, the mechanics of dynamically creating the input sequences (adding prefixes, concatentating evidence, etc.) is straightforward in the abstract, but there are some complications in practice as each of the sequences is represented by three different sequence structures: The word embedding indexes (as direct input to the memory layers); the WordPiece embedding indexes (as input to BERT); and the masks (for BERT) corresponding to the WordPiece embeddings. We have to be careful about removing padding, removing/adding special symbols, and using the correct max length. For example, ``utils_search.construct_predicted_title_structures()`` is used for constructing level 2 support sequences and  ``utils_search.construct_predicted_title_structures_with_titles_list()`` is used for constructing level 3 support sequences. We can pre-calculate the query sequences (since they do not change during search), and additionally, for training, we can pre-calculate some of the ground-truth support sequences, as in ``utils.init_data_structure_for_split()``.

4. Note that in the current version, the datastores (i.e., Wikipedia sentences) are kept separate for training, dev, and test. That makes sense for FEVER to keep things straightforward, but there is potentially an additional possible efficiency gain for particularly wide search graphs for level 1. Namely, in training, if there is significant overlap between the support sequences of training and dev in level 1, we can batch them together, avoiding duplicate forward passes, which could make a difference if the number of sequences (and overlap) is very large. More generally, once the model is trained, we could in principle pre-calculate all (if a closed set, or some, e.g., high-frequency instances, if not) of the level 1 support sequences. (For a real-world deployment, there is a space/time tradeoff to be made here.)

5. By design, ``model.py`` only has properties for the memory layers, with the lower Transformer layers handled via input to the forward method; here, with the top layers from the getters ``utils_ft_train.get_bert_representations_ft()`` and ``memory_match.get_bert_representations()``, the latter of which is for frozen parameters. As such, it is relatively straightforward to substitute in an alternative Transformer (or other architecture) as input to the memory layers and for use with the coarse-to-fine search mechanism.

6. Combined with earlier work on classification analyzing token-level features (BLADE/mulitBLADE), the addition of this retrieval-classification memory matching mechanism yields a rather flexible set of tools for language modeling. I refer to this informally as 'Full Resolution Language/Sequence Modeling':

 * Updatability via the retrieval datastore
 * Updatability via exemplar auditing: Both in terms of difference vectors and token-level memories in the case
    of BLADE/mulitBLADE
 * Visualization (and associated analysis use cases) of level alignments and token-level contributions
 * Ability to constrain and analyze the model via level distances and exemplar distances
 * Ability to analyze corpora via the above and the sequence feature weighting as demonstrated in the BLADE paper
    (i.e., defacto extractive, comparative summarization).


# Acknowledgements

This repo incorporates code from a few additional public repos. If you use this repo, like their repos, as well.

1. The HuggingFace Transformer repo, the most recent iteration of which is here: https://github.com/huggingface/transformers

2. Taeuk Kim's PyTorch reimplementation (a version downloaded circa summer 2018) of Yoon Kim's 2014 paper ["Convolutional Neural Networks for Sentence Classification"](https://www.aclweb.org/anthology/D14-1181.pdf): https://github.com/galsang/CNN-sentence-classification-pytorch

3. [scorer.py](hhttps://github.com/sheffieldnlp/fever-scorer/tree/master/src/fever), used for local FEVER calculations, which is from the University of Sheffield's Natural Language Processing group: https://github.com/sheffieldnlp/fever-scorer

4. Here, and in the past, https://github.com/alvations/sacremoses has saved me time in generating consistent [de-]tokenizations.

Additionally, the re-annotated data of https://github.com/TalSchuster/FeverSymmetric adds a useful analysis test-bed for FEVER models.

More information about the Fact Extraction and VERification (FEVER) task is available at https://fever.ai/.

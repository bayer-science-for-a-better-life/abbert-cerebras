import os
import random
from math import floor, isclose
from pathlib import Path

import numpy as np
import tensorflow as tf
from abbert2.vendored.cerebras_modelzoo.common.input.utils import transform_dataset
from abbert2.vendored.cerebras_modelzoo.common.model_utils.shard_dataset import shard_dataset
from abbert2.vendored.cerebras_bayer_shared.bert.tf.input.Tokenization import BaseTokenizer
from abbert2.vendored.cerebras_bayer_shared.bert.tf.utils import get_oas_vocab


class OasMlmOnlyTfRecordsDynamicMaskProcessorCdrWeights:
    def __init__(self, params):
        self.data_dir = params["data_dir"]
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", True)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
        self.mp_type = (
            tf.float16 if params.get("mixed_precision") else tf.float32
        )

        assert self.batch_size > 0, "Batch size should be positive."
        assert (
            self.max_sequence_length > 0
        ), "Max sequence length should be positive."
        assert (
            self.max_predictions_per_seq > 0
        ), "Max predictions per seq should be positive."

        if not self.shuffle_buffer:
            self.shuffle_buffer = 10 * self.batch_size

        self.repeat = params.get("repeat", True)
        self.n_parallel_reads = params.get("n_parallel_reads", 4)

        # For sharding on CS-1, we need to explicitly retrieve `TF_CONFIG`.
        self.use_multiple_workers = params.get("use_multiple_workers", False)

        # No seed if not deterministic.
        self.deterministic = False
        self.rng = random.Random()

        (
            self.vocab_word_to_id_dict,
            self.min_aa_id,
            self.max_aa_id,
        ) = get_oas_vocab(params.get("vocab_type"), params.get("dummy_vocab_size"))

        self.vocab_words = list(self.vocab_word_to_id_dict.keys())

        self.tokenizer = BaseTokenizer(self.vocab_word_to_id_dict)
        self.special_tokens = ["[CLS]", "[PAD]", "[MASK]", "[SEP]"]
        self.special_tokens_word_id_dict = self._get_special_token_ids(
            self.special_tokens, self.tokenizer
        )

        self.fw_masked_lm_prob = params["fw_masked_lm_prob"]
        self.cdr_masked_lm_prob = params["cdr_masked_lm_prob"]
        self.use_segment_embedding = params["use_segment_embedding"]
        self.cdr3_loss_scale = params.get("cdr3_loss_scale", 2)

        self.species = params["species"]
        self.chain = params["chain"]

        self.subregion = params.get("subregion", None)
        self.subregion_overlap = params.get("subregion_overlap", 2)

        if self.subregion == "cdr" and self.cdr_masked_lm_prob:
            sum_cdr_prob = sum(self.cdr_masked_lm_prob)
            self.cdr_masked_lm_prob = [
                i / sum_cdr_prob for i in self.cdr_masked_lm_prob
            ]
            self.fw_masked_lm_prob = None
            tf.compat.v1.logging.info(
                f"cdr_masked_lm_prob: {self.cdr_masked_lm_prob}"
            )
            tf.compat.v1.logging.info(
                f"Since subregion is {self.subregion}, setting fw_masked_lm_prob to None"
            )

        elif self.subregion == "fw" and self.fw_masked_lm_prob:
            sum_fw_prob = sum(self.fw_masked_lm_prob)
            self.fw_masked_lm_prob = [
                i / sum_fw_prob for i in self.fw_masked_lm_prob
            ]
            self.cdr_masked_lm_prob = None
            tf.compat.v1.logging.info(
                f"fw_masked_lm_prob: {self.fw_masked_lm_prob}"
            )
            tf.compat.v1.logging.info(
                f"Since subregion is {self.subregion}, setting cdr_masked_lm_prob to None"
            )

        self.scale_mlm_weights = params.get("scale_mlm_weights", False)
        self._check_fw_cdr_masked_lm_prob(
            self.fw_masked_lm_prob, self.cdr_masked_lm_prob
        )

        self.input_files = []
        # When datadir is specified.
        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]
        tf.compat.v1.logging.debug(f"datadir: {self.data_dir}")

        for folder in self.data_dir:
            parent_folder, glob_pattern = os.path.split(folder)
            files_matched = list(Path(parent_folder).rglob(glob_pattern))
            files_str = [str(element) for element in files_matched]
            self.input_files.extend(files_str)

        # Sort files to ensure that workers get different files during sharding.
        self.input_files.sort()
        tf.compat.v1.logging.debug(f"globbed input files: {self.input_files}")

    def create_tf_dataset(
        self, mode=tf.estimator.ModeKeys.TRAIN, input_context=None
    ):
        """
        Create tf dataset.

        :param mode : Specifies whether the data is for training
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
        """

        is_training = mode == tf.estimator.ModeKeys.TRAIN

        self.deterministic = (not is_training) or not (
            self.shuffle and self.shuffle_seed is None
        )

        if self.deterministic:
            self.rng = random.Random(self.shuffle_seed)

        # This way is faster than using `list_files` when on remote storage systems.
        # https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files
        filelist = tf.data.Dataset.from_tensor_slices(
            tf.constant(self.input_files)
        )

        filelist = filelist.shuffle(self.shuffle_buffer, seed=self.shuffle_seed)

        filelist = shard_dataset(
            filelist, self.use_multiple_workers, input_context
        )

        # Note: ideally we should use self.deterministic flag to control num_parallel_calls
        # but this would make training dataloader very slow, so controlling data tighter during eval
        dataset = filelist.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=self.n_parallel_reads,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=1
            if not (is_training)
            else tf.data.experimental.AUTOTUNE,
            deterministic=self.deterministic,
        )

        # Note: ideally we should use self.deterministic flag to control num_parallel_calls
        # but this would make training dataloader very slow, so controlling data tighter during eval
        dataset = dataset.map(
            self._parse_raw_tfrecord,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=1
            if not (is_training)
            else tf.data.experimental.AUTOTUNE,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        # Filter out records where start id is greater than MSL
        dataset = dataset.filter(self._filter_fn)

        if self.species:
            dataset = dataset.filter(self._species_filter_fn)

        if self.chain:
            dataset = dataset.filter(self._chain_filter_fn)

        # Note: ideally we should use self.deterministic flag to control num_parallel_calls
        # but this would make training dataloader very slow, so controlling data tighter during eval
        return transform_dataset(
            dataset,
            self._map_fn,
            self.batch_size,
            is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            map_before_batch=True,
            # TODO: master vs 0.8.1
            num_parallel_calls=1
            if not (is_training)
            else tf.data.experimental.AUTOTUNE,
            post_batch_map_fn=self.post_batch_map_fn,
        )

    def _parse_raw_tfrecord(self, raw_record):
        feature_map = {"tokens": tf.io.FixedLenSequenceFeature((), tf.string)}
        context_features_map = {}

        ## Note: Uncomment the following lines only if these fields are needed for some processing
        # context_features_map["study_year"] = tf.io.FixedLenFeature([], tf.int64)
        # context_features_map["oas_subset"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["unit_id"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["study_id"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["v_call"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["j_call"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["subject"] = tf.io.FixedLenFeature([], tf.string)
        # context_features_map["index_in_unit"] = tf.io.FixedLenFeature([], tf.int64)

        context_features_map["species"] = tf.io.FixedLenFeature([], tf.string)
        context_features_map["chain"] = tf.io.FixedLenFeature([], tf.string)
        context_features_map["cdr_start"] = tf.io.FixedLenFeature([3], tf.int64)
        context_features_map["cdr_length"] = tf.io.FixedLenFeature(
            [3], tf.int64
        )
        context_features_map["fw_start"] = tf.io.FixedLenFeature([4], tf.int64)
        context_features_map["fw_length"] = tf.io.FixedLenFeature([4], tf.int64)

        context_features, raw_features = tf.io.parse_single_sequence_example(
            raw_record,
            sequence_features=feature_map,
            context_features=context_features_map,
        )

        features = {
            "tokens": raw_features["tokens"],
            "cdr_start": context_features["cdr_start"],
            "cdr_length": context_features["cdr_length"],
            "fw_start": context_features["fw_start"],
            "fw_length": context_features["fw_length"],
            "species": context_features["species"],
            "chain": context_features["chain"]
            ## Note: Uncomment the following lines only if these fields are needed for some processing
            # "study_year": context_features["study_year"],
            # "oas_subset": context_features["oas_subset"],
            # "species": context_features["species"],
            # "unit_id": context_features["unit_id"],
            # "study_id": context_features["study_id"],
        }

        return features

    def _filter_fn(self, raw_record):
        cdr_start = tf.math.reduce_all(
            tf.math.less(raw_record["cdr_start"], self.max_sequence_length - 2)
        )
        fw_start = tf.math.reduce_all(
            tf.math.less(raw_record["fw_start"], self.max_sequence_length - 2)
        )
        aligned_sequence = tf.math.less(
            tf.shape(raw_record["tokens"])[0], self.max_sequence_length - 2
        )
        cdr3_non_zero = tf.math.greater(raw_record["cdr_length"][-1], 0)

        is_less = tf.math.logical_and(
            tf.math.logical_and(cdr_start, fw_start),
            tf.math.logical_and(aligned_sequence, cdr3_non_zero),
        )

        # Remove sequences which have '*', 'X', 'Z'
        invalid_tokens = tf.reshape(tf.constant(['*', 'X', 'Z']), [3, 1])
        has_valid_tokens = tf.logical_not(
            tf.reduce_any(tf.math.equal(raw_record["tokens"], invalid_tokens))
        )

        is_valid = tf.math.logical_and(is_less, has_valid_tokens)

        # tf.compat.v1.logging.info(f"----{cdr_start}, {type(cdr_start)}")
        # tf.compat.v1.logging.info(f"----{raw_record}, {type(raw_record)}")
        return is_valid

    
    def _species_filter_fn(self, raw_record):

        is_species = tf.reduce_any(tf.math.equal(raw_record["species"], tf.constant(self.species)))
        return is_species

    def _chain_filter_fn(self, raw_record):
        is_chain = tf.math.equal(raw_record["chain"], tf.constant(self.chain))
        return is_chain

    def _map_fn(self, features):
        """
        Creates input features for pretraining.
        """
        # tf.compat.v1.logging.info(f"IN _MAP_FN")
        (
            input_ids,
            masked_lm_positions,
            masked_lm_ids,
            segment_ids,
            cdr3_weights,
        ) = tf.numpy_function(
            self._create_mlm_input_features,
            [
                features["tokens"],
                features["cdr_start"],
                features["cdr_length"],
                features["fw_start"],
                features["fw_length"],
            ],
            [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32],
        )
        input_ids.set_shape(self.max_sequence_length)
        masked_lm_positions.set_shape(self.max_predictions_per_seq)
        masked_lm_ids.set_shape(self.max_predictions_per_seq)
        cdr3_weights.set_shape(self.max_predictions_per_seq)
        input_mask = tf.cast(
            tf.equal(input_ids, self.vocab_word_to_id_dict["[PAD]"]), tf.int32
        )
        masked_lm_weights = tf.cast(
            tf.not_equal(masked_lm_positions, 0), self.mp_type
        )
        features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_weights": masked_lm_weights,
            "cdr3_weights": cdr3_weights
        }

        if self.use_segment_embedding:
            segment_ids.set_shape(self.max_sequence_length)
            features["segment_ids"] = segment_ids
        label = tf.zeros(())  # Pretraining is MLM only

        return features, label

    def _process_subregion(self, input_ids, start, length, subregion_overlap):

        sub_input_ids = []
        subregion_start = []
        subregion_len = []

        for k in range(len(start)):
            lower = max(0, start[k] - subregion_overlap)
            upper = min(
                len(input_ids), start[k] + length[k] + subregion_overlap
            )

            # Need to get correct start and len positions for the new list
            if k == 0:
                subregion_start.append(0 if lower == 0 else subregion_overlap)
            else:
                subregion_start.append(len(sub_input_ids) + subregion_overlap)
            subregion_len.append(length[k])

            sub_input_ids.extend(input_ids[lower:upper])
            sub_input_ids.append("[SEP]")

        return subregion_start, subregion_len, sub_input_ids

    def _create_mlm_input_features(
        self, tokens, cdr_start, cdr_length, fw_start, fw_length
    ):
        """
        Truncates, masks, and pads input ids.
        Note: If an `aligned_sequence` is greater than MSl, 
            then the `aligned_sequence` is truncated to MSL.
        When an id is masked, it is:
            - replaced with [MASK] 80% of the time.
            - replaced with a random amino acid 10% of the time.
            - left the same 10% of the time.

        :param array input_ids: sequence from a Parquet file.

        :returns: input_ids, masked_lm_positions, masked_lm_ids.
        """
        input_ids = [token.decode() for token in tokens]

        # tf.compat.v1.logging.info(f"cdr_start before {cdr_start}, {cdr_length}")
        # tf.compat.v1.logging.info(f"fw_start before {fw_start}, {fw_length}")

        # ### DEBUG  ###
        # for i, start_id in enumerate(cdr_start):
        #     end = start_id + cdr_length[i]
        #     for k in range(start_id, end):
        #         input_ids[k] = f"cdr{i+1}"

        # for i, start_id in enumerate(fw_start):
        #     end = start_id + fw_length[i]
        #     for k in range(start_id, end):
        #         input_ids[k] = f"fw{i+1}"

        # tf.compat.v1.logging.info(f"input_ids: {input_ids}, cdr_start before {cdr_start}, {cdr_length}, fw_start before {fw_start}, {fw_length}")

        if self.subregion:
            if self.subregion == "cdr":
                cdr_start, cdr_length, sub_input_ids = self._process_subregion(
                    input_ids, cdr_start, cdr_length, self.subregion_overlap
                )

            elif self.subregion == "fw":
                fw_start, fw_length, sub_input_ids = self._process_subregion(
                    input_ids, fw_start, fw_length, self.subregion_overlap
                )

            # Remove last [SEP] token added
            sub_input_ids.pop()
            input_ids = sub_input_ids

        num_ids = len(input_ids)
        num_pad_pos = self.max_sequence_length - (
            num_ids + 2
        )  # num_ids + 2 for CLS and SEP tokens

        input_ids = (
            [self.special_tokens_word_id_dict["[CLS]"]]
            + self.tokenizer.convert_tokens_to_ids(input_ids)
            + [self.special_tokens_word_id_dict["[SEP]"]]
            + [self.special_tokens_word_id_dict["[PAD]"]] * num_pad_pos
        )

        segment_ids = [0] * self.max_sequence_length
        if self.use_segment_embedding:
            # use 1 for FW regions and 0 for CDR so that we can have a differentiation where pad starts
            if self.subregion is None or self.subregion == "fw":
                segment_ids = [0] * self.max_sequence_length
                for strt, len_reg in zip(fw_start, fw_length):
                    segment_ids[strt + 1 : strt + 1 + len_reg] = [
                        1
                    ] * len_reg  # +1 in idx to account for CLS
            elif self.subregion == "cdr":
                segment_ids = [1] * self.max_sequence_length
                for strt, len_reg in zip(cdr_start, cdr_length):
                    segment_ids[strt + 1 : strt + 1 + len_reg] = [
                        0
                    ] * len_reg  # +1 in idx to account for CLS
                segment_ids[num_ids + 2 :] = [0] * num_pad_pos

        num_to_predict = min(
            self.max_predictions_per_seq,
            max(1, int(round(num_ids * self.masked_lm_prob))),
        )

        masked_lm_positions = self._get_masked_lm_positions(
            num_ids, num_to_predict, cdr_start, cdr_length, fw_start, fw_length
        )

        masked_lm_ids = [input_ids[pos] for pos in masked_lm_positions]
        for pos in masked_lm_positions:
            # Mask with `[MASK]` token 80% of time,
            # 10% replace with random token,
            # 10% retain original token.
            random_val = self.rng.random()
            if random_val < 0.8:
                input_ids[pos] = self.special_tokens_word_id_dict["[MASK]"]
            elif random_val < 0.9:
                input_ids[pos] = self.rng.randint(
                    self.min_aa_id, self.max_aa_id
                )

        masked_lm_padding = [0] * (
            self.max_predictions_per_seq - len(masked_lm_positions)
        )

        cdr3_weights = []
        for val in masked_lm_positions:
            if (val >= cdr_start[2]) and (val < cdr_start[2] + cdr_length[2]):
                cdr3_weights.append(1)
            else:
                cdr3_weights.append(0)
        
        masked_lm_positions += masked_lm_padding
        masked_lm_ids += masked_lm_padding
        cdr3_weights += masked_lm_padding

        return (
            np.int32(input_ids),
            np.int32(masked_lm_positions),
            np.int32(masked_lm_ids),
            np.int32(segment_ids),
            np.int32(cdr3_weights)
        )

    def _get_region_masked_lm_positions(
        self,
        start,
        length,
        num_to_predict,
        region_masked_lm_prob,
        num_input_ids,
    ):

        masked_lm_positions = []
        for i, id_len in enumerate(zip(start, length)):
            start_id, len_region = id_len
            start_id = start_id + 1  # To account for CLS token in index 0
            end_id = min(num_input_ids + 1, start_id + len_region)
            num_to_mask = min(
                len_region,
                end_id - start_id,
                max(1, int(floor(region_masked_lm_prob[i] * num_to_predict)),),
            )

            try:
                masked_lm_positions += self.rng.sample(
                    range(start_id, end_id), num_to_mask
                )

            except ValueError as e:
                tf.compat.v1.logging.info(
                    f"---cdr_{i} --- num_to_mask: {num_to_mask}, startid:{start_id}, end_id: {end_id}, start-end: {end_id - start_id}, len_region: {len_region}, num_to_predict: {num_to_predict}"
                )

                raise e

        return masked_lm_positions

    def _get_masked_lm_positions(
        self,
        num_input_ids,
        num_to_predict,
        cdr_start,
        cdr_length,
        fw_start,
        fw_length,
    ):
        """
        Function to get indices of masked tokens in the example.
        :param int num_input_ids: length of the example to be masked (excluding CLS and SEP token).
        :param int num_to_predict: Upper bound on the number of masked tokens.
        :param list cdr_start: List indicating starting index of each CDR region.
        :param list cdr_length: List indicating length of each CDR region.
        :param list fw_start: List indicating starting index of each FW region.
        :param list fw_length: List indicating length of each CDR region.

        :returns list masked_lm_positions A list of sorted indices of input_ids to be masked

        Note: 
            1. If `cdr_masked_lm_prob` and `fw_masked_lm_prob` are None, then a uniform masking probability of
                `masked_lm_prob` is applied over the entire `input_ids`.
            2. If any one of `cdr_masked_lm_prob` or `fw_masked_lm_prob` are not None, 
                then all the masking happens in those regions only.
            3. Even though saome of the regions may have a masking probability of 1.0, the number of tokens to be
                masked is still dependent on `max_predictions_per_seq` and `len(input_ids)` param.
                For ex: If the `fw_masked_lm_prob` = [0, 0, 0, 1.0] and `cdr_masked_lm_prob` is None
                and the length of `fw4` region is say 35 and `max_predictions_per_seq` is 20, then 
                only 20 random tokens from fw4 region would be masked.
        """

        if self.cdr_masked_lm_prob is None and self.fw_masked_lm_prob is None:
            masked_lm_positions = self.rng.sample(
                range(1, num_input_ids + 1), num_to_predict
            )

        else:
            cdr_masked_lm_positions = []
            fw_masked_lm_positions = []
            if self.cdr_masked_lm_prob:
                cdr_masked_lm_positions = self._get_region_masked_lm_positions(
                    cdr_start,
                    cdr_length,
                    num_to_predict,
                    self.cdr_masked_lm_prob,
                    num_input_ids,
                )

            if self.fw_masked_lm_prob:
                fw_masked_lm_positions = self._get_region_masked_lm_positions(
                    fw_start,
                    fw_length,
                    num_to_predict,
                    self.fw_masked_lm_prob,
                    num_input_ids,
                )

            masked_lm_positions = (
                cdr_masked_lm_positions + fw_masked_lm_positions
            )

        return sorted(masked_lm_positions)

    def _get_special_token_ids(self, special_tokens, tokenizer):
        """
        Function to get integer ids for special tokens.
        :param list special_tokens: special tokens in vocab.
        :param tokenizer: Tokenizer with function `check_word_and_get_id`
            which checks if the special token is part of vocab and returns it's id.
        :returns dict: dict with `special_tokens` as keys and their integer ids 
            as values.
        """
        special_tokens_word_id_dict = {}

        for key in special_tokens:
            special_tokens_word_id_dict[key] = tokenizer.check_word_and_get_id(
                key
            )

        return special_tokens_word_id_dict

    def _check_fw_cdr_masked_lm_prob(
        self, fw_masked_lm_prob, cdr_masked_lm_prob
    ):
        """
        Function to validate inputs to the `cdr_masked_lm_prob
        and `fw_masked_lm_prob`.
        """

        if fw_masked_lm_prob is not None and cdr_masked_lm_prob is not None:
            assert (
                len(fw_masked_lm_prob) == 4
            ), "Length of `fw_masked_lm_prob` is not 4"
            assert (
                len(cdr_masked_lm_prob) == 3
            ), "Length of `cdr_masked_lm_prob` is not 3"
            assert isclose(
                sum(fw_masked_lm_prob + cdr_masked_lm_prob), 1
            ), "Sum of probability for masking CDR and FW region is not 1"

        elif fw_masked_lm_prob is None and cdr_masked_lm_prob is not None:
            tf.compat.v1.logging.info(
                f"*** Since `fw_masked_lm_prob` is None, no values in FW region will be masked. ***"
            )
            assert (
                len(cdr_masked_lm_prob) == 3
            ), "Length of `cdr_masked_lm_prob` is not 3"
            assert isclose(
                sum(cdr_masked_lm_prob), 1
            ), "Sum of probability for masking CDR region is not 1"

        elif fw_masked_lm_prob is not None and cdr_masked_lm_prob is None:
            tf.compat.v1.logging.info(
                f"*** Since `cdr_masked_lm_prob` is None, no values in CDR region will be masked. ***"
            )
            assert (
                len(fw_masked_lm_prob) == 4
            ), "Length of `fw_masked_lm_prob` is not 4"
            assert isclose(
                sum(fw_masked_lm_prob), 1
            ), "Sum of probability for masking FW region is not 1"

        else:
            tf.compat.v1.logging.info(
                f"*** Since `cdr_masked_lm_prob` & `fw_masked_lm_prob` are None, the masking is done uniformly across the entire `aligned_sequence` ***"
            )

    def post_batch_map_fn(self, features, label):
        """
        When appropriate, scale mlm weights by `batch_size / num_valid_tokens`.
        This is used to compute the correct scaling factor on the loss without
        running into precision issues. Intended for use in situations when the
        loss will be divided by `batch_size` at the time of computation.
        """
        if self.scale_mlm_weights:
            mlm_weights = features["masked_lm_weights"]
            scale = self.batch_size / tf.reduce_sum(mlm_weights)
            features["masked_lm_weights"] = tf.cast(
                mlm_weights * scale, self.mp_type
            )

            features["cdr3_weights"] = tf.cast(tf.where(features["cdr3_weights"]>0, self.cdr3_loss_scale, 1), self.mp_type)
            features["masked_lm_weights"] = tf.cast(features["masked_lm_weights"] * features["cdr3_weights"], self.mp_type)

        return features, label

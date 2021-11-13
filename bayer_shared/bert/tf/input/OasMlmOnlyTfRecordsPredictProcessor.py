import random
from math import floor, isclose
from pathlib import Path

import numpy as np
import tensorflow as tf
import os

from bayer_shared.bert.tf.input.Tokenization import BaseTokenizer
from bayer_shared.bert.tf.utils import get_oas_vocab


class OasMlmOnlyTfRecordsPredictProcessor:
    def __init__(self, params):
        self.data_dir = params["data_dir"]
        self.batch_size = params["batch_size"]
        self.shuffle = params.get("shuffle", False)
        self.shuffle_seed = params.get("shuffle_seed", None)
        self.shuffle_buffer = params.get("shuffle_buffer", None)
        self.max_sequence_length = params["max_sequence_length"]
        self.max_predictions_per_seq = params["max_predictions_per_seq"]
        # self.masked_lm_prob = params.get("masked_lm_prob", 0.15)
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

        self.repeat = params.get("repeat", False)
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
        ) = get_oas_vocab(params.get("dummy_vocab_size"))
        self.vocab_words = list(self.vocab_word_to_id_dict.keys())

        self.tokenizer = BaseTokenizer(self.vocab_word_to_id_dict)
        self.special_tokens = ["[CLS]", "[PAD]", "[MASK]", "[SEP]"]
        self.special_tokens_word_id_dict = self._get_special_token_ids(
            self.special_tokens, self.tokenizer
        )

        self.subregion = params.get("subregion", None)
        self.subregion_overlap = params.get("subregion_overlap", 2)

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


    def _parse_raw_tfrecord(self, raw_record):
        feature_map = {"tokens": tf.io.FixedLenSequenceFeature((), tf.string)}
        context_features_map = {}

        ## Note: Uncomment the following lines only if these fields are needed for some processing       
        context_features_map["study_year"] = tf.io.FixedLenFeature([], tf.int64)
        context_features_map["oas_subset"] = tf.io.FixedLenFeature([], tf.string)
        context_features_map["normalized_species"] = tf.io.FixedLenFeature([], tf.string)
        context_features_map["unit_id"] = tf.io.FixedLenFeature([], tf.string)
        context_features_map["study_id"] = tf.io.FixedLenFeature([], tf.string)
        
        context_features_map["cdr_start"] = tf.io.FixedLenFeature([3], tf.int64)
        context_features_map["cdr_length"] = tf.io.FixedLenFeature([3], tf.int64)
        context_features_map["fw_start"] = tf.io.FixedLenFeature([4], tf.int64)
        context_features_map["fw_length"] = tf.io.FixedLenFeature([4], tf.int64)

        context_features, raw_features = tf.io.parse_single_sequence_example(
            raw_record, sequence_features=feature_map, context_features=context_features_map,
        )

        features = {
            "tokens": raw_features["tokens"],
            "cdr_start": context_features["cdr_start"],
            "cdr_length": context_features["cdr_length"],
            "fw_start": context_features["fw_start"],
            "fw_length": context_features["fw_length"],

            ## Note: Uncomment the following lines only if these fields are needed for some processing  
            "study_year": context_features["study_year"],
            "oas_subset": context_features["oas_subset"],
            "normalized_species": context_features["normalized_species"],
            "unit_id": context_features["unit_id"],
            "study_id": context_features["study_id"],
        }

        return features


    def _is_less_than_max_sequence_length(self, raw_record):
        cdr_start = tf.math.reduce_all(tf.math.less(raw_record["cdr_start"], self.max_sequence_length - 2))
        fw_start = tf.math.reduce_all(tf.math.less(raw_record["fw_start"], self.max_sequence_length - 2))
        aligned_sequence = tf.math.less(tf.shape(raw_record["tokens"])[0], self.max_sequence_length - 2)
        cdr3_non_zero = tf.math.greater(raw_record["cdr_length"][-1], 0)

        is_less = tf.math.logical_and(
            tf.math.logical_and(cdr_start, fw_start), 
            tf.math.logical_and(aligned_sequence, cdr3_non_zero)
            )
        
        # tf.compat.v1.logging.info(f"----{cdr_start}, {type(cdr_start)}")
        return is_less


    def _map_fn(self, features):
        """
        Creates input features for pretraining.
        """
        input_ids, masked_lm_positions, masked_lm_ids = tf.numpy_function(
            self._create_input_features,
            [
                features["tokens"],
                features["cdr_start"],
                features["cdr_length"],
                features["fw_start"],
                features["fw_length"],
            ],
            [tf.int32, tf.int32, tf.int32],
        )
        input_ids.set_shape(self.max_sequence_length)
        masked_lm_positions.set_shape(self.max_predictions_per_seq)
        masked_lm_ids.set_shape(self.max_predictions_per_seq)
        input_mask = tf.cast(
            tf.equal(input_ids, self.vocab_word_to_id_dict["[PAD]"]), tf.int32
        )
        masked_lm_weights = tf.cast(
            tf.not_equal(masked_lm_positions, 0), self.mp_type
        )

        out_features = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_weights": masked_lm_weights,
            "study_id": features["study_id"],
            "unit_id": features["unit_id"],
            "oas_subset": features["oas_subset"],
            "study_year": features["study_year"]
        }
        label = tf.zeros(())  # Pretraining is MLM only

        return out_features, label

    def _create_input_features(
        self, tokens, cdr_start, cdr_length, fw_start, fw_length
    ):
        """
        Truncates, masks, and pads input ids.
        Note: If an `aligned_sequence` is greater than MSl, 
            then the `aligned_sequence` is truncated to MSL.
        :param array input_ids: sequence from a Parquet file.

        :returns: input_ids, masked_lm_positions, masked_lm_ids.
        """

        # tf.compat.v1.logging.info(f"---- tokens {tokens}")
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

        # print(f"input_ids: {input_ids}")
            
        sub_input_ids = []
        subregion_start = []
        subregion_len = []
        
        if self.subregion:
            if self.subregion == "cdr":
                for k in range(len(cdr_start)):
                    lower = max(0, cdr_start[k] - self.subregion_overlap)
                    upper = min(len(input_ids), cdr_start[k] + cdr_length[k] + self.subregion_overlap)

                    # Need to get correct start and len positions for the new list
                    if k == 0:
                        subregion_start.append(0 if lower==0 else self.subregion_overlap)
                    else:
                        subregion_start.append(len(sub_input_ids))
                    subregion_len.append(cdr_length[k])

                    sub_input_ids.extend(input_ids[lower:upper])
                    sub_input_ids.append("[SEP]")
                    
                # tf.compat.v1.logging.info(f"subregion_start {subregion_start}, {subregion_len}")
                cdr_start = subregion_start
                cdr_length = subregion_len
                
            
            elif self.subregion == "fw":
                for k in range(len(fw_start)):
                    lower = max(0, fw_start[k] - self.subregion_overlap)
                    upper = min(len(input_ids), fw_start[k] + fw_length[k] + self.subregion_overlap)

                    # Need to get correct start and len positions for the new list
                    if k == 0:
                        subregion_start.append(0 if lower==0 else self.subregion_overlap)
                    else:
                        subregion_start.append(len(sub_input_ids))
                    subregion_len.append(fw_length[k])

                    sub_input_ids.extend(input_ids[lower:upper])
                    sub_input_ids.append("[SEP]")

                fw_start = subregion_start
                fw_length = subregion_len

            # Remove last [SEP] token added
            sub_input_ids.pop()
            input_ids = sub_input_ids
        
        # tf.compat.v1.logging.info(f"cdr_start after {cdr_start}, {cdr_length}")
        # tf.compat.v1.logging.info(f"fw_start after {fw_start}, {fw_length}")


        # DO NOT truncate
        # input_ids = tokens[: self.max_sequence_length - 2]

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

        ####### NO MASKING IN PREDICT PROCESSOR  ########
        # num_to_predict = min(
        #     self.max_predictions_per_seq,
        #     max(1, int(round(num_ids * self.masked_lm_prob))),
        # )

        # masked_lm_positions = self._get_masked_lm_positions(
        #     num_ids, num_to_predict, cdr_start, cdr_length, fw_start, fw_length
        # )

        # masked_lm_ids = [input_ids[pos] for pos in masked_lm_positions]
        # for pos in masked_lm_positions:
        #     # Mask with `[MASK]` token 80% of time,
        #     # 10% replace with random token,
        #     # 10% retain original token.
        #     random_val = self.rng.random()
        #     if random_val < 0.8:
        #         input_ids[pos] = self.special_tokens_word_id_dict["[MASK]"]
        #     elif random_val < 0.9:
        #         input_ids[pos] = self.rng.randint(self.min_aa_id, self.max_aa_id)

        # masked_lm_padding = [0] * (
        #     self.max_predictions_per_seq - len(masked_lm_positions)
        # )
        # masked_lm_positions += masked_lm_padding
        # masked_lm_ids += masked_lm_padding
        #############################################################

        masked_lm_positions = np.zeros((self.max_predictions_per_seq))
        masked_lm_ids = np.zeros((self.max_predictions_per_seq))

        return (
            np.int32(input_ids),
            np.int32(masked_lm_positions),
            np.int32(masked_lm_ids),
        )


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
        
        # Note: ideally we should use self.deterministic flag to control num_parallel_calls
        # but this would make training dataloader very slow, so controlling data tighter during eval
        dataset = filelist.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=self.n_parallel_reads,
            num_parallel_calls=1,
            deterministic=self.deterministic,
        )

        # Note: ideally we should use self.deterministic flag to control num_parallel_calls
        # but this would make training dataloader very slow, so controlling data tighter during eval
        dataset = dataset.map(
            self._parse_raw_tfrecord,
            num_parallel_calls=1,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        # Filter out records where start id is greater than MSL
        dataset = dataset.filter(self._is_less_than_max_sequence_length)

        dataset = dataset.map(
            self._map_fn,
            num_parallel_calls=1,
            # only allow nondeterminism when shuffling unseeded
            deterministic=self.deterministic
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=False)

        return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#!/bin/bash
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

input_folder=${1}
output_folder="/cb/ml/language/datasets/bayer_20211202/bayer_filter_default_20211202_tfrecs_hash"
hash_partition="1"

echo "input_folder: " ${input_folder}
echo "output_folder: " ${output_folder}
echo "hash_partition: " ${hash_partition}


python /cb/home/aarti/ws/code/bayer_tfrecs_filtering/tfrecord_scripts/create_tf_records.py \
   --src_input_folder=${input_folder} \
   --out_tf_records_fldr=${output_folder} \
   --hash_partition=${hash_partition} \

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

folder_path=${1}

echo ${folder_path}

cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
cpus=$((cpus - 1))
echo "Using $cpus CPU cores"

find ${folder_path} -name '*.parquet' ! -name 'nice_unit_meta*.parquet' ! -name 'bulk_download*.parquet' -exec dirname '{}' \; | xargs --max-args=1 --max-procs=$cpus  ./create_pretraining_data_wrapper.sh 
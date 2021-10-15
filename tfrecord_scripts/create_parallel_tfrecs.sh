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

file_path=${1}

echo ${file_path}

cpus=$( ls -d /sys/devices/system/cpu/cpu[[:digit:]]* | wc -w )
cpus=$((cpus - 2))
echo "Using $cpus CPU cores"


xargs -a ${file_path} --max-procs=$cpus --max-args=1 ./create_pretraining_data_wrapper.sh 
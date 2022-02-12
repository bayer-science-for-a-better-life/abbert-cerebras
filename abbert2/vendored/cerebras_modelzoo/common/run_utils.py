# Copyright 2021 Cerebras Systems.
#
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

"""
Defining run time utilities of estimator workflow for device-specific execution

Key functions include:
    is_cs: checks whether this is a CS1 runtime environment
    get_gpu_distribution_strategy: set up GPU distributed training
    save_params: save params in yaml format in model directory
    update_params_from_args: update command line arguments into params
    save_predictions: save predictions from estimator.predict into npy files
"""

import os

import numpy as np
import tensorflow as tf
import yaml


#######################################
## Utils for CS1 runtime
#######################################
def is_cs(params):
    """
    Check if the runtime enviroment is that of a CS-1.
    If yes, return True, else False

    :param dict params: runconfig dict to provide parameters for check
    """
    return params.get("cs_ip") is not None


def check_env(params):
    """
    Perform basic checks for parameters and env

    :param dict params: runconfig dict we want to validate
    """
    assert "mode" in params, "mode is required to check for runtime env"
    if is_cs(params):
        if params["mode"] == "eval_all":
            raise EnvironmentError("Cannot run eval_all mode on the CS-1!")
        if params["mode"] == "train_and_eval":
            raise EnvironmentError(
                "Cannot run train_and_eval mode on the CS-1!"
            )
        if params["mode"] == "eval" and params["eval_steps"] is None:
            raise ValueError(
                "When performing eval on CS-1, eval_steps must be provided."
            )


#######################################
## Utils for GPU runtime
#######################################
def _get_device_fn(params):
    """
    Gets the device_fn for GPU training

    :param dict params: runconfig dict with options to set device for training
    """
    device_fn = (lambda op: params["device"]) if params.get("device") else None
    return device_fn


def _get_gpu_distribution_strategy(params):
    """
    Gets the distribution strategy for multi-GPU scenarios to enable
    data-parallel training. If multiple workers are available, sets up a
    multi-worker multi-gpu training scheme. In the multi-worker scenario,
    this workflow assumes TF_CONFIG is set on the worker machines.

    :param dict params: dict with options to enable distribution strategies
    """
    if is_cs(params) and params.get("enable_distributed"):
        raise RuntimeError(
            "Running distributed training does not work on the CS-1!!"
            + " Please set the enable_distributed flag to False."
        )

    if params.get("enable_distributed"):
        if params.get("multiple_workers"):
            dist_strategy = (
                tf.distribute.experimental.MultiWorkerMirroredStrategy()
            )
        else:
            dist_strategy = tf.distribute.MirroredStrategy()
    else:
        dist_strategy = None

    return dist_strategy


#######################################
## Utils for handling args and params
#######################################
def save_params(params, model_dir, fname="params.yaml"):
    """
    Writes and saves a dictionary to a file in the model_dir.

    :param dict params: dict we want to write to a file in model_dir
    :param string model_dir: Directory we want to write to
    :param string fname: Name of file in model_dir we want to save to.
    """
    if not model_dir:
        raise ValueError(
            "model_dir is not provided. For saving params, user-defined"
            + " model_dir must be passed either through the command line"
            + " or from the yaml"
        )

    params_fname = os.path.join(model_dir, fname,)
    try:
        os.makedirs(os.path.dirname(params_fname), exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"Invalid path {model_dir} provided. Check the model_dir path!"
        )

    with open(params_fname, "w+") as _fout:
        yaml.dump(params, _fout, default_flow_style=False)


def update_params_from_args(args, params):
    """
    Sets command line arguments from args into params.

    :param argparse namespace args: Command line arguments
    :param dict params: runconfig dict we want to update
    """

    if args:
        for (k, v) in list(vars(args).items()):
            params[k] = v if v is not None else params.get(k)

    # Provisional handling of negative or 0 values. According to the estimator
    # source code passing negative or 0 steps raises an error in the estimator.
    # However, handling None in yaml is not straightforward. We have to pass
    # `null` there which is converted to None in python which is clumsy for users.
    if params.get("eval_steps") is not None:
        if params["eval_steps"] <= 0:
            params["eval_steps"] = None

    # setting cs_ip based on is_cs
    params["cs_ip"] = params["cs_ip"] + ":9000" if is_cs(params) else None


#######################################
## Utils for loading checkpoints
#######################################
def create_warm_start_settings(runconfig_params, exclude_string=None):
    """
    Creates warm start settings for estimator.

    Does not load any weights that include exclude string. This is useful when
    fine-tuning pretrained models.

    :param dict runconfig_params: runconfig params
    :param str exclude_string: any weights with this string in the name will be
        initialized from scratch instead of coming from the checkpoint.

    :returns: a WarmStartSettings object (or None if no checkpoint_path is
        provided) to be passed into estimator's warm_start_from field.
    """
    checkpoint_path = runconfig_params.get("checkpoint_path")
    if checkpoint_path is None:
        return None
    regex = f"^((?!{exclude_string}).)*$" if exclude_string else ".*"
    return tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=checkpoint_path, vars_to_warm_start=regex,
    )


#######################################
## Utils for inference workloads
#######################################
def _set_predict_directory(model_dir):
    """
    Make a `predict` directory within the given model_dir similar to `eval`

    :param string model_dir: Directory we want to write to
    """
    valid_predict_path = os.path.join(model_dir, "predict")
    try:
        os.makedirs(valid_predict_path, exist_ok=True)
    except OSError as e:
        raise ValueError(f"Path {valid_predict_path} is an invalid path!!")

    return valid_predict_path


def get_predict_directory(model_dir):
    """
    Gets the `predict` directory within the given model_dir if it exists

    :param string model_dir: Directory we want to write to
    """
    valid_predict_path = os.path.join(model_dir, "predict")
    if not os.path.isdir(valid_predict_path):
        raise ValueError(
            "Predictions not available. Please first run with --mode predict!"
        )

    return valid_predict_path


def save_predictions(model_dir, outputs, name="outputs.npz"):
    """
    Save outputs in give model_dir to give name, by initializing the `predict`
    dir within model_dir

    :param string model_dir: Directory we want to write to
    :param list outputs: List of dictionaries returned by estimator.predict
    :param string name: Name of output, generally in .npy format
    """
    predict_dir = _set_predict_directory(model_dir)
    fname = os.path.join(predict_dir, name)
    if os.path.exists(fname):
        tf.compat.v1.logging.warn(
            f"Output file with {name} exists in the model_dir."
            + " Overriding the file with new predictions!!"
        )

    outputs_to_save = []
    for output in outputs:
        outputs_to_save.append(output)
    if len(outputs_to_save) > 0:
        np.savez(fname, outputs_to_save)


#######################################
## Utils for csconfig/runconfig dicts
#######################################
def get_csrunconfig_dict(params):
    dist_strategy = _get_gpu_distribution_strategy(params)
    device_fn = _get_device_fn(params)

    keys_to_extract = [
        "tf_random_seed",
        "save_summary_steps",
        "save_checkpoints_steps",
        "keep_checkpoint_max",
        "log_step_count_steps",
    ]

    runconfig_dict = dict()
    for key in keys_to_extract:
        # update values if they exist, else fallback to defaults in CSRunConfig
        if key in params:
            runconfig_dict[key] = params[key]

    runconfig_dict["train_distribute"] = dist_strategy
    runconfig_dict["eval_distribute"] = dist_strategy
    runconfig_dict["device_fn"] = device_fn

    # Turn off standard estimator logging.
    # This may be needed when we want to
    # use a custom logging hook.
    if params.get("disable_standard_logs"):
        runconfig_dict["log_step_count_steps"] = None

    return runconfig_dict


def get_csconfig(params):
    """
    Returns CSConfig proto.
    """
    try:
        from cerebras.pb.tf.csconfig_pb2 import CSConfig

        cs_config = CSConfig(**params)
    except ImportError:
        cs_config = None

    return cs_config

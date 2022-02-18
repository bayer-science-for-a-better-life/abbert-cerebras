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

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from abbert2.vendored.cerebras_bayer_shared.bert.tf.data import eval_input_fn, train_input_fn
from abbert2.vendored.cerebras_bayer_shared.bert.tf.model import model_fn
from abbert2.vendored.cerebras_bayer_shared.bert.tf.utils import get_params
from abbert2.vendored.cerebras_modelzoo.common.estimator.tf.cs_estimator import CerebrasEstimator
from abbert2.vendored.cerebras_modelzoo.common.estimator.tf.run_config import CSRunConfig
from abbert2.vendored.cerebras_modelzoo.common.run_utils import (
    check_env,
    create_warm_start_settings,
    get_csconfig,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)
from abbert2.vendored.cerebras_modelzoo.transformers.bert.tf.utils import (
    get_custom_stack_params,
)

CS1_MODES = ["train", "eval"]


def create_arg_parser(default_model_dir):
    """
    Create parser for command line args.

    :param str default_model_dir: default value for the model_dir
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--cs_ip",
        default=None,
        help="CS-1 IP address, defaults to None. Ignored on GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of steps to run mode train."
            + " Runs repeatedly for the specified number."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Number of total steps to run in train mode.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Number of steps to run in eval or eval_all mode.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "eval_all", "predict",],
        help=(
            "Can train, eval, eval_all, or predict."
            + "  Train, eval, and predict will compile and train if on CS-1,"
            + "  and just run locally (CPU/GPU) if not on CS-1."
            + "  Eval_all will run eval locally for all available checkpoints."
        ),
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force model to run on a specific device (e.g., --device /gpu:0)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )

    return parser


def validate_params(params, cs1_modes):
    # check validate_only/compile_only
    runconfig_params = params["runconfig"]
    assert not (
        runconfig_params["validate_only"] and runconfig_params["compile_only"]
    ), "Please only use one of validate_only and compile_only."

    # check for gpu optimization flags
    if (
        runconfig_params["mode"] not in ["compile_only", "validate_only"]
        and not is_cs(runconfig_params)
        and not params["model"]["enable_gpu_optimizations"]
    ):
        tf.compat.v1.logging.warn(
            "Set enable_gpu_optimizations to True in model params "
            "to improve GPU performance."
        )

    # ensure runconfig is compatible with CS-1
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        assert runconfig_params["mode"] in cs1_modes, (
            "To run this model on CS-1, please use one of the following modes: "
            ", ".join(cs1_modes)
        )


def run(
    args,
    params,
    model_fn,
    train_input_fn=None,
    eval_input_fn=None,
    predict_input_fn=None,
    output_layer_name=None,
    cs1_modes=CS1_MODES,
):
    """
    Set up estimator and run based on mode

    :params dict params: dict to handle all parameters
    :params tf.estimator.EstimatorSpec model_fn: Model function to run with
    :params tf.data.Dataset train_input_fn: Dataset to train with
    :params tf.data.Dataset eval_input_fn: Dataset to validate against
    :params tf.data.Dataset predict_input_fn: Dataset to run inference on
    :params str output_layer_name: name of the output layer to be excluded
        from weight initialization when performing fine-tuning.
    """
    # update and validate runtime params
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params, cs1_modes)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # get cs-specific configs
    cs_config = get_csconfig(params.get("csconfig", dict()))
    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    stack_params = get_custom_stack_params(params)

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        cs_config=cs_config,
        stack_params=stack_params,
        **csrunconfig_dict,
    )
    warm_start_settings = create_warm_start_settings(
        runconfig_params, exclude_string=output_layer_name
    )
    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
        warm_start_from=warm_start_settings,
    )

    # execute based on mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        if runconfig_params["mode"] == "train":
            input_fn = train_input_fn
            mode = tf.estimator.ModeKeys.TRAIN
        elif runconfig_params["mode"] == "eval":
            input_fn = eval_input_fn
            mode = tf.estimator.ModeKeys.EVAL
        else:
            input_fn = predict_input_fn
            mode = tf.estimator.ModeKeys.PREDICT
        est.compile(
            input_fn, validate_only=runconfig_params["validate_only"], mode=mode
        )
    elif runconfig_params["mode"] == "train":
        est.train(
            input_fn=train_input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval":
        metrics_dict = est.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=runconfig_params["checkpoint_path"],
            steps=runconfig_params["eval_steps"],
            use_cs=use_cs,
        )
        # fname = os.path.join(runconfig_params["model_dir"], "eval", f"eval_{metrics_dict['global_step']}.npy")
        # np.save(fname, metrics_dict)
        return metrics_dict

    elif runconfig_params["mode"] == "eval_all":
        # fix metrics dict here
        ckpt_list = tf.train.get_checkpoint_state(
            runconfig_params["model_dir"]
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            metrics_dict = est.evaluate(
                eval_input_fn,
                checkpoint_path=ckpt,
                steps=runconfig_params["eval_steps"],
                use_cs=use_cs,
            )
            fname = os.path.join(runconfig_params["model_dir"], "eval", f"eval_{metrics_dict['global_step']}.npy")
            np.save(fname, metrics_dict)
    elif runconfig_params["mode"] == "predict":
        predictions = est.predict(
            input_fn=predict_input_fn,
            checkpoint_path=runconfig_params["checkpoint_path"],
            num_samples=runconfig_params["predict_steps"],
            use_cs=use_cs,
            yield_single_examples=True
        )
        
        return predictions


def main():
    """
    Main function
    """
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params, mode=args.mode)
    run(
        args=args,
        params=params,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
    )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()

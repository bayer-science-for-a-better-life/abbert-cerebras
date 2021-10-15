import argparse
import sys
import tensorflow as tf
import os



def create_arg_parser():
    """
    Create parser for command line args.
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_input_folder",
        required=True,
        help="src_input_folder",
    )
    parser.add_argument(
        "--out_tf_records_fldr",
        required=True,
        help="out_tf_records_fldr",
    )

    return parser


def create(inp, out):

    print(f"{inp}, {os.listdir(inp)}, {out}")


def main():
    """
    Main function
    """
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    create(args.src_input_folder, args.out_tf_records_fldr)
    


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
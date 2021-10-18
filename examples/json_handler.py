from pathlib import Path
import multiprocessing as mp
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
from typing import Optional, Union
from multiprocessing import Pool
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Preprocess')
parser.add_argument("-m", '--multiplier', type=int, default=1, help='random seed (default: 1111)')
parser.add_argument("-s", '--source_path', type=str, default="/Users/hossein/Desktop/paired", help='main path')
parser.add_argument("-d", '--dest_file', type=str, default="/Users/hossein/Desktop", help='dest file to log')
parser.add_argument("-c", '--chain_type', type=str, default="heavy", help='heavy or light')
parser.add_argument('--split', type=str, default="test", help='train or eval or test')
args = parser.parse_args()

if (args.split == "train"):
    split_type = "lte2017"
elif (args.split == "eval"):
    split_type = "eq2018"
elif (args.split == "test"):
    split_type = "gte2019"
else:
    raise Exception("split type not supported!")


def process_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data["max_aligned_sequence_length"], data["num_examples"]


def process_files(list_files, max_length, num_examples):
    final_max_length = max_length
    final_num_examples = num_examples
    for file in list_files:
        max_length, num_examples = process_json(file)
        final_max_length = max(final_max_length, max_length)
        final_num_examples += num_examples
    return final_max_length, final_num_examples


if __name__ == "__main__":
    print(mp.cpu_count())
    max_length = 0
    num_examples = 0
    num_workers = 16
    src_folder_path = args.source_path
    dest_file_path = os.path.join(args.dest_file,"{}_{}_{}_log.pkl".format(args.chain_type,args.split,args.multiplier))
    json_files = list(Path(src_folder_path).rglob("*{}*.json".format(args.chain_type + "_" + split_type)))
    json_files = sorted(json_files, key=lambda x: str(x))
    max_length, num_examples = process_files(json_files, max_length, num_examples)
    dict={"max_length":max_length,
          "num_examples":num_examples,
          "chain_type": args.chain_type,
          "split_type": args.split,
          "seed": args.multiplier}
    with open(dest_file_path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(dict, outp, pickle.HIGHEST_PROTOCOL)
    with open(dest_file_path, 'rb') as inp:
        tech_companies = pickle.load(inp)
        print("yes")

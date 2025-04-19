"""
Usage: python preprocess.py \
    --model-path lmsys/vicuna-7b-v1.5 \
    --model-id vicuna_v1.1 \
    --in-file raw_data/chat \
    --out-dir processed_data/sharegpt_processed_20k
"""

import sys
import os
import json
import argparse

from datasets import Dataset
from transformers import AutoTokenizer


def load_data(in_file, max_recursion_depth=1):
    if in_file.endswith('json'):
        with open(in_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif in_file.endswith('jsonl'):
        with open(in_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line.strip()))
    elif os.path.isdir(in_file) and max_recursion_depth > 0:
        data = []
        for file_or_folder in os.listdir(in_file):
            data.extend(
                load_data(os.path.join(in_file, file_or_folder), max_recursion_depth - 1)
            )
    else:
        raise ValueError(f"Loading script for {in_file} is not implemented.")
    
    return data


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, revision=args.revision, use_fast=False, trust_remote_code=True # Whether or not to allow for custom models defined on the Hub executing code on your local machine
    )
    
    def tokenize(data_point):
        result = tokenizer(
            data_point['text'], padding=False, truncation=True, max_length=args.max_length, return_tensors=None
        )
        result["labels"] = result["input_ids"].copy()
        return result
    
    data = load_data(args.in_file)
    data = Dataset.from_list(data)
    data = data.map(tokenize, num_proc=8, remove_columns=list(data.features))
    data = data.filter(lambda x: len(x['input_ids']) >= 2048, num_proc=8)
    print(data)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    data.save_to_disk(args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/wangqianle/models/llama2-7b",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument("--in-file", type=str, default="raw_data/base")
    parser.add_argument("--out-dir", type=str, default="processed_data/c4_processed_10k")
    parser.add_argument("--max-length", type=int, default=2048)

    args = parser.parse_args()
    
    main(args)
    

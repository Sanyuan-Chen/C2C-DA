# coding: utf-8

""" Surface Realization script"""

import argparse
import json
import random
import numpy as np
import torch
from realizers import ThesaurusRealizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='the path to the file to be filled')
    parser.add_argument('--values_path', required=True, help='the path to the value dict file.')
    parser.add_argument('--output_path', required=True, help='the path to the finished file.')

    parser.add_argument('--seed', type=int, default=42, help="the ultimate answer")

    parser.add_argument("--mode", default='thesaurus', type=str,
                        choices=['thesaurus'], help="Define methods of surface realization")
    parser.add_argument('--thesaurus_times', type=int, default=2, help='Times of re-fill for each sentences')
    parser.add_argument('--value_len_sampling', type=str, default='random',
                        choices=['random', 'gaussian'], help='method to get re-fill span length for each values')

    opt = parser.parse_args()
    return opt


def load_raw_sentences(opt):
    sents = []
    with open(opt.input_path, 'r') as reader:
        for line in reader:
            sents.append(line)
    return sents


def load_values_dict(opt):
    with open(opt.values_path, 'r') as reader:
        return json.load(reader)


def format_output(opt, realized_sentences):
    with open(opt.output_path, 'w') as writer:
        json.dump(realized_sentences, writer)


def main():
    opt = get_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    raw_sentences = load_raw_sentences(opt)
    values = load_values_dict(opt)

    realizer = ThesaurusRealizer(opt, values)

    realized_sentences = realizer.do_realize(raw_sentences)
    format_output(opt, realized_sentences)


if __name__ == "__main__":
    main()

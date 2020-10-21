import argparse
import os

import numpy as np

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv, save_csv


def shuffle(in_root, out_root, s, mode, num_epochs):
    meta = load_csv(os.path.join(in_root, '{}_{}.csv'.format(s, mode)))  # Not using query locations for this
    for e in range(num_epochs):
        out_file = os.path.join(out_root, '{}_{}_{:03d}.csv'.format(s, mode, e))
        if os.path.exists(out_file):
            print('{} exists. Not recalculating.'.format(out_file))
        else:
            print('Shuffling {}.'.format(out_file))
            shuffled_indices = np.random.permutation(len(meta['t']))

            shuffled_meta = dict()
            for key in meta.keys():
                shuffled_meta[key] = [meta[key][i] for i in shuffled_indices]
            save_csv(shuffled_meta, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/shuffled'))
    parser.add_argument('--max_epochs', default=5, type=int)
    args = parser.parse_args()

    flags_to_args(args)

    in_root = args.in_root
    out_root = args.out_root
    max_epochs = args.max_epochs

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for mode in ['ref', 'query']:
        for s in ['train', 'val', 'test']:
            shuffle(in_root, out_root, s, mode, max_epochs)

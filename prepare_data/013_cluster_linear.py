import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv
from learnlarge.util.io import save_pickle, load_pickle, save_csv


def get_xy(meta):
    return np.array([[e, n] for e, n in zip(meta['easting'], meta['northing'])], dtype=float)


def cluster(in_root, out_root, s, mode, r):
    out_file = os.path.join(out_root, '{}_{}_{}.pickle'.format(s, mode, r))

    meta_file = os.path.join(in_root, '{}_{}_000.csv'.format(s, mode))
    meta = load_csv(meta_file)

    if not os.path.exists(out_file):

        date = getattr(sys.modules[__name__], '{}_ref_date'.format(s))

        temp_meta = dict()
        for key in meta.keys():
            temp_meta[key] = [e for e, d in zip(meta[key], meta['date']) if d in date]

        t_idx = np.argsort(temp_meta['t'])
        date_meta = dict()
        for key in meta.keys():
            date_meta[key] = [temp_meta[key][i] for i in t_idx]

        print(len(date_meta['t']))
        xy = get_xy(date_meta)

        ref_xy = [xy[0, :]]
        ref_idx = [0]
        for i in tqdm(range(len(date_meta['t']))):
            if sum((xy[i, :] - ref_xy[-1]) ** 2) > r ** 2:
                ref_xy.append(xy[i, :])
                ref_idx.append(i)

        ref_xy = np.array(ref_xy)
        save_pickle([ref_xy, date_meta, ref_idx], out_file)
    else:
        ref_xy, date_meta, ref_idx = load_pickle(out_file)

    print('{}: {}'.format(s, len(ref_idx)))

    out_img = os.path.join(out_root, '{}_{}_{}.png'.format(s, mode, r))
    plt.clf()
    plt.clf()
    f, (ax1) = plt.subplots(1, 1, sharey=False)
    f.set_figheight(50)
    f.set_figwidth(50)
    ax1.scatter(ref_xy[:, 0], ref_xy[:, 1], c=np.arange(len(ref_xy)))
    plt.savefig(out_img)

    out_meta = dict()
    for key in meta.keys():
        out_meta[key] = [date_meta[key][i] for i in ref_idx]

    out_file = os.path.join(out_root, '{}_{}_{}.csv'.format(s, mode, r))
    save_csv(out_meta, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--test_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--val_ref_date', default='2014-05-14-13-50-20')  # Sunny, alternate-route
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/shuffled'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/clusters'))
    parser.add_argument('--num_clusters', type=dict, default={'train': 7000, 'test': 2000, 'val': 1000})
    parser.add_argument('--r', type=int, default=5)
    args = parser.parse_args()

    flags_to_args(args)

    in_root = args.in_root
    num_clusters = args.num_clusters
    out_root = args.out_root
    r = args.r
    test_ref_date = args.test_ref_date
    train_ref_date = args.train_ref_date
    val_ref_date = args.val_ref_date

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for mode in ['ref']:
        for s in ['train', 'val', 'test']:
            cluster(in_root, out_root, s, mode, r)

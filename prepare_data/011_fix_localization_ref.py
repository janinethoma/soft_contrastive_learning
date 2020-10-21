import argparse
import math
import os

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv, save_txt, save_csv
from learnlarge.util.io import load_img, save_img
from learnlarge.util.sampling import greedy


def img_path(info):
    date = info[0]
    folder = info[1]
    t = info[2]
    return os.path.join(img_root, '{}_stereo_centre_{:02d}'.format(date, int(folder)), '{}.png'.format(t))


def get_l_based_fixed_localization_reference(in_root, out_root, s, r):
    out_txt = os.path.join(out_root, '{}_ref_l_{}.txt'.format(s, int(r)))
    out_csv = os.path.join(out_root, '{}_ref_l_{}.csv'.format(s, int(r)))

    if not os.path.exists(out_csv):
        meta = load_csv(os.path.join(in_root, '{}_ref.csv'.format(s)))  # Not using query locations for this

        l = np.array(meta['l']).reshape(-1, 1)
        ll = np.arange(math.floor(l[-1]), step=r).reshape(-1, 1)

        l_tree = KDTree(l)
        i_l = l_tree.query(ll, return_distance=False, k=1)
        i_l = np.squeeze(i_l)

        save_txt('\n'.join(['{}'.format(i) for i in i_l]), out_txt)

        selected_meta = dict()
        for key in meta.keys():
            selected_meta[key] = [meta[key][i] for i in i_l]

        save_csv(selected_meta, out_csv)

    else:
        selected_meta = load_csv(out_csv)

    out_folder = os.path.join(out_root, '{}_ref_l_{}'.format(s, int(r)))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for i, (d, f, t) in tqdm(enumerate(zip(selected_meta['date'], selected_meta['folder'], selected_meta['t']))):
        f = int(f)
        img = load_img(img_path((d, f, t)))
        save_img(img, os.path.join(out_folder, '{:04d}_{}_{:02d}_{}.png'.format(i, d, f, t)))


def get_greedy_fixed_localization_reference(in_root, out_root, s, r):
    out_file = os.path.join(out_root, '{}_greedy_{}_ref.txt'.format(s, r))

    if not os.path.exists(out_file):
        meta = load_csv(os.path.join(in_root, '{}_ref.csv'.format(s)))  # Not using query locations for this

        xy = np.array([(e, n) for e, n in zip(meta['northing'], meta['easting'])], dtype=float)

        ref_ids = greedy(xy, 1)
        print(len(ref_ids))

        save_txt('\n'.join(['{}'.format(i) for i in ref_ids]), os.path.join(out_root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--test_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--val_ref_date', default='2014-05-14-13-50-20')  # Sunny, alternate-route
    parser.add_argument('--r', default='5', type=float)  # Sunny, alternate-route
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/localization_references'))
    parser.add_argument('--img_root',
                        default=os.path.join(fs_root(), 'datasets/oxford_512'))
    args = parser.parse_args()

    flags_to_args(args)

    img_root = args.img_root
    in_root = args.in_root
    out_root = args.out_root
    r = args.r
    test_ref_date = args.test_ref_date
    train_ref_date = args.train_ref_date
    val_ref_date = args.val_ref_date

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for s in ['train', 'test', 'val']:
        get_l_based_fixed_localization_reference(in_root, out_root, s, r)

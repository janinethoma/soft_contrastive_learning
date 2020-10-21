# Remove alternate-routes

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_pickle, load_pickle, save_csv


def clean_parametrization(in_root, folds, cols_to_keep, out_root):
    full_data = dict()
    full_ref_data = dict()
    full_query_data = dict()

    for key in cols_to_keep:
        full_data[key] = []
        full_ref_data[key] = []
        full_query_data[key] = []

    meta = dict()
    for s in folds:
        ref_data = load_csv(os.path.join(in_root, '{}_ref.csv'.format(s)))
        query_data = load_csv(os.path.join(in_root, '{}_query.csv'.format(s)))  # Not used to detect ref outliers

        for key in ['l', 'northing', 'easting']:
            ref_data[key] = np.array(ref_data[key], dtype=float)
            query_data[key] = np.array(query_data[key], dtype=float)

        l_max = max(ref_data['l'])
        num_bins = math.ceil(l_max)

        ref_member_path = os.path.join(out_root, '{}_ref_bin_raw_members.pickle'.format(s))
        if not os.path.exists(ref_member_path):
            bin_members = [[i for i in range(len(ref_data['t'])) if math.floor(ref_data['l'][i]) == j] for j in
                           tqdm(range(num_bins))]
            save_pickle(bin_members, ref_member_path)
        else:
            bin_members = load_pickle(ref_member_path)

        ref_bin_xy_path = os.path.join(out_root, '{}_ref_bin_raw_xy.pickle'.format(s))
        if not os.path.exists(ref_bin_xy_path):
            ref_bin_xy = [
                np.median(np.array([[ref_data['easting'][i], ref_data['northing'][i]] for i in bin_members[j]]),
                          axis=0) if len(
                    bin_members[j]) else np.array([-1, -1]) for j
                in tqdm(range(num_bins))]
            save_pickle(ref_bin_xy, ref_bin_xy_path)
        else:
            ref_bin_xy = load_pickle(ref_bin_xy_path)

        meta['{}_ref'.format(s)], clean_ref_data = find_and_remove_errors('ref', out_root, ref_bin_xy, ref_data, s)

        # Cleaning query files to allow for more efficient testing, should not influence performance
        # (other than possibly excluding faulty gps/ins 'ground truth', which we don't want anyways)
        meta['{}_query'.format(s)], clean_query_data = find_and_remove_errors('query', out_root, ref_bin_xy, query_data,
                                                                              s)

        fold_clean_data = dict()
        for key in clean_ref_data.keys():
            fold_clean_data[key] = []

            fold_clean_data[key].extend(clean_ref_data[key])
            fold_clean_data[key].extend(clean_query_data[key])

            full_data[key].extend(clean_ref_data[key])
            full_data[key].extend(clean_query_data[key])

            full_query_data[key].extend(clean_ref_data[key])
            full_ref_data[key].extend(clean_query_data[key])

        save_csv(fold_clean_data, os.path.join(out_root, '{}.csv'.format(s)))

    save_csv(full_data, os.path.join(out_root, 'full.csv'.format(s)))
    save_csv(full_ref_data, os.path.join(out_root, 'full_ref.csv'.format(s)))
    save_csv(full_query_data, os.path.join(out_root, 'full_query.csv'.format(s)))

    save_csv(meta, os.path.join(out_root, 'meta.csv'))


def find_and_remove_errors(mode, out_root, ref_bin_xy, ref_data, s):
    true_ref_xy = np.array([[e, n] for e, n in zip(ref_data['easting'], ref_data['northing'])])
    binned_ref_xy = np.array([ref_bin_xy[math.floor(l)] for l in ref_data['l']])
    ref_errors = np.linalg.norm(true_ref_xy - binned_ref_xy, axis=1)
    ref_hist_path = os.path.join(out_root, '{}_{}_bin_errors.png'.format(s, mode))
    if not os.path.exists(ref_hist_path):
        plt.clf()
        plt.hist(ref_errors, bins=1000, histtype='step')
        plt.savefig(ref_hist_path)
    for key in ref_data.keys():
        ref_data[key] = [el for el, er in zip(ref_data[key], ref_errors) if er < 5.0]
    save_csv(ref_data, os.path.join(out_root, '{}_{}.csv'.format(s, mode)))

    stats = dict()
    stats['raw_mean_error'] = np.mean(ref_errors)
    stats['raw_median_error'] = np.median(ref_errors)
    stats['raw_max_error'] = np.max(ref_errors)
    stats['raw_min_error'] = np.min(ref_errors)
    stats['raw_error_std'] = np.std(ref_errors)
    clean_errors = [er for er in ref_errors if er < 5.0]
    stats['clean_mean_error'] = np.mean(clean_errors)
    stats['clean_median_error'] = np.median(clean_errors)
    stats['clean_max_error'] = np.max(clean_errors)
    stats['clean_min_error'] = np.min(clean_errors)
    stats['clean_error_std'] = np.std(clean_errors)
    save_csv(stats, os.path.join(out_root, '{}_{}_errors.csv'.format(s, mode)))
    return len(ref_data['t']), ref_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/merged_parametrized'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/clean_merged_parametrized'))
    parser.add_argument('--folds', default=['train', 'val', 'test'])
    parser.add_argument('--cols_to_keep', default=['easting', 'northing', 'folder', 't', 'yaw', 'date', 'l'])
    args = parser.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    folds = args.folds
    cols_to_keep = args.cols_to_keep

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    clean_parametrization(in_root, folds, cols_to_keep, out_root)


if __name__ == '__main__':
    main()

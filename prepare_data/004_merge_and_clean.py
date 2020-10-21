import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_csv
from learnlarge.util.plot import dict_to_bar

bad_dates = [
    '2014-06-25-16-45-34',
    '2014-08-11-10-49-42',
    '2014-08-11-10-59-18',
    '2014-11-14-16-34-33',
    '2015-05-26-13-59-22',
    '2015-05-29-09-36-29',
    '2015-08-04-09-12-27',
    '2015-08-27-10-06-57',
    '2015-08-27-16-07-13',
]


def merge_dates(in_root, ins_root, out_root):
    # Find all dates with INS data (not all images have ins, but all ins should have images)
    all_dates = sorted(os.listdir(ins_root))  # Sort to make sure we always get the same order

    first = True
    all_info = dict()
    for date in all_dates:

        split_file = os.path.join(in_root, '{}.csv'.format(date))
        if not os.path.exists(split_file):
            print('Missing {}.'.format(split_file))
            continue

        date_info = load_csv(split_file)

        # Add date and tags column
        num_entries = len(date_info['easting'])
        rep_date = [date] * num_entries

        date_info['date'] = rep_date

        if first:
            all_info = date_info
            first = False
        else:
            for key in all_info.keys():
                all_info[key] = all_info[key] + date_info[key]

    out_file = os.path.join(out_root, 'merged.csv')
    save_csv(all_info, out_file)


def clean(in_root, out_root, folds, cols_to_keep):
    merged_file = os.path.join(in_root, 'merged.csv')
    meta_file = os.path.join(out_root, 'meta.csv')
    meta_info = dict()

    merged = load_csv(merged_file)

    # Original number of imgs
    meta_info['total_imgs'] = len(merged['exposure'])

    # Valid ins
    valid_ins = np.array(merged['ins_good'], dtype=int)
    meta_info['valid_ins'] = sum(valid_ins)

    # Valid location on grid
    valid_grid = np.array(merged['full'], dtype=int)
    meta_info['valid_grid'] = sum(valid_grid)

    # Analise and clean exposure
    # Visual inspection shows that images below 50'000'000 are very dark and above 110'000'000 very light
    exposures = np.array(merged['exposure'], dtype=float)
    low_exposure = np.percentile(exposures, 1)
    high_exposure = np.percentile(exposures, 99)
    print('Lo: {} \nHi: {}'.format(low_exposure, high_exposure))

    plt.clf()
    plt.hist(exposures, bins=10000, histtype='step')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(out_root, 'exposures.pdf'))

    valid_exposure = [1 if low_exposure < e < high_exposure else 0 for e in exposures]
    meta_info['valid_exposures'] = sum(valid_exposure)

    # Manual cleaning
    valid_date = [1 if d not in bad_dates else 0 for d in merged['date']]
    meta_info['valid_date'] = sum(valid_date)

    # Get fully valid
    fully_valid = np.array(valid_exposure) * np.array(valid_grid) * np.array(valid_ins) * np.array(valid_date)
    meta_info['fully_valid'] = sum(fully_valid)

    # Save for different folds
    for fold in folds:
        fold_valid = np.array(fully_valid) * np.array(merged[fold], dtype=int)
        meta_info['valid_{}'.format(fold)] = sum(fold_valid)

        out_data = dict()
        for col in cols_to_keep:
            out_col = [e for e, v in zip(merged[col], fold_valid) if v == 1]
            out_data[col] = out_col
        clean_file = os.path.join(out_root, 'clean_{}.csv'.format(fold))
        save_csv(out_data, clean_file)

        # Plot fold exposure:
        fold_exposure = [e for e, v in zip(exposures, fold_valid) if v == 1]
        plt.clf()
        plt.hist(fold_exposure, bins=10000, histtype='step')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(out_root, 'exposures_{}.pdf'.format(fold)))

    save_csv(meta_info, meta_file)
    dict_to_bar(meta_info, os.path.join(out_root, 'meta_info.pdf'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/splits'))
    parser.add_argument('--ins_root', default=os.path.join(fs_root(), 'data/datasets/oxford_extracted'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/merged'))
    parser.add_argument('--folds', default=['train', 'val', 'test', 'full'])
    parser.add_argument('--cols_to_keep', default=['easting', 'northing', 'folder', 't', 'yaw', 'date'])
    args = parser.parse_args()

    in_root = args.in_root
    ins_root = args.ins_root
    out_root = args.out_root
    folds = args.folds
    cols_to_keep = args.cols_to_keep

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    merge_dates(in_root, ins_root, out_root)
    clean(out_root, out_root, folds, cols_to_keep)


if __name__ == '__main__':
    main()

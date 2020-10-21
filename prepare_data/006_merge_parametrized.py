import argparse
import os

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_csv


def merge_parametrized(in_root, folds, cols_to_keep, out_root):
    files = os.listdir(in_root)

    meta_info = dict()

    full_data = dict()
    for c in cols_to_keep:
        full_data[c] = []

    for fold in folds:
        data = dict()
        date_count = dict()
        for c in cols_to_keep:
            data[c] = []

        fold_files = [f for f in files if f.split('_')[0] == fold]
        for file in fold_files:
            if '.csv' in file:
                date_data = load_csv(os.path.join(in_root, file))
                if len(date_data['t']) < 100:  # Very few files indicate bad l alignment or bad ins estimates
                    continue

                for c in cols_to_keep:
                    data[c].extend(date_data[c])
                    full_data[c].extend(date_data[c])
                date_count[file.split('_')[1]] = len(date_data['t'])
        out_file = os.path.join(out_root, '{}.csv'.format(fold))
        save_csv(data, out_file)
        meta_info[fold] = len(data['t'])
        save_csv(date_count, os.path.join(out_root, '{}_date_count.csv'.format(fold)))
    out_file = os.path.join(out_root, 'full.csv')
    save_csv(full_data, out_file)
    meta_info['full'] = len(full_data['t'])
    save_csv(meta_info, os.path.join(out_root, 'meta.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/parametrized'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/merged_parametrized'))
    parser.add_argument('--folds', default=['train', 'val', 'test'])
    parser.add_argument('--cols_to_keep', default=['easting', 'northing', 'folder', 't', 'yaw', 'date', 'l'])
    args = parser.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    folds = args.folds
    cols_to_keep = args.cols_to_keep

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    merge_parametrized(in_root, folds, cols_to_keep, out_root)


if __name__ == '__main__':
    main()

import argparse
import os

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv, save_csv


def set_aside_queries(in_root, folds, query_dates):
    num_per_fold = dict()

    for fold in folds:
        clean_file = os.path.join(in_root, '{}.csv'.format(fold))
        data = load_csv(clean_file)

        query_out = clean_file.replace(fold, '{}_query'.format(fold))
        ref_out = clean_file.replace(fold, '{}_ref'.format(fold))

        query_data = dict()
        ref_data = dict()

        for key in data.keys():
            query_data[key] = [el for el, date in zip(data[key], data['date']) if date in query_dates]
            ref_data[key] = [el for el, date in zip(data[key], data['date']) if date not in query_dates]

        num_per_fold['{}_query'.format(fold)] = len(query_data['t'])
        num_per_fold['{}_ref'.format(fold)] = len(ref_data['t'])
        save_csv(query_data, query_out)
        save_csv(ref_data, ref_out)
    save_csv(num_per_fold, os.path.join(in_root, 'num_per_fold.csv'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/merged_parametrized'))
    parser.add_argument('--folds', default=['train', 'val', 'test', 'full'])
    parser.add_argument('--query_dates', type=list, default=[
        '2015-08-14-14-54-57',  # roadworks, overcast
        '2014-11-18-13-20-12',  # sun, clouds
        '2014-12-17-18-18-43',  # night, rain
        '2015-02-03-08-45-10',  # snow
        '2014-06-26-09-24-58'  # overcast, alternate-route (validation area)
    ]
                        )
    args = parser.parse_args()
    print(flags_to_args(args))

    folds = args.folds
    in_root = args.in_root
    query_dates = args.query_dates

    set_aside_queries(in_root, folds, query_dates)


if __name__ == '__main__':
    main()

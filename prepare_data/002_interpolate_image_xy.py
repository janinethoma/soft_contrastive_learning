import argparse
import os
from math import pi

import numpy as np
from sklearn.neighbors import KDTree

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_csv
from learnlarge.util.sge import run_one_job


def create_array_job(ins_root, log_dir):
    run_one_job(script=__file__, queue='middle', cpu_only=True, memory=10, script_parameters=[], out_dir=log_dir,
                name='xy', overwrite=True, hold_off=True, num_jobs=len(os.listdir(ins_root)), array=True)


def lin_ip(x1, x2, dt1, dt2):  # Interpolate assuming linear change of place over time
    return (x1 * dt2 + x2 * dt1) / (dt1 + dt2)


def interpolate_xy(task_id, in_root, ins_root, out_root):
    # Find all dates with INS data (not all images have ins, but all ins should have images)
    all_dates = sorted(os.listdir(ins_root))  # Sort to make sure we always get the same order

    date = all_dates[int(task_id) - 1]

    out_file = os.path.join(out_root, '{}.csv'.format(date))
    if os.path.exists(out_file):
        # print('Already calculated {}.'.format(out_file))
        return

    imgs_file = os.path.join(in_root, '{}.csv'.format(date))
    if not os.path.exists(imgs_file):
        print('Missing {}: {}.'.format(task_id, imgs_file))
        return

    imgs = load_csv(imgs_file)
    ins = load_csv(os.path.join(ins_root, date, 'gps', 'ins.csv'))

    ins_ts = np.array(ins['timestamp'], dtype=int).reshape((-1, 1))  # num_samples x num_features
    img_ts = np.array(imgs['t'], dtype=int).reshape((-1, 1))
    northing = np.array(ins['northing'], dtype=float)
    easting = np.array(ins['easting'], dtype=float)
    yaw = np.array(ins['yaw'], dtype=float)  # Yaw range: 0-2pi
    status = ins['ins_status']

    # Ins measures are roughly 3 times more frequent than images
    mean_td_img = np.mean([img_ts[i, 0] - img_ts[i - 1, 0] for i in range(1, img_ts.shape[0])])
    mean_td_ins = np.mean([ins_ts[i, 0] - ins_ts[i - 1, 0] for i in range(1, ins_ts.shape[0])])
    print('Found {} times more ins measures than images.'.format(mean_td_img / mean_td_ins))
    print('The mean time between ins measures is {}.'.format(mean_td_ins))
    print('The mean time between img measures is {}.'.format(mean_td_img))

    ins_ts_tree = KDTree(ins_ts)
    d_closest, i_closest = ins_ts_tree.query(img_ts, 2)

    img_northing = [lin_ip(northing[i_c[0]], northing[i_c[1]], d_c[0], d_c[1]) for d_c, i_c in
                    zip(d_closest, i_closest)]
    img_easting = [lin_ip(easting[i_c[0]], easting[i_c[1]], d_c[0], d_c[1]) for d_c, i_c in
                   zip(d_closest, i_closest)]

    img_yaw = [lin_ip(yaw[i_c[0]], yaw[i_c[1]], d_c[0], d_c[1]) % (2 * pi) for d_c, i_c in
               zip(d_closest, i_closest)]  # Yaw range: 0-2pi

    # Remove interpolations of unclean ins states
    ins_good = [0] * len(img_easting)
    for j, i_c in enumerate(i_closest):
        if status[i_c[0]] == 'INS_SOLUTION_GOOD' and status[i_c[1]] == 'INS_SOLUTION_GOOD':
            ins_good[j] = 1

    imgs['northing'] = img_northing
    imgs['easting'] = img_easting
    imgs['ins_good'] = ins_good
    imgs['yaw'] = img_yaw

    ic1 = [i_c[0] for i_c in i_closest]
    ic2 = [i_c[1] for i_c in i_closest]
    tn1 = [ins_ts[i, 0] for i in ic1]
    tn2 = [ins_ts[i, 0] for i in ic2]

    imgs['ic1'] = ic1  # Index of closest ins point
    imgs['ic2'] = ic2
    imgs['tn1'] = tn1  # Timestamp of closest ins point
    imgs['tn2'] = tn2

    save_csv(imgs, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/img_info'))
    parser.add_argument('--ins_root', default=os.path.join(fs_root(), 'data/datasets/oxford_extracted'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/xy'))
    parser.add_argument('--log_dir', default=os.path.join(fs_root(), 'cluster_log/xy'))
    args = parser.parse_args()

    task_id = args.task_id
    in_root = args.in_root
    ins_root = args.ins_root
    out_root = args.out_root
    log_dir = args.log_dir

    if task_id == -1:
        create_array_job(ins_root, log_dir)
        interpolate_xy(98, in_root, ins_root, out_root)
    elif task_id == 0:
        for task_id in range(1, len(os.listdir(ins_root)) + 1):
            interpolate_xy(task_id, in_root, ins_root, out_root)
    else:
        interpolate_xy(task_id, in_root, ins_root, out_root)


if __name__ == "__main__":
    main()

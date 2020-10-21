import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_csv, save_csv
from learnlarge.util.sge import run_one_job


def create_array_job(ins_root, log_dir):
    run_one_job(script=__file__, queue='middle', cpu_only=True, memory=10, script_parameters=[], out_dir=log_dir,
                name='xy', overwrite=True, hold_off=False, num_jobs=len(os.listdir(ins_root)), array=True)


def draw_grid(X, Y, out_path):
    """
    :param X: easting - 619500 (int)
    :param Y: 5736480 - northing (int)
    """
    grid = np.zeros([1800, 1200])
    for x, y in zip(X, Y):
        if x < 0 or y < 0 or x >= grid.shape[1] or y >= grid.shape[0]:
            continue
        else:
            grid[y, x] = 255
    cv2.imwrite(out_path, grid)


def get_splits(task_id, grids, in_root, ins_root, out_root):
    # Find all dates with INS data (not all images have ins, but all ins should have images)
    all_dates = sorted(os.listdir(ins_root))  # Sort to make sure we always get the same order

    date = all_dates[int(task_id) - 1]
    print(date)

    out_file = os.path.join(out_root, '{}.csv'.format(date))
    if os.path.exists(out_file):
        print('Already calculated {}.'.format(out_file))
        return

    xy_file = os.path.join(in_root, '{}.csv'.format(date))
    if not os.path.exists(xy_file):
        print('Missing {}.'.format(xy_file))
        return

    xy = load_csv(xy_file)

    X = [0 if math.isnan(float(e)) else int(float(e) - 619500.0) for e in xy['easting']]
    Y = [0 if math.isnan(float(n)) else int(5736480.0 - float(n)) for n in xy['northing']]

    out_img_grid = os.path.join(out_root, '{}_grid.png'.format(date))
    draw_grid(X, Y, out_img_grid)

    out_img_scatter = os.path.join(out_root, '{}_scatter.png'.format(date))
    plt.clf()
    plt.scatter(np.array(xy['easting'], dtype=float), np.array(xy['northing'], dtype=float),
                c=np.array(xy['yaw'], dtype=float))
    plt.savefig(out_img_scatter)

    for grid_name in grids.keys():

        grid = cv2.imread(grids[grid_name])
        grid = np.asarray(grid, dtype=np.uint8)  # Fix for failing img loading

        in_fold = list()

        for x, y in zip(X, Y):
            if x < 0 or y < 0 or x >= grid.shape[1] or y >= grid.shape[0]:
                in_fold.append(0)
            elif grid[y, x, 0] > 0:  # All color channels are the same
                in_fold.append(1)
            else:
                in_fold.append(0)

        xy[grid_name] = in_fold

    max_assigned = [a1 + a2 + a3 for a1, a2, a3 in zip(xy['train'], xy['test'], xy['val'])]
    assert max(max_assigned) <= 1, 'Please increase in_fold grid threshold.'

    for grid_name in grids.keys():
        X_g = [x for x, in_fold in zip(X, xy[grid_name]) if in_fold == 1]
        Y_g = [y for y, in_fold in zip(Y, xy[grid_name]) if in_fold == 1]
        print('Found {} imgs in {} for {}.'.format(len(X_g), grid_name, date))
        out_img_file = os.path.join(out_root, '{}_{}.png'.format(date, grid_name))
        draw_grid(X_g, Y_g, out_img_file)
    save_csv(xy, out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=-1)
    parser.add_argument('--grids', type=dict,
                        default={'full': os.path.join(fs_root(), 'data/learnlarge/map_grids/full.png'),
                                 'test': os.path.join(fs_root(), 'data/learnlarge/map_grids/test.png'),
                                 'train': os.path.join(fs_root(), 'data/learnlarge/map_grids/train.png'),
                                 'val': os.path.join(fs_root(), 'data/learnlarge/map_grids/val.png')
                                 })
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/xy'))
    parser.add_argument('--ins_root', default=os.path.join(fs_root(), 'data/datasets/oxford_extracted'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/splits'))
    parser.add_argument('--log_dir', default=os.path.join(fs_root(), 'cpu_log/split'))
    args = parser.parse_args()

    task_id = args.task_id
    grids = args.grids
    in_root = args.in_root
    ins_root = args.ins_root
    out_root = args.out_root
    log_dir = args.log_dir

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if task_id == -1:
        create_array_job(ins_root, log_dir)
        get_splits(20, grids, in_root, ins_root, out_root)
    elif task_id == 0:
        for task_id in range(1, len(os.listdir(ins_root)) + 1):
            get_splits(task_id, grids, in_root, ins_root, out_root)
    else:
        get_splits(task_id, grids, in_root, ins_root, out_root)


if __name__ == '__main__':
    main()

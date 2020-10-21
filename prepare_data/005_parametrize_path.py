# Parametrize path to detect and remove alternate routes

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv, save_csv
from learnlarge.util.sge import run_one_job


def lin_ip(x1, x2, dt1, dt2):  # Interpolate assuming linear change of place over time
    return (x1 * dt2 + x2 * dt1) / (dt1 + dt2)


def create_array_job(num_jobs, log_dir):
    run_one_job(script=__file__, queue='short', cpu_only=True, memory=40, script_parameters=[], out_dir=log_dir,
                name='parametrize', overwrite=True, hold_off=False, num_jobs=num_jobs, array=True)


def create_reference(s):
    date = getattr(sys.modules[__name__], '{}_ref_date'.format(s))
    out_file = os.path.join(out_root, '{}_{}_geodesic.csv'.format(s, date))
    if not os.path.exists(out_file):

        data = load_csv(os.path.join(in_root, 'clean_{}.csv'.format(s)))

        ref_data = dict()
        for key in data.keys():
            ref_data[key] = [e for e, d in zip(data[key], data['date']) if
                             d == date]

        ref_xy = [(float(x), float(y)) for x, y in zip(ref_data['easting'], ref_data['northing'])]
        ref_d = [0] + [math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) for p, q in
                       zip(ref_xy[1:], ref_xy[:-1])]
        ref_l = [sum(ref_d[:i]) for i in range(1, len(ref_data['date']) + 1)]

        vmin = min(ref_l)
        vmax = max(ref_l)

        ref_data['l'] = ref_l
        ref_yaw = np.array(ref_data['yaw'], dtype=float)
        plot_results(ref_xy, ref_yaw, ref_l, date, ref_data, s, vmin, vmax)
        save_csv(ref_data, out_file)


def parametrize(s, date):
    ref_date = getattr(sys.modules[__name__], '{}_ref_date'.format(s))
    ref_file = os.path.join(out_root, '{}_{}_geodesic.csv'.format(s, ref_date))

    data = load_csv(os.path.join(in_root, 'clean_{}.csv'.format(s)))

    ref_data = load_csv(ref_file)
    ref_xy = [(float(x), float(y)) for x, y in zip(ref_data['easting'], ref_data['northing'])]

    ref_l = np.array(ref_data['l'], dtype=float)
    ref_yaw = np.array(ref_data['yaw'], dtype=float)

    ref_tree = KDTree(np.array(ref_xy))

    vmin = min(ref_l)
    vmax = max(ref_l)

    date_data = dict()
    for key in data.keys():
        date_data[key] = [e for e, d in zip(data[key], data['date']) if d == date]
    date_xy = [(float(x), float(y)) for x, y in zip(date_data['easting'], date_data['northing'])]
    date_d = [0] + [math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) for p, q in
                    zip(date_xy[1:], date_xy[:-1])]
    date_l = [sum(date_d[:i]) for i in range(1, len(date_d) + 1)]
    date_yaw = np.array(date_data['yaw'], dtype=float)

    matched_l = np.zeros(len(date_yaw))
    matchable = []
    r = 20
    if s == 'val':
        r = 100

    date_ni, date_nd = ref_tree.query_radius(np.array(date_xy), r=100, return_distance=True,
                                             sort_results=True)

    current_l = 0
    latest_valid = 0

    for j, (yaw, ni, nd) in enumerate(zip(date_yaw, date_ni, date_nd)):

        if len(ni) < 2:
            continue

        angle_neighbors = [i for i in range(len(ni)) if
                           abs(yaw - ref_yaw[ni[i]]) % (2 * math.pi) < math.pi / 3]

        ni = [ni[i] for i in angle_neighbors]
        nd = [nd[i] for i in angle_neighbors]

        if len(ni) < 2:
            continue

        potential_l = np.array([ref_l[i] for i in ni])

        if j == 0:
            threshold = 40
            if s == 'val':
                threshold = 5

            km = KMeans(n_clusters=2, random_state=0).fit(potential_l.reshape(-1, 1))
            if abs(km.cluster_centers_[0] - km.cluster_centers_[1]) > threshold:
                closest_center = km.predict(np.array(current_l).reshape(-1, 1))[0]
                assignments = km.labels_
                l_neighbors = [i for i, a in zip(range(len(ni)), assignments) if a == closest_center]
            else:
                l_neighbors = range(len(ni))
        else:
            l_neighbors = [i for i, l in enumerate(potential_l) if
                           abs(current_l - date_l[latest_valid] + date_l[j] - l) < 500]
        ni = [ni[i] for i in l_neighbors]
        nd = [nd[i] for i in l_neighbors]

        if len(ni) < 2:
            continue

        interp_l = lin_ip(ref_l[ni[0]], ref_l[ni[1]], nd[0], nd[1])
        current_l = interp_l
        latest_valid = j
        matched_l[j] = interp_l
        print(interp_l)
        matchable.append(j)

    if len(matchable) > 0:
        date_data['l'] = matched_l
        for key in ref_data.keys():
            date_data[key] = [date_data[key][i] for i in matchable]
        plot_results(date_xy, date_yaw, date_l, date, date_data, s, vmin, vmax)
        out_file = os.path.join(out_root, '{}_{}_geodesic.csv'.format(s, date))
        save_csv(date_data, out_file)


def plot_results(original_xy, date_yaw, date_l, date, date_data, s, vmin, vmax):
    out_img_scatter = os.path.join(out_root, '{}_{}_l_scatter.png'.format(s, date))
    plt.clf()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    f.set_figheight(7)
    f.set_figwidth(30)

    # Aligned l
    ax1.scatter([p[0] for p in original_xy], [p[1] for p in original_xy], c='black')
    m1 = ax1.scatter(np.array(date_data['easting'], dtype=float), np.array(date_data['northing'], dtype=float),
                     c=np.array(date_data['l'], dtype=float), vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(m1, cax=cax)
    ax1.title.set_text('Aligned l')
    ax1.set_xlabel('Easting [m]')
    ax1.set_ylabel('Northing [m]')

    # Original l
    m2 = ax2.scatter([p[0] for p in original_xy], [p[1] for p in original_xy], c=date_l, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(m2, cax=cax)
    ax2.title.set_text('Original l')
    ax2.set_xlabel('Easting [m]')
    ax2.set_ylabel('Northing [m]')

    # Original l
    m3 = ax3.scatter([p[0] for p in original_xy], [p[1] for p in original_xy], c=date_yaw)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(m3, cax=cax)
    ax3.title.set_text('Yaw')
    ax3.set_xlabel('Easting [m]')
    ax3.set_ylabel('Northing [m]')

    plt.savefig(out_img_scatter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--test_ref_date', default='2014-12-02-15-30-08')  # Overcast
    parser.add_argument('--val_ref_date', default='2014-05-14-13-50-20')  # Sunny, alternate-route
    parser.add_argument('--in_root', default=os.path.join(fs_root(), 'data/learnlarge/merged'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/parametrized'))
    parser.add_argument('--log_root', default=os.path.join(fs_root(), 'cpu_logs/learnlarge/parametrized'))
    parser.add_argument('--date_list', default=os.path.join(fs_root(), 'data/learnlarge/ins_dates.txt'))
    parser.add_argument('--task_id', default=-1, type=int)
    args = parser.parse_args()

    flags_to_args(args)

    date_list = args.date_list
    in_root = args.in_root
    log_root = args.log_root
    out_root = args.out_root
    task_id = args.task_id
    test_ref_date = args.test_ref_date
    train_ref_date = args.train_ref_date
    val_ref_date = args.val_ref_date

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    if not os.path.exists(log_root):
        os.makedirs(log_root)

    settings = list()

    # Get settings:
    sets = ['train', 'test', 'val']
    for s in sets:
        dates = sorted(list(set(load_csv(os.path.join(in_root, 'clean_{}.csv'.format(s)))['date'])))
        for date in dates:
            if not (s == 'val' and date in ['2014-05-14-13-59-05', '2014-05-14-13-53-47']):  # Wrong direction
                settings.append((s, date))

    if task_id == -1:
        for s in sets:
            create_reference(s)
        create_array_job(len(settings), log_root)
    else:
        setting = settings[task_id - 1]
        parametrize(s=setting[0], date=setting[1])

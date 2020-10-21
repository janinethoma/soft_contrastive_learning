import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree

from learnlarge.util.helper import flags_to_args, fs_root
from learnlarge.util.io import load_csv
from learnlarge.util.io import save_csv


def get_xy(meta):
    return np.array([[e, n] for e, n in zip(meta['easting'], meta['northing'])], dtype=float)


def sample_anchors(shuffled_root, cluster_root, out_root, s, mode, r, epoch):
    train_meta = load_csv(os.path.join(shuffled_root, '{}_{}_{:03d}.csv'.format(s, mode, epoch)))
    train_xy = get_xy(train_meta)

    out_file = os.path.join(out_root, '{}_{}_{}_{:03d}.csv'.format(s, mode, r, epoch))
    if not os.path.exists(out_file):

        ref_meata = load_csv(os.path.join(cluster_root, '{}_{}_{}.csv'.format(s, mode, r)))
        ref_xy = get_xy(ref_meata)

        # Sample reference images (random image withing r/2 of reference location)
        ref_tree = KDTree(train_xy)
        ref_neighbors = ref_tree.query_radius(ref_xy, r=1, return_distance=False)
        anchors = [np.random.choice(potential_anchors) for potential_anchors in ref_neighbors]

        np.random.shuffle(anchors)
        anchor_indices = {'idx': anchors}
        save_csv(anchor_indices, out_file)

    else:
        anchor_indices = load_csv(out_file)

    anchor_xy = np.array([train_xy[int(i), :] for i in anchor_indices['idx']])

    out_img = os.path.join(out_root, '{}_{}_{}_{}.png'.format(s, mode, r, epoch))
    plt.clf()
    plt.clf()
    f, (ax1) = plt.subplots(1, 1, sharey=False)
    f.set_figheight(50)
    f.set_figwidth(50)
    ax1.scatter(anchor_xy[:, 0], anchor_xy[:, 1], c=np.arange(len(anchor_xy)))
    plt.savefig(out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffled_root', default=os.path.join(fs_root(), 'data/learnlarge/shuffled'))
    parser.add_argument('--cluster_root', default=os.path.join(fs_root(), 'data/learnlarge/clusters'))
    parser.add_argument('--out_root', default=os.path.join(fs_root(), 'data/learnlarge/anchors'))
    parser.add_argument('--r', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=5)
    args = parser.parse_args()

    flags_to_args(args)

    cluster_root = args.cluster_root
    out_root = args.out_root
    r = args.r
    shuffled_root = args.shuffled_root
    max_epoch = args.max_epoch

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    for epoch in range(max_epoch):
        for mode in ['ref']:
            for s in ['train', 'val', 'test']:
                sample_anchors(shuffled_root, cluster_root, out_root, s, mode, r, epoch)

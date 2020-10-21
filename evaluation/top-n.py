import argparse
import os
from datetime import datetime

import numpy as np
from learnlarge.util.experiments import get_checkpoints
from learnlarge.util.helper import flags_to_globals
from learnlarge.util.helper import srv_root, mkdir
from learnlarge.util.io import load_csv, save_pickle, save_csv
from learnlarge.util.io import load_pickle
from learnlarge.util.meta import get_xy
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def get_top_n():
    # check if complete:
    ld_checkpoints = get_checkpoints('obm')

    ld_cp_names = []
    for cp in ld_checkpoints:
        cp_name = cp.split('/')[-2]
        cp_name = ''.join(os.path.basename(cp_name).split('.'))  # Removing '.'
        cp_name += '_e{}'.format(cp[-1])
        ld_cp_names.append(cp_name)

    if any([x in QUERY_LV_PICKLE for x in ld_cp_names]):
        L = [0.0, 0.3, 1.0, 5.0]
        D = [64, 128, 256, 512, 1024, 2048, 4096]
    else:
        L = [0.0]
        D = [256]

    complete = True
    for l in L:
        for d in D:

            out_folder = os.path.join(OUT_ROOT, 'l{}_dim{}'.format(l, d))
            name = ''.join(os.path.basename(QUERY_LV_PICKLE).split('.')[:-1])
            out_pickle = os.path.join(out_folder, '{}.pickle'.format(name))

            if not os.path.exists(out_pickle):
                complete = False
                break
        if not complete:
            break

    if complete:
        print('Skipping complete {}'.format(QUERY_LV_PICKLE))
        return

    ref_meta = load_csv(REF_CSV)
    query_meta = load_csv(QUERY_CSV)
    full_ref_xy = get_xy(ref_meta)
    full_query_xy = get_xy(query_meta)
    num_q = full_query_xy.shape[0]

    pca_f = np.array(load_pickle(PCA_LV_PICKLE))
    full_ref_f = np.array(load_pickle(REF_LV_PICKLE))
    full_query_f = np.array(load_pickle(QUERY_LV_PICKLE))

    full_xy_dists = pairwise_distances(full_query_xy, full_ref_xy, metric='euclidean')

    for d in D:

        print(d)
        pca = PCA(whiten=True, n_components=d)
        pca = pca.fit(pca_f)
        pca_ref_f = pca.transform(full_ref_f)
        pca_query_f = pca.transform(full_query_f)

        for l in L:
            print(l)

            out_folder = os.path.join(OUT_ROOT, 'l{}_dim{}'.format(l, d))
            mkdir(out_folder)
            name = ''.join(os.path.basename(QUERY_LV_PICKLE).split('.')[:-1])
            out_pickle = os.path.join(out_folder, '{}.pickle'.format(name))

            if os.path.exists(out_pickle):
                print('{} already exists. Skipping.'.format(out_pickle))
                continue

            ref_idx = [0]
            for i in range(len(full_ref_xy)):
                if sum((full_ref_xy[i, :] - full_ref_xy[ref_idx[-1], :]) ** 2) >= l ** 2:
                    ref_idx.append(i)

            if len(ref_idx) < N:
                continue

            ref_f = np.array([pca_ref_f[i, :] for i in ref_idx])
            xy_dists = np.array([full_xy_dists[:, i] for i in ref_idx]).transpose()

            print('Building tree')
            ref_tree = KDTree(ref_f)

            print('Retrieving')
            top_f_dists, top_i = np.array(ref_tree.query(pca_query_f, k=N, return_distance=True, sort_results=True))
            top_f_dists = np.array(top_f_dists)
            top_i = np.array(top_i, dtype=int)

            top_g_dists = [[xy_dists[q, r] for r in top_i[q, :]] for q in range(num_q)]

            gt_i = np.argmin(xy_dists, axis=1)
            gt_g_dist = np.min(xy_dists, axis=1)

            # Translate to original indices
            top_i = [[ref_idx[r] for r in top_i[q, :]] for q in range(num_q)]
            gt_i = [ref_idx[r] for r in gt_i]

            save_pickle([top_i, top_g_dists, top_f_dists, gt_i, gt_g_dist, ref_idx], out_pickle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--pca_lv_pickle',
                        default='/srv/beegfs02/scratch/toploc/data/neurips/lv/oxford_pca_loevil_quadruplet_ou512_repca_vl0_pca_neurips_002.pickle')
    parser.add_argument('--query_lv_pickle',
                        default='/srv/beegfs02/scratch/toploc/data/neurips/lv/oxford_snow_loevil_quadruplet_ou512_repca_vl0_pca_neurips_002.pickle')
    parser.add_argument('--ref_lv_pickle',
                        default='/srv/beegfs02/scratch/toploc/data/neurips/lv/oxford_fure_loevil_quadruplet_ou512_repca_vl0_pca_neurips_002.pickle')
    parser.add_argument('--query_csv', default='/srv/beegfs02/scratch/toploc/data/neurips/lists/oxford_snow.csv')
    parser.add_argument('--ref_csv', default='/srv/beegfs02/scratch/toploc/data/neurips/lists/oxford_fure.csv')

    parser.add_argument('--N', default=25, type=int)

    # Output
    parser.add_argument('--out_root', default=os.path.join(srv_root(), 'neurips/top_n'))
    parser.add_argument('--log_dir', default=os.path.join(srv_root(), 'neurips/logs/top_n'))

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    N = FLAGS.N

    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root
    QUERY_CSV = FLAGS.query_csv
    QUERY_LV_PICKLE = FLAGS.query_lv_pickle
    REF_CSV = FLAGS.ref_csv
    REF_LV_PICKLE = FLAGS.ref_lv_pickle
    PCA_LV_PICKLE = FLAGS.pca_lv_pickle

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'top_n_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    get_top_n()

    LOG.close()

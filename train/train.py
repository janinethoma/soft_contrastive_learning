import argparse
import math
import os
import random
import sys
from datetime import datetime
from queue import Queue
from threading import Thread, Lock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import tensorflow as tf
from learnlarge.model.losses import distance_triplet_loss, distance_quadruplet_loss, pairwise_distance_loss, \
    neg_eigenvalue_loss, evil_triplet_loss, evil_quadruplet_loss, ntuplet_evmm_loss, ntuplet_trace_loss, \
    residual_det_loss, residual_trace_loss, incremental_det_loss, incremental_residual_det_loss, incremental_mm_loss, \
    incremental_residual_mm_loss, ms_loss, ms_sum, ms_det, swrd_loss,  wrd_loss, wms_loss, prodwrd_loss, sumwrd_loss, logratio_loss
from learnlarge.model.nets import vgg16Netvlad, vgg16
from learnlarge.util.cv import put_text, merge_images, resize_img, standard_size
from learnlarge.util.helper import flags_to_globals
from learnlarge.util.helper import fs_root, debugging, location, srv_root
from learnlarge.util.io import load_img, load_csv, save_img
from learnlarge.util.sge import run_one_job
from pointnetvlad.pointnetvlad_cls import triplet_loss, lazy_triplet_loss, quadruplet_loss, lazy_quadruplet_loss
from sklearn.neighbors import KDTree
from time import time
from learnlarge.model.incremental_skl import skl_init, single_skl_increment, multiple_skl_increments
from learnlarge.model.mac import spp
from math import e

matplotlib.use('Agg')


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def decode(i):
    k = math.floor((1 + math.sqrt(1 + 8 * i)) / 2)
    return k, i - k * (k - 1) // 2


def rand_pair(n):
    return decode(random.randrange(n * (n - 1) // 2))


def rand_pairs(n, m):
    return [decode(i) for i in random.sample(range(n * (n - 1) // 2), m)]


def read_loss_pca_globals():
    global LOSS_PCA_LOCK

    with LOSS_PCA_LOCK:
        s = LOSS_S
        v = LOSS_V
        m = LOSS_M
        seen = LOSS_SEEN
        true_seen = LOSS_TRUE_SEEN
        var = LOSS_VAR

    return s, v, m, seen, true_seen, var


def write_loss_pca_globals(s, v, m, seen, true_seen, var):
    global LOSS_PCA_LOCK
    global LOSS_S
    global LOSS_V
    global LOSS_M
    global LOSS_SEEN
    global LOSS_TRUE_SEEN
    global LOSS_VAR

    with LOSS_PCA_LOCK:
        LOSS_S = s
        LOSS_V = v
        LOSS_M = m
        LOSS_SEEN = seen
        LOSS_TRUE_SEEN = true_seen
        LOSS_VAR = var


def read_pca_globals():
    global PCA_LOCK

    with PCA_LOCK:
        s = S
        v = V
        m = M
        seen = SEEN
        true_seen = TRUE_SEEN
        var = VAR

    return s, v, m, seen, true_seen, var


def write_pca_globals(s, v, m, seen, true_seen, var):
    global PCA_LOCK
    global S
    global V
    global M
    global SEEN
    global TRUE_SEEN
    global VAR

    with PCA_LOCK:
        S = s
        V = v
        M = m
        SEEN = seen
        TRUE_SEEN = true_seen
        VAR = var


def get_learning_rate(epoch):
    learning_rate = BASE_LR * (LR_DOWN_FACTOR ** (epoch // LR_DOWN_FREQUENCY))
    learning_rate = tf.maximum(learning_rate, MINIMAL_LR)
    return learning_rate


def img_path(info):
    date = info[0]
    folder = info[1]
    t = info[2]
    return os.path.join(IMG_ROOT, '{}_stereo_centre_{:02d}'.format(date, int(folder)), '{}.png'.format(t))


def localization_cpu_thread():
    global LOCALIZATION_CPU_IN_QUEUE
    global LOCALIZATION_GPU_IN_QUEUE
    while True:
        t = time()
        index, image_info = LOCALIZATION_CPU_IN_QUEUE.get()
        images = load_images(image_info)
        LOCALIZATION_GPU_IN_QUEUE.put((index, images), block=True)
        LOCALIZATION_CPU_IN_QUEUE.task_done()
        print('Loaded localization images in {}s.'.format(time() - t))


def localization_gpu_thread(sess, ops):
    global LOCALIZATION_GPU_IN_QUEUE
    global LOCALIZATION_GPU_OUT_QUEUE
    global FULL_FEATS
    while True:
        t = time()
        index, img = LOCALIZATION_GPU_IN_QUEUE.get()

        if FULL_FEATS:
            feat = sess.run(ops['full_out'], feed_dict={ops['input']: img})
        else:
            if 'pca' in REDUCTION:
                _, v, m, _, _, var = read_pca_globals()
                feat = sess.run(ops['output'], feed_dict={ops['input']: img, ops['v']: v, ops['var']: var, ops['m']: m})
            else:
                feat = sess.run(ops['output'], feed_dict={ops['input']: img})
        LOCALIZATION_GPU_OUT_QUEUE.put((index, feat))
        LOCALIZATION_GPU_IN_QUEUE.task_done()
        print('Inferred localization images in {}s.'.format(time() - t))


def eval_loss_cpu_thread(tuple_shape):
    global EVAL_LOSS_CPU_IN_QUEUE
    global EVAL_LOSS_GPU_IN_QUEUE

    current_epoch = EPOCH
    meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(OTHER_REF_SET, current_epoch)))
    xy = get_xy(meta)
    ref_tree = KDTree(xy)
    yaw = np.array(meta['yaw'], dtype=float)

    while True:
        t = time()
        original_indices = EVAL_LOSS_CPU_IN_QUEUE.get()
        if not EPOCH == current_epoch:
            current_epoch = EPOCH
            meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(OTHER_REF_SET, current_epoch)))
            xy = get_xy(meta)
            ref_tree = KDTree(xy)
            yaw = np.array(meta['yaw'], dtype=float)

        distances, image_info, used_indices = get_tuple(original_indices, tuple_shape, False, meta, xy, yaw, ref_tree)

        if len(image_info) == TUPLES_PER_BATCH * sum(tuple_shape):
            images = load_images(image_info)
            EVAL_LOSS_GPU_IN_QUEUE.put((distances, images), block=True)
        EVAL_LOSS_CPU_IN_QUEUE.task_done()
        print('Loaded eval loss tuples in {}s.'.format(time() - t))


def eval_loss_gpu_thread(sess, ops):
    global EVAL_LOSS_GPU_IN_QUEUE
    global EVAL_LOSS_GPU_OUT_QUEUE

    while True:
        t = time()
        distances, img = EVAL_LOSS_GPU_IN_QUEUE.get()
        feed_dict = {ops['input']: img, ops['epoch_num']: EPOCH}

        if not DISTANCE_TYPE == 'none':
            feed_dict[ops['distances']] = distances

        if 'incremental' in LOSS:
            s, v, m, seen, _, _ = read_loss_pca_globals()

            feed_dict.update({ops['l_v']: v, ops['l_m']: m, ops['l_seen']: seen, ops['l_s']: s})

        if 'pca' in REDUCTION:
            _, v, m, _, _, var = read_pca_globals()
            feed_dict.update({ops['v']: v, ops['var']: var, ops['m']: m})

        if not PN_LOSS:
            loss = sess.run(ops['loss'], feed_dict=feed_dict)
        else:
            loss_pos = sess.run(ops['loss'][0], feed_dict=feed_dict)
            loss_neg = sess.run(ops['loss'][1], feed_dict=feed_dict)
            loss = [loss_pos, loss_neg]

        EVAL_LOSS_GPU_OUT_QUEUE.put(loss)
        EVAL_LOSS_GPU_IN_QUEUE.task_done()
        print('Inferred eval loss tuples in {}s.'.format(time() - t))


def train_cpu_thread(tuple_shape):
    global TRAIN_CPU_IN_QUEUE
    global TRAIN_GPU_IN_QUEUE
    global USED_IMAGE_LOCK
    global USED_IMAGES

    current_epoch = EPOCH
    meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(LOCAL_REF_SET, current_epoch)))
    xy = get_xy(meta)
    ref_tree = KDTree(xy)
    yaw = np.array(meta['yaw'], dtype=float)

    while True:
        t = time()
        original_indices = TRAIN_CPU_IN_QUEUE.get()

        if not EPOCH == current_epoch:
            current_epoch = EPOCH
            meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(LOCAL_REF_SET, current_epoch)))
            xy = get_xy(meta)
            ref_tree = KDTree(xy)
            yaw = np.array(meta['yaw'], dtype=float)

        distances, image_info, used_indices = get_tuple(original_indices, tuple_shape, True, meta, xy, yaw, ref_tree)

        if len(image_info) == TUPLES_PER_BATCH * sum(tuple_shape):
            images = load_images(image_info)
            TRAIN_GPU_IN_QUEUE.put((distances, images), block=True)
            with USED_IMAGE_LOCK:
                USED_IMAGES.update(used_indices)
        else:
            log('Faulty training batch... ')
            log(image_info)
        TRAIN_CPU_IN_QUEUE.task_done()
        print('Loaded train tuples in {}s.'.format(time() - t))


def train_gpu_thread(sess, ops, train_writer):
    global TRAIN_GPU_IN_QUEUE
    global GLOBAL_STEP
    global GLOBAL_STEP_LOCK
    global TRAIN_GPU_OUT_QUEUE
    global LOSS_PCA_IN_QUEUE

    while True:
        t = time()
        distances, img = TRAIN_GPU_IN_QUEUE.get()
        feed_dict = {ops['input']: img, ops['epoch_num']: EPOCH, ops['keep_prob']: 0.5}
        if not DISTANCE_TYPE == 'none':
            feed_dict[ops['distances']] = np.array(distances, dtype=float)

        if 'incremental' in LOSS:
            s, v, m, seen, _, _ = read_loss_pca_globals()
            feed_dict.update({ops['l_v']: v, ops['l_m']: m, ops['l_seen']: seen, ops['l_s']: s})

        if 'pca' in REDUCTION:
            _, v, m, _, _, var = read_pca_globals()
            feed_dict.update({ops['v']: v, ops['var']: var, ops['m']: m})

        if not PN_LOSS:
            pca_in, loss_pca_in, summary, step, _, loss_test = sess.run(
                [ops['pca_in'], ops['loss_pca_in'], ops['merged'], ops['step'], ops['train_op'], ops['loss']],
                feed_dict=feed_dict)
            log('Train batch loss: {}'.format(loss_test))
        else:
            _, loss_test_pos = sess.run([ops['train_op'][0], ops['loss'][0]], feed_dict=feed_dict)
            pca_in, loss_pca_in, summary, step, _, loss_test_neg = sess.run(
                [ops['pca_in'], ops['loss_pca_in'], ops['merged'], ops['step'], ops['train_op'][1], ops['loss'][1]],
                feed_dict=feed_dict)
            log('Train batch loss pos: {}'.format(loss_test_pos))
            log('Train batch loss neg: {}'.format(loss_test_neg))

        if 'pca' in REDUCTION:
            TRAIN_GPU_OUT_QUEUE.put(pca_in)

        if 'incremental' in LOSS:
            LOSS_PCA_IN_QUEUE.put(loss_pca_in)

        train_writer.add_summary(summary, step)

        with GLOBAL_STEP_LOCK:
            GLOBAL_STEP = step
        TRAIN_GPU_IN_QUEUE.task_done()
        print('Training gpu step in {}s.'.format(time() - t))


def pca_cpu_thread(tuple_shape):
    global TRAIN_GPU_OUT_QUEUE

    while True:
        t = time()
        features = TRAIN_GPU_OUT_QUEUE.get()
        print('Starting PCA')

        if (not (features.shape[0] == TUPLES_PER_BATCH *
                 sum(tuple_shape))):
            print('Faulty feature shape in pca thread.')
        else:
            print('Reading PCA')
            s, v, m, seen, true_seen, _ = read_pca_globals()
            print('Single Increment PCA')
            s, v, m, seen, true_seen, var \
                = single_skl_increment(features, s, v, m, seen, true_seen, F)
            print('Writing Increment PCA')
            write_pca_globals(s, v, m, seen, true_seen, var)

        print('Updated PCA in {}s.'.format(time() - t))
        TRAIN_GPU_OUT_QUEUE.task_done()


def loss_pca_cpu_thread(tuple_shape):
    global LOSS_PCA_IN_QUEUE

    while True:
        t = time()
        features = LOSS_PCA_IN_QUEUE.get()
        print('Starting PCA')

        if (not (features.shape[0] == TUPLES_PER_BATCH *
                 sum(tuple_shape))):
            print('Faulty feature shape in pca thread.')
        else:
            print('Reading PCA')
            s, v, m, seen, true_seen, _ = read_loss_pca_globals()
            print('Single Increment PCA')
            s, v, m, seen, true_seen, var \
                = single_skl_increment(features, s, v, m, seen, true_seen, F)
            print('Writing Increment PCA')
            write_loss_pca_globals(s, v, m, seen, true_seen, var)

        print('Updated PCA in {}s.'.format(time() - t))
        LOSS_PCA_IN_QUEUE.task_done()


def evaluate_localization_thread(global_step, mode, nearest_d_dist, nearest_d_indices, nearest_latent_indices, out_name,
                                 query_image_info, query_xy, ref_image_info, ref_xy, writer):
    summary = tf.Summary()
    d_to_nearest_latent = np.empty(nearest_latent_indices.shape)
    for i in range(NUM_EVAL_QUERIES):
        for j in range(nearest_latent_indices.shape[1]):
            other_index = nearest_latent_indices[i][j]
            d_to_nearest_latent[i, j] = np.linalg.norm(query_xy[i, :] - ref_xy[other_index, :])
    top_n = np.empty(nearest_latent_indices.shape)
    for i in range(NUM_EVAL_QUERIES):
        for j in range(nearest_latent_indices.shape[1]):
            top_n[i, j] = min(d_to_nearest_latent[i, 0:(j + 1)])
    for rad in [50, 25, 10]:
        plt.clf()
        X = np.linspace(0, rad, num=25)
        for n in range(top_n.shape[1]):
            Y = [float(sum(top_n[:, n] < x)) / float(len(top_n[:, n])) * 100 for x in X]
            plt.plot(X, Y)
            if n == 0:
                auc = sklearn.metrics.auc(X, Y)
                summary.value.add(tag='{}m-auc@Top1'.format(rad), simple_value=auc)
                plt.text(0.5 * float(rad), 8, 'AUC@Top1={:7.2f}'.format(auc))

                correct_fraction = Y[-1]
                summary.value.add(tag='%<{}m@Top1'.format(rad), simple_value=correct_fraction)
                plt.text(0.5 * float(rad), 2, '%<{}m@Top1={:7.2f}'.format(rad, correct_fraction))
        Y = [float(sum(np.array(nearest_d_dist).reshape(-1) < x)) / float(len(top_n[:, n])) * 100 for x in X]
        plt.plot(X, Y)

        plt.legend(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5', 'Optimum'])
        plt.ylabel('Correctly localized')
        plt.xlabel('Tolerance [m]')
        plt.xlim(0, rad)
        title = os.path.basename(os.path.dirname(OUT_DIR)) + '\n' + os.path.basename(
            OUT_DIR) + '\n' + mode + ' ' + out_name
        plt.title(title)
        plt.savefig(os.path.join(OUT_DIR, mode + '_' + out_name + '_{}.pdf'.format(rad)))
    writer.add_summary(summary, global_step)
    # Save visual examples
    example_img_dir = os.path.join(OUT_DIR, mode + '_' + out_name)
    os.mkdir(example_img_dir)
    for index in np.random.choice(NUM_EVAL_QUERIES, 10, replace=False):
        # Query image
        query_file_path = img_path(query_image_info[index])
        query_image = load_img(query_file_path)
        query_image = put_text('Query', query_image)

        # Retrieved image
        file_path = img_path(ref_image_info[nearest_latent_indices[index][0]])
        retrieved_image = load_img(file_path)
        dist = d_to_nearest_latent[index][0]
        retrieved_image = put_text('Retrieved {}'.format(dist), retrieved_image)

        # Optimal image
        file_path = img_path(ref_image_info[nearest_d_indices[index][0]])
        optimal_image = load_img(file_path)
        dist = nearest_d_dist[index][0]
        optimal_image = put_text('Optimal {}'.format(dist), optimal_image)
        merged_images = merge_images(query_image, retrieved_image)
        merged_images = merge_images(merged_images, optimal_image)
        save_img(merged_images, os.path.join(example_img_dir, os.path.basename(query_file_path)))


def load_images(image_info):
    images = [[]] * len(image_info)
    for i in range(len(image_info)):
        if VLAD_CORES > 0:
            images[i] = resize_img(load_img(img_path(image_info[i])), 240)
        else:
            images[i] = standard_size(load_img(img_path(image_info[i])), h=180, w=240)
    return images


def get_tuple(original_indices, tuple_shape, use_hard_negatives, meta, xy, yaw, ref_tree):
    global USED_IMAGE_LOCK
    global USED_IMAGES
    global CACHED_FEATURE_LOCK
    global CACHED_FEATURE_TREE
    global CACHED_FEATURE_INDICES
    global CACHED_FEATURES
    t = time()

    distances = []
    image_info = []
    for index in original_indices:

        if use_hard_negatives:  # Reusing previously used images if they happen to be hard negatives
            with CACHED_FEATURE_LOCK:
                fis = np.where(CACHED_FEATURE_INDICES == index)[0]
                if len(fis) > 0:
                    fi = fis[0]
                    sorted_ni = CACHED_FEATURE_TREE.query(CACHED_FEATURES[fi, :].reshape(1, -1), k=MINING_CACHE_SIZE,
                                                          return_distance=False,
                                                          sort_results=True)[0]
                    true_sorted = [CACHED_FEATURE_INDICES[ni] for ni in sorted_ni]

        dirty_positives = np.setdiff1d(
            ref_tree.query_radius(xy[index, :].reshape(1, -1), r=MAX_POS_RADIUS, return_distance=False)[0], [index])
        potential_positives = [p for p in dirty_positives if abs(yaw[index] - yaw[p]) % (2 * math.pi) < (math.pi / 6.0)]

        hard_positives = []
        if use_hard_negatives and HARD_POSITIVES_PER_TUPLE > 0:
            for ti in reversed(true_sorted):
                if ti in potential_positives:
                    hard_positives.append(ti)
                    if len(hard_positives) >= HARD_POSITIVES_PER_TUPLE:
                        log('Found desired amount of hard negatives.')
                        break
        positives = np.random.choice(potential_positives, POSITIVES_PER_TUPLE - len(hard_positives))
        if len(hard_positives) > 0:
            positives = np.concatenate((positives, hard_positives)).tolist()

        excluded = set(ref_tree.query_radius(xy[index, :].reshape(1, -1), r=MIN_NEG_RADIUS)[0])
        hard_negatives = []
        if use_hard_negatives:  # Reusing previously used images if they happen to be hard negatives
            for ti in true_sorted:
                if ti not in excluded:
                    hard_negatives.append(ti)
                    if MUTUALLY_EXCLUSIVE_NEGS:
                        excluded.update(ref_tree.query_radius(xy[ti, :].reshape(1, -1), r=MIN_NEG_RADIUS)[0])
                    else:
                        excluded.add(ti)
                    if len(hard_negatives) >= HARD_NEGATIVES_PER_TUPLE:
                        log('Found desired amount of hard negatives.')
                        break
        num_neg_remaining = NEGATIVES_PER_TUPLE - len(hard_negatives)
        negatives = []
        while len(excluded) < len(yaw):
            remaining_negs = [i for i in np.arange(len(yaw)) if i not in excluded]
            if len(remaining_negs) == 0:
                log('Not enough negatives. Dropping batch.')
                return [], [], []
            next_i = np.random.choice(remaining_negs)
            negatives.append(next_i)
            if MUTUALLY_EXCLUSIVE_NEGS:
                excluded.update(ref_tree.query_radius(xy[next_i, :].reshape(1, -1), r=MIN_NEG_RADIUS)[0])
            else:
                excluded.add(ti)
            if len(negatives) >= num_neg_remaining:
                break
        negatives = np.concatenate((negatives, hard_negatives)).tolist()

        if len(tuple_shape) == 3:  # Anchor, positives and negatives
            tuple_indices = np.concatenate(([index], positives, negatives))

        elif len(tuple_shape) == 4:  # Anchor, positives, negatives, and other negative
            # get neighbors of negatives and query

            if not MUTUALLY_EXCLUSIVE_NEGS:
                original_negatives = excluded.copy()
                for original_negative in original_negatives:
                    excluded.update(
                        ref_tree.query_radius(xy[original_negative, :].reshape(1, -1), r=MIN_NEG_RADIUS)[0])

            remaining_negs = [i for i in np.arange(len(yaw)) if i not in excluded]

            if len(remaining_negs) == 0:
                log('Not enough negatives. Dropping batch.')
                return [], [], []
            other_neg = [np.random.choice(remaining_negs)]
            tuple_indices = np.concatenate(([index], positives, negatives, other_neg))
        else:
            log('Invalid tuple shape. Dropping batch.')
            return [], [], []

        if DISTANCE_TYPE == 'none':
            distances.append([])
        else:
            positive_locations = np.array([xy[i, :] for i in np.concatenate(([index], positives))], dtype=float)
            if DISTANCE_TYPE == 'anchor':
                anchor_location = xy[index, :]
                distances.append(
                    np.squeeze(
                        sklearn.metrics.pairwise_distances(positive_locations[1:], anchor_location.reshape(1, -1),
                                                           metric='sqeuclidean')))
            elif DISTANCE_TYPE == 'pairwise':
                distances.append(sklearn.metrics.pairwise_distances(positive_locations, positive_locations,
                                                                    metric='sqeuclidean'))
            elif 'wrd' in DISTANCE_TYPE:
                anchor_location = xy[index, :]
                pos_dists = np.squeeze(
                        sklearn.metrics.pairwise_distances(positive_locations[1:], anchor_location.reshape(1, -1),
                                                           metric='euclidean'))
                negative_locations = np.array([xy[int(i), :] for i in np.concatenate(([index], negatives))], dtype=float)
                neg_dists = np.squeeze(
                        sklearn.metrics.pairwise_distances(negative_locations[1:], anchor_location.reshape(1, -1),
                                                           metric='euclidean'))
                if DISTANCE_TYPE == 'swrd':
                    pos_weights =  1/(1+e**(ALPHA*(pos_dists-BETA )))
                    neg_weights = 1/(1+e**(ALPHA*(BETA - neg_dists)))
                    distances.append(np.concatenate((pos_weights,neg_weights)))

                else:
                    all_dists = np.concatenate((pos_dists, neg_dists))
                    pos_weights = 1 / (1 + e ** (ALPHA * (all_dists- BETA)))
                    neg_weights = 1 / (1 + e ** (ALPHA * (BETA - all_dists)))
                    distances.append(np.array(np.concatenate((pos_weights, neg_weights)), dtype=float))
            elif 'wms' in DISTANCE_TYPE:
                negative_locations = np.array([xy[int(i), :] for i in negatives],
                                              dtype=float)

                all_locations = np.concatenate((positive_locations, negative_locations), 0)
                distances.append(sklearn.metrics.pairwise_distances(all_locations, all_locations,
                                                           metric='euclidean'))
            elif 'logratio' in DISTANCE_TYPE:
                anchor_location = xy[index, :]
                negative_locations = np.array([xy[int(i), :] for i in np.concatenate(([index], negatives))],
                                              dtype=float)

                pos_dists = np.squeeze(sklearn.metrics.pairwise_distances(positive_locations[1:], anchor_location.reshape(1, -1), metric='sqeuclidean'))
                neg_dists = np.squeeze(sklearn.metrics.pairwise_distances(negative_locations[1:], anchor_location.reshape(1, -1), metric='sqeuclidean'))
                distances.append(np.concatenate((pos_dists,neg_dists)))


        if not len(tuple_indices) == sum(tuple_shape):
            log('Skipping batch with faulty tuple.')
            return [], [], []
        else:
            tuple_indices = np.array(tuple_indices, dtype=int)
            image_info.extend([(meta['date'][i], meta['folder'][i],
                                meta['t'][i]) for i in tuple_indices])
            print('Sampled tuple in {}s.'.format(time() - t))
    return distances, image_info, tuple_indices


def build_model():
    global NEGATIVES_PER_TUPLE
    global FULL_FEATS

    if 'quadruplet' in LOSS:
        # The last negative becomes 'other negative'
        NEGATIVES_PER_TUPLE = NEGATIVES_PER_TUPLE - 1
        tuple_shape = [1, POSITIVES_PER_TUPLE, NEGATIVES_PER_TUPLE, 1]
    else:
        tuple_shape = [1, POSITIVES_PER_TUPLE, NEGATIVES_PER_TUPLE]

    ops = dict()
    ops['is_training'] = tf.placeholder(dtype=tf.bool, shape=[1])

    if VLAD_CORES > 0 and 'spp' not in REDUCTION:
        ops['input'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH * sum(tuple_shape), None, None, 3])
    else:
        ops['input'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH * sum(tuple_shape), 180, 240, 3])
    ops['keep_prob'] = tf.placeholder_with_default(1.0, shape=())

    if REDUCTION == 'spp':
        ops['full_out'] = vgg16(ops['input'])
    else:
        if VLAD_CORES == 64:
            ops['full_out'] = vgg16Netvlad(ops['input'])
        else:
            ops['full_out'] = tf.layers.flatten(vgg16(ops['input']))

    ops['pca_in'] = tf.no_op()
    ops['loss_pca_in'] = tf.no_op()

    if REDUCTION == 'spp':
        ops['output'] = spp(ops['full_out'], L)

    # if REDUCTION == 'SPP':
    #     ops['output'] = rmac(ops['full_out'])
    #
    # elif REDUCTION == 'RCAT':
    #     ops['output'] = rcat(ops['full_out'])
    #
    # elif REDUCTION == 'MAC':
    #     ops['output'] = mac(ops['full_out'])

    elif REDUCTION == 'none':
        ops['output'] = ops['full_out']

    elif REDUCTION == '3fc':
        fc1 = tf.layers.dense(ops['full_out'], units=4096, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(fc1, ops['keep_prob'], name='fc1')
        fc2 = tf.layers.dense(dropout1, units=4096, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(fc2, ops['keep_prob'], name='fc2')
        ops['output'] = tf.layers.dense(dropout2, units=OUT_DIM, name='fc3')

    elif REDUCTION == '2fc':
        fc1 = tf.layers.dense(ops['full_out'], units=4096, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(fc1, ops['keep_prob'], name='fc1')
        ops['output'] = tf.layers.dense(dropout1, units=OUT_DIM, name='fc2')

    elif REDUCTION == '1fc':
        ops['output'] = tf.layers.dense(ops['full_out'], units=OUT_DIM, name='fc1')

    elif REDUCTION == 'pca':
        ops['v'] = tf.placeholder(shape=[OUT_DIM, ops['full_out'].shape[-1]], dtype=tf.float32)
        ops['m'] = tf.placeholder(shape=[ops['full_out'].shape[-1]], dtype=tf.float32)
        ops['var'] = tf.placeholder(shape=[OUT_DIM], dtype=tf.float32)
        x_tf = tf.matmul(ops['full_out'] - ops['m'], ops['v'], adjoint_b=True)
        ops['output'] = tf.truediv(x_tf, tf.sqrt(ops['var']))
        ops['pca_in'] = ops['full_out']

    ops['outputs'] = tf.split(tf.reshape(ops['output'], [TUPLES_PER_BATCH, sum(tuple_shape), -1]), tuple_shape, 1)
    ops['step'] = tf.Variable(0)  # Updated when calling optimizer.minimize() i.e. train_op
    ops['epoch_num'] = tf.placeholder(tf.float32, shape=())

    if 'incremental' in LOSS:
        ops['l_s'] = tf.placeholder(shape=[LOSS_DIM], dtype=tf.float32)
        ops['l_seen'] = tf.placeholder(shape=(), dtype=tf.float32)
        ops['l_v'] = tf.placeholder(shape=[LOSS_DIM, ops['output'].shape[-1]], dtype=tf.float32)
        ops['l_m'] = tf.placeholder(shape=[ops['output'].shape[-1]], dtype=tf.float32)

    # Some losses use additional placeholders and parameters
    if DISTANCE_TYPE == 'pairwise':
        ops['distances'] = tf.placeholder(dtype=tf.float32,
                                          shape=[TUPLES_PER_BATCH, POSITIVES_PER_TUPLE + 1, POSITIVES_PER_TUPLE + 1])
    elif DISTANCE_TYPE == 'anchor':
        ops['distances'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH, POSITIVES_PER_TUPLE])

    elif DISTANCE_TYPE == 'swrd':
        ops['distances'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH, POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE])
        split_dists = tf.split(tf.reshape(ops['distances'], [TUPLES_PER_BATCH, POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE, -1]), [POSITIVES_PER_TUPLE, NEGATIVES_PER_TUPLE], 1)
        ops['pos_weights'] = split_dists[0]
        ops['neg_weights'] = split_dists[1]

    elif DISTANCE_TYPE == 'wrd':
        ops['distances'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH, 2*POSITIVES_PER_TUPLE + 2*NEGATIVES_PER_TUPLE])
        split_dists = tf.split(tf.reshape(ops['distances'], [TUPLES_PER_BATCH, 2*(POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE), -1]), [POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE, POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE], 1)
        ops['pos_weights'] = split_dists[0]
        ops['neg_weights'] = split_dists[1]


    elif DISTANCE_TYPE == 'wms':
        ops['distances'] = tf.placeholder(dtype=tf.float32,
                                          shape=[TUPLES_PER_BATCH, sum(tuple_shape), sum(tuple_shape)])
    elif DISTANCE_TYPE == 'logratio':
        ops['distances'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH, POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE])
        split_dists = tf.split(tf.reshape(ops['distances'], [TUPLES_PER_BATCH, (POSITIVES_PER_TUPLE + NEGATIVES_PER_TUPLE), -1]), [POSITIVES_PER_TUPLE , NEGATIVES_PER_TUPLE], 1)
        ops['pos_distances'] = split_dists[0]
        ops['neg_distances'] = split_dists[1]


    if not DISTANCE_TYPE == 'none':
        d_max_squared = float(MAX_POS_RADIUS) ** 2
        f_max_squared = 2.0  # Calculated from 10'000 train ref features

        # Add loss to network
    # Losses from PointNetVLAD
    if LOSS == 'triplet':
        ops['loss'] = triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)
    elif LOSS == 'lazy_triplet':
        ops['loss'] = lazy_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)
    elif LOSS == 'evil_triplet':
        ops['loss'] = evil_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)
    elif LOSS == 'quadruplet':
        ops['loss'] = quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['outputs'][3],
                                      MARGIN_1, MARGIN_2)
    elif LOSS == 'lazy_quadruplet':
        ops['loss'] = lazy_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                           ops['outputs'][3],
                                           MARGIN_1, MARGIN_2)
    elif LOSS == 'evil_quadruplet':
        ops['loss'] = evil_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                           ops['outputs'][3],
                                           MARGIN_1, MARGIN_2)

    # Losses with simple distance term
    elif LOSS == 'distance_triplet':
        ops['loss'] = distance_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1, LAM,

                                            ops['distances'], d_max_squared, f_max_squared, 'triplet_loss',
                                            'distance_loss')
    elif LOSS == 'distance_lazy_triplet':
        ops['loss'] = distance_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1, LAM,
                                            ops['distances'], d_max_squared, f_max_squared, 'lazy_triplet_loss',
                                            'distance_loss')
    elif LOSS == 'distance_quadruplet':
        ops['loss'] = distance_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                               ops['outputs'][3],
                                               MARGIN_1, MARGIN_2, LAM, ops['distances'], d_max_squared,
                                               f_max_squared,
                                               'triplet_loss', 'distance_loss')
    elif LOSS == 'distance_lazy_quadruplet':
        ops['loss'] = distance_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                               ops['outputs'][3],
                                               MARGIN_1, MARGIN_2, LAM, ops['distances'], d_max_squared,
                                               f_max_squared,

                                               'lazy_triplet_loss', 'distance_loss')

    # Losses with huber distance term
    elif LOSS == 'huber_distance_triplet':
        ops['loss'] = distance_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1, LAM,

                                            ops['distances'], d_max_squared, f_max_squared, 'triplet_loss',
                                            'huber_distance_loss')
    elif LOSS == 'huber_distance_lazy_triplet':
        ops['loss'] = distance_triplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1, LAM,
                                            ops['distances'], d_max_squared, f_max_squared, 'lazy_triplet_loss',
                                            'huber_distance_loss')
    elif LOSS == 'huber_distance_quadruplet':
        ops['loss'] = distance_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                               ops['outputs'][3],
                                               MARGIN_1, MARGIN_2, LAM, ops['distances'], d_max_squared,
                                               f_max_squared,
                                               'triplet_loss', 'huber_distance_loss')
    elif LOSS == 'huber_distance_lazy_quadruplet':
        ops['loss'] = distance_quadruplet_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2],
                                               ops['outputs'][3],
                                               MARGIN_1, MARGIN_2, LAM, ops['distances'], d_max_squared,
                                               f_max_squared,
                                               'lazy_triplet_loss', 'huber_distance_loss')

    # Losses with eigenvalue term
    elif LOSS == 'pairwise_distance_neg_eigenvalue':  # Huber for positives, maximizing minimal ev for negatives
        loss_pos = pairwise_distance_loss(ops['outputs'][0], ops['outputs'][1], ops['distances'], d_max_squared,
                                          f_max_squared)
        loss_neg = neg_eigenvalue_loss(ops['outputs'][0], ops['outputs'][2])
        ops['loss'] = [loss_pos, loss_neg]

    elif LOSS == 'pairwise_huber_distance_neg_eigenvalue':  # Huber for positives, maximizing min ev for negatives
        loss_pos = pairwise_distance_loss(ops['outputs'][0], ops['outputs'][1], ops['distances'], d_max_squared,
                                          f_max_squared,
                                          distance_loss_name='huber_distance_loss')
        loss_neg = neg_eigenvalue_loss(ops['outputs'][0], ops['outputs'][2])
        ops['loss'] = [loss_pos, loss_neg]

    # Ntuplet losses
    elif LOSS == 'ntuplet_evmm':
        ops['loss'] = ntuplet_evmm_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)

    elif LOSS == 'ntuplet_trace':
        ops['loss'] = ntuplet_trace_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)

    # Residual losses
    elif LOSS == 'residual_det':
        ops['loss'] = residual_det_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)

    elif LOSS == 'residual_trace':
        ops['loss'] = residual_trace_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)

    elif LOSS == 'incremental_residual_det':
        ops['loss'], ops['loss_pca_in'] = incremental_residual_det_loss(ops['outputs'][0],
                                                                        ops['outputs'][1],
                                                                        ops['outputs'][2],
                                                                        MARGIN_1, ops['l_s'],
                                                                        ops['l_v'], ops['l_m'],
                                                                        ops['l_seen'], LOSS_DIM)

    elif LOSS == 'incremental_det':
        ops['loss'] = incremental_det_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1,
                                           ops['l_s'],
                                           ops['l_v'], ops['l_m'], ops['l_seen'], LOSS_DIM)
        ops['loss_pca_in'] = ops['output']

    elif LOSS == 'incremental_residual_mm':
        ops['loss'], ops['loss_pca_in'] = incremental_residual_mm_loss(ops['outputs'][0],
                                                                       ops['outputs'][1],
                                                                       ops['outputs'][2],
                                                                       MARGIN_1, ops['l_s'],
                                                                       ops['l_v'], ops['l_m'],
                                                                       ops['l_seen'], LOSS_DIM)

    elif LOSS == 'incremental_mm':
        ops['loss'] = incremental_mm_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1,
                                          ops['l_s'],
                                          ops['l_v'], ops['l_m'], ops['l_seen'], LOSS_DIM)
        ops['loss_pca_in'] = ops['output']

    elif LOSS == 'ms_loss':
        one_batch_classes = np.concatenate((np.zeros(1 + POSITIVES_PER_TUPLE), np.arange(NEGATIVES_PER_TUPLE) + 1))
        all_labels = one_batch_classes
        for batch in range(1, TUPLES_PER_BATCH):
            all_labels = np.concatenate((all_labels, one_batch_classes + batch * (NEGATIVES_PER_TUPLE + 1)))
        classes = tf.constant(all_labels)
        ops['loss'] = ms_loss(classes, ops['output'], ms_mining=MSMINING)

    elif LOSS == 'ms_sum':
        one_batch_classes = np.concatenate((np.zeros(1 + POSITIVES_PER_TUPLE), np.arange(NEGATIVES_PER_TUPLE) + 1))
        all_labels = one_batch_classes
        for batch in range(1, TUPLES_PER_BATCH):
            all_labels = np.concatenate((all_labels, one_batch_classes + batch * (NEGATIVES_PER_TUPLE + 1)))
        classes = tf.constant(all_labels)
        ms = ms_loss(classes, ops['output'],ms_mining=MSMINING)
        det = residual_det_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], MARGIN_1)
        ops['loss'] = ms*5.0 + det

    elif LOSS == 'swrd':
        ops['loss'] = swrd_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['pos_weights'], ops['neg_weights'], MARGIN_1)

    elif LOSS == 'wrd':
        ops['loss'] = wrd_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['pos_weights'], ops['neg_weights'], MARGIN_1)

    elif LOSS == 'prodwrd':
        ops['loss'] = prodwrd_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['pos_weights'], ops['neg_weights'], MARGIN_1)

    elif LOSS == 'sumwrd':
        ops['loss'] = sumwrd_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['pos_weights'], ops['neg_weights'], MARGIN_1)

    elif LOSS == 'wms':
        ops['loss'] = wms_loss(ops['distances'], ops['output'], d_alpha=ALPHA, d_beta=BETA, wfunction=WFUNCTION, sumfunction=SUMFUNCTION)

    elif LOSS == 'logratio':
        ops['loss'] = logratio_loss(ops['outputs'][0], ops['outputs'][1], ops['outputs'][2], ops['pos_distances'], ops['neg_distances'])


    # Define summaries
    if PN_LOSS:
        tf.summary.scalar('loss_pos', loss_pos)
        tf.summary.scalar('loss_neg', loss_neg)
    else:
        tf.summary.scalar('loss', ops['loss'])
    # Get training operator
    learning_rate = get_learning_rate(ops['epoch_num'])
    tf.summary.scalar('learning_rate', learning_rate)
    if OPTIMIZER == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if PN_LOSS:
        with tf.control_dependencies(update_ops):
            ops['train_op'] = [optimizer.minimize(ops['loss'][0]),
                               optimizer.minimize(ops['loss'][1], global_step=ops['step'])]
    else:
        with tf.control_dependencies(update_ops):
            ops['train_op'] = optimizer.minimize(ops['loss'], global_step=ops['step'])
    return ops, tuple_shape


def restore_weights():
    to_restore = {}
    for var in tf.trainable_variables():
        if 'vgg16_netvlad_pca' in var.name:
            # name_here = var.name
            saved_name = var._shared_name
            to_restore[saved_name] = var
            log(var)
        else:
            log('Newly initialized: {}'.format(var))
    restoration_saver = tf.train.Saver(to_restore)
    # Create a session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.gpu_options.polling_inactive_delay_msecs = 10
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    # Initialize a new model
    init = tf.global_variables_initializer()
    sess.run(init)
    restoration_saver.restore(sess, CHECKPOINT)
    return sess


def add_pca_vals(name, pca, vals):
    pos_sv = pca.singular_values_
    for i in range(len(pos_sv)):
        vals['{}_{}'.format(name, i)].append(pos_sv[i])
    vals['{}_min'.format(name)].append(np.min(pos_sv))
    vals['{}_max'.format(name)].append(np.max(pos_sv))
    vals['{}_sum'.format(name)].append(np.sum(pos_sv))


def train():
    global EPOCH
    global USED_IMAGE_LOCK
    global USED_IMAGES

    with tf.Graph().as_default() as graph:
        log("In Graph")

        ops, tuple_shape = build_model()
        sess = restore_weights()

        # Add summary writers
        ops['merged'] = tf.summary.merge_all()
        writers = dict()
        writers['local'] = tf.summary.FileWriter(os.path.join(OUT_DIR, 'local'), sess.graph)
        writers['other'] = tf.summary.FileWriter(os.path.join(OUT_DIR, 'other'))

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=MAX_TO_KEEP)
        epoch_saver = tf.train.Saver(max_to_keep=0)
        part_saver = tf.train.Saver(max_to_keep=0)

        # For better gpu utilization, tuple loading processes and gpu inference are done in separate threads.
        # Start CPU threads
        num_workers = 5

        for i in range(num_workers):
            worker = Thread(target=train_cpu_thread, args=[tuple_shape])
            worker.setDaemon(True)
            worker.start()
            worker = Thread(target=localization_cpu_thread)
            worker.setDaemon(True)
            worker.start()
            worker = Thread(target=eval_loss_cpu_thread, args=[tuple_shape])
            worker.setDaemon(True)
            worker.start()

            # PCA thread
            if 'incremental' in LOSS:
                worker = Thread(target=loss_pca_cpu_thread, args=[tuple_shape])
                worker.setDaemon(True)
                worker.start()

            # PCA thread
            if 'pca' in REDUCTION:
                worker = Thread(target=pca_cpu_thread, args=[tuple_shape])
                worker.setDaemon(True)
                worker.start()

        # Start GPU threads
        worker = Thread(target=localization_gpu_thread, args=(sess, ops))
        worker.setDaemon(True)
        worker.start()
        worker = Thread(target=eval_loss_gpu_thread, args=(sess, ops))
        worker.setDaemon(True)
        worker.start()
        worker = Thread(target=train_gpu_thread, args=(sess, ops, writers['local']))
        worker.setDaemon(True)
        worker.start()

        for epoch in range(MAX_EPOCH):
            log('**** EPOCH {} ****'.format(epoch))
            sys.stdout.flush()
            EPOCH = epoch
            with USED_IMAGE_LOCK:
                USED_IMAGES.clear()
            train_one_epoch(sess, epoch, writers, saver, part_saver, tuple_shape, ops)
            epoch_saver.save(sess, os.path.join(OUT_DIR, "epoch-checkpoint"), global_step=epoch)


def train_one_epoch(sess, epoch, writers, saver, part_saver, tuple_shape, ops):
    global GLOBAL_STEP
    global GLOBAL_STEP_LOCK
    global CACHED_FEATURE_LOCK
    global CACHED_FEATURE_INDICES
    global CACHED_FEATURES
    global CACHED_FEATURE_TREE
    global USED_IMAGE_LOCK
    global USED_IMAGES
    global REF_FEATURE_LOCK
    global REF_FEATURES
    global TRAIN_XY
    global TRAIN_XY_LOCK
    global FULL_FEATS

    train_meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(LOCAL_REF_SET, epoch)))
    train_xy = get_xy(train_meta)
    with TRAIN_XY_LOCK:
        TRAIN_XY = train_xy

    anchor_indices = np.array(
        load_csv(os.path.join(ANCHOR_ROOT, '{}_{}_{:03d}.csv'.format(LOCAL_REF_SET, TRAIN_REF_R, epoch)))['idx'],
        dtype=int)

    mining_count = 0
    for step in np.arange(len(anchor_indices), step=TUPLES_PER_BATCH):

        if step % MINING_STEP == 0:
            TRAIN_CPU_IN_QUEUE.join()
            TRAIN_GPU_IN_QUEUE.join()
            TRAIN_GPU_OUT_QUEUE.join()
            LOSS_PCA_IN_QUEUE.join()
            log('Caching features for hard negative mining.')

            mining_indices = np.arange(mining_count * MINING_CACHE_SIZE, (mining_count + 1) * MINING_CACHE_SIZE) % len(
                train_meta['t'])
            anchors_to_mine = np.array(anchor_indices[step:np.min([step + MINING_STEP, len(anchor_indices)])])
            mining_indices = np.concatenate([mining_indices, anchors_to_mine])
            num_to_mine = len(mining_indices)
            padding = np.zeros(
                TUPLES_PER_BATCH * sum(tuple_shape) - (num_to_mine % (TUPLES_PER_BATCH * sum(tuple_shape))),
                dtype=int)
            image_info = [(train_meta['date'][i], train_meta['folder'][i], train_meta['t'][i]) for i in
                          np.concatenate((mining_indices, padding))]
            with CACHED_FEATURE_LOCK:

                if REDUCTION == 'pca':
                    FULL_FEATS = True
                CACHED_FEATURES = np.array(extract_features(image_info, tuple_shape)[:num_to_mine])
                FULL_FEATS = False
                CACHED_FEATURE_INDICES = mining_indices

                if REDUCTION == 'pca':
                    if step == 0 and epoch == 0:  # Restart every epoch does not work...
                        [s, v, m, seen, true_seen, var] = skl_init(CACHED_FEATURES, OUT_DIM)
                        write_pca_globals(s, v, m, seen, true_seen, var)

                    else:

                        s, v, m, seen, true_seen, _ = read_pca_globals()
                        s, v, m, seen, true_seen, var \
                            = multiple_skl_increments(CACHED_FEATURES, TUPLES_PER_BATCH *
                                                      sum(tuple_shape), s, v, m, seen, true_seen, F)
                        write_pca_globals(s, v, m, seen, true_seen, var)

                    CACHED_FEATURES = np.matmul(CACHED_FEATURES - m, np.transpose(v))
                    CACHED_FEATURES /= np.sqrt(var)

                if step == 0 and epoch == 0:
                    if 'incremental' in LOSS:
                        if 'residual' in LOSS:
                            residuals = np.array([CACHED_FEATURES[i, :] - CACHED_FEATURES[j, :] for (i, j) in
                                                  rand_pairs(len(mining_indices), LOSS_DIM + 1)])
                            [s, v, m, seen, true_seen, var] = skl_init(residuals, LOSS_DIM)
                            write_loss_pca_globals(s, v, m, seen, true_seen, var)
                        else:
                            [s, v, m, seen, true_seen, var] = skl_init(CACHED_FEATURES, LOSS_DIM)
                            write_loss_pca_globals(s, v, m, seen, true_seen, var)

                CACHED_FEATURE_TREE = KDTree(CACHED_FEATURES)

            mining_count = mining_count + 1

        if step % EVAL_STEP == 0:
            TRAIN_CPU_IN_QUEUE.join()
            TRAIN_GPU_IN_QUEUE.join()
            TRAIN_GPU_OUT_QUEUE.join()
            LOSS_PCA_IN_QUEUE.join()
            log('EVALUATING')
            with GLOBAL_STEP_LOCK:
                global_step = GLOBAL_STEP  # Some steps produce invalid tuples, and are therefore skipped

            save_path = saver.save(sess, os.path.join(OUT_DIR, "checkpoint"), global_step=global_step)
            out_name = '{:02d}_{}'.format(epoch, os.path.basename(save_path))

            # Get loss for other region
            log('Calculating test loss.')
            get_eval_loss(global_step, writers['other'], epoch)

            # Test localization on other region
            evaluate_localization(global_step, OTHER_REF_SET, OTHER_QUERY_SET, 'other', out_name, tuple_shape,
                                  writers['other'], epoch)

            # Evaluate localization on training region
            evaluate_localization(global_step, LOCAL_REF_SET, LOCAL_QUERY_SET, 'local', out_name, tuple_shape,
                                  writers['local'], epoch)

        if step % SAVE_STEP == 0:
            TRAIN_CPU_IN_QUEUE.join()
            TRAIN_GPU_IN_QUEUE.join()
            TRAIN_GPU_OUT_QUEUE.join()
            LOSS_PCA_IN_QUEUE.join()
            with GLOBAL_STEP_LOCK:
                global_step = GLOBAL_STEP  # Some steps produce invalid tuples, and are therefore skipped
            log('Saving model.')
            part_saver.save(sess, os.path.join(OUT_DIR, "part-checkpoint"), global_step=global_step)

        # Train one step:
        TRAIN_CPU_IN_QUEUE.put(anchor_indices[step:step + TUPLES_PER_BATCH])

    # Finish training at end of epoch
    TRAIN_CPU_IN_QUEUE.join()
    TRAIN_GPU_IN_QUEUE.join()


def get_eval_loss(global_step, test_writer, epoch):
    meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(OTHER_REF_SET, epoch)))
    test_number = (global_step // EVAL_STEP)
    actual_num_eval_queries = (NUM_EVAL_QUERIES // TUPLES_PER_BATCH) * TUPLES_PER_BATCH
    test_indices = np.arange(test_number * actual_num_eval_queries, (test_number + 1) * actual_num_eval_queries) % len(
        meta['t'])

    batched_indices = np.reshape(test_indices, (-1, TUPLES_PER_BATCH))

    # Start queues
    for index_batch in batched_indices:
        EVAL_LOSS_CPU_IN_QUEUE.put(index_batch)

    # Wait for completion & order output
    EVAL_LOSS_CPU_IN_QUEUE.join()
    EVAL_LOSS_GPU_IN_QUEUE.join()

    eval_losses = list(EVAL_LOSS_GPU_OUT_QUEUE.queue)
    EVAL_LOSS_GPU_OUT_QUEUE.queue.clear()

    if len(eval_losses) > 0:
        summary = tf.Summary()
        if PN_LOSS:
            eval_losses = np.reshape(eval_losses, (-1, 2))
            loss_pos = np.mean(eval_losses[:, 0])
            loss_neg = np.mean(eval_losses[:, 1])

            summary.value.add(tag='loss_pos', simple_value=loss_pos)
            summary.value.add(tag='loss_neg', simple_value=loss_neg)
            log('Other region loss_pos: {}'.format(loss_pos))
            log('Other region loss_neg: {}'.format(loss_neg))
        else:
            loss = np.mean(eval_losses)
            summary.value.add(tag='loss', simple_value=loss)
            log('Other region loss: {}'.format(loss))
        test_writer.add_summary(summary, global_step)
    else:
        log('Evaluated but got no valid losses.')


def get_xy(meta):
    return np.array([[e, n] for e, n in zip(meta['easting'], meta['northing'])], dtype=float)


def evaluate_localization(global_step, ref_set_name, query_set_name, mode, out_name, tuple_shape, writer, epoch):
    # Get ref features
    ref_meta = load_csv(os.path.join(LOC_REF_ROOT, '{}_{}.csv'.format(ref_set_name, EVAL_REF_R)))
    num_ref = len(ref_meta['t'])
    padding = np.zeros(TUPLES_PER_BATCH * sum(tuple_shape) - (num_ref % (TUPLES_PER_BATCH * sum(tuple_shape))),
                       dtype=int)
    ref_image_info = [(ref_meta['date'][i], ref_meta['folder'][i], ref_meta['t'][i]) for i in
                      np.concatenate((np.arange(num_ref), padding))]
    ref_features = extract_features(ref_image_info, tuple_shape)
    ref_features = np.array(ref_features[0:num_ref])
    ref_image_info = ref_image_info[0:num_ref]
    ref_xy = get_xy(ref_meta)

    # Get query features
    query_meta = load_csv(os.path.join(SHUFFLED_ROOT, '{}_{:03d}.csv'.format(query_set_name, epoch)))
    test_number = (global_step // EVAL_STEP)
    query_indices = np.arange(test_number * NUM_EVAL_QUERIES, (test_number + 1) * NUM_EVAL_QUERIES) \
                    % len(query_meta['t'])
    padding = np.zeros(TUPLES_PER_BATCH * sum(tuple_shape) - (NUM_EVAL_QUERIES % (TUPLES_PER_BATCH * sum(tuple_shape))),
                       dtype=int)
    query_image_info = [(query_meta['date'][i], query_meta['folder'][i], query_meta['t'][i]) for i in
                        np.concatenate((query_indices, padding))]
    query_features = np.array(extract_features(query_image_info, tuple_shape))[:len(query_indices), :]
    query_xy = np.array([xy for i, xy in enumerate(get_xy(query_meta)) if i in query_indices])

    ref_feature_tree = KDTree(ref_features)
    nearest_latent_dists, nearest_latent_indices = ref_feature_tree.query(query_features, k=5)

    ref_xy_tree = KDTree(ref_xy)
    nearest_d_dist, nearest_d_indices = ref_xy_tree.query(query_xy, k=1)

    # CPU part of evaluation is done asynchronously
    worker = Thread(target=evaluate_localization_thread,
                    args=(global_step, mode, nearest_d_dist, nearest_d_indices, nearest_latent_indices, out_name,
                          query_image_info, query_xy, ref_image_info, ref_xy, writer))
    worker.setDaemon(True)
    worker.start()
    return


def extract_features(image_info, tuple_shape):
    num_to_extract = len(image_info)
    batched_indices = np.reshape(np.arange(num_to_extract), (-1, TUPLES_PER_BATCH * sum(tuple_shape)))
    batched_image_info = np.reshape(image_info, (-1, TUPLES_PER_BATCH * sum(tuple_shape), 3))

    for batch_indices, batch_image_info in zip(batched_indices, batched_image_info):
        LOCALIZATION_CPU_IN_QUEUE.put((batch_indices, batch_image_info))

    # Wait for completion & order output
    LOCALIZATION_CPU_IN_QUEUE.join()
    LOCALIZATION_GPU_IN_QUEUE.join()
    feature_pairs = list(LOCALIZATION_GPU_OUT_QUEUE.queue)
    LOCALIZATION_GPU_OUT_QUEUE.queue.clear()
    features = [[]] * num_to_extract
    for pair in feature_pairs:
        for i, f in zip(pair[0], pair[1]):
            features[i] = f
    return features


def create_array_job(loss, out_dir):
    run_one_job(script=__file__, queue='48h', cpu_only=False, memory=50,
                script_parameters=[('out_folder', os.path.basename(out_dir))], out_dir=out_dir,
                name='train_{}'.format(loss), overwrite=True, hold_off=False, array=True, num_jobs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--img_root',
                        default=os.path.join(fs_root(), 'datasets/oxford_512'))

    # Location of meta data (file lists & checkpoint)
    parser.add_argument('--shuffled_root', default=os.path.join(fs_root(), 'data/learnlarge/shuffled'))
    parser.add_argument('--loc_ref_root', default=os.path.join(fs_root(), 'data/learnlarge/clusters'))
    parser.add_argument('--anchor_root', default=os.path.join(fs_root(), 'data/learnlarge/anchors'))
    parser.add_argument('--checkpoint', default=os.path.join(fs_root(), 'data/learnlarge/checkpoint/offtheshelf'))

    # Output
    parser.add_argument('--out_root', default=os.path.join(srv_root(), 'mai_2020_logs'))
    parser.add_argument('--out_folder', default='')
    parser.add_argument('--max_to_keep', type=int, default=1)

    # Tuple size
    parser.add_argument('--positives_per_tuple', type=int, default=12,
                        help='Number of positives per training tuple.')
    parser.add_argument('--negatives_per_tuple', type=int, default=12,
                        help='Number of negatives per training tuple.')
    parser.add_argument('--hard_negatives_per_tuple', type=int, default=6,
                        help='Number of hard negatives per training tuple.')
    parser.add_argument('--hard_positives_per_tuple', type=int, default=6,
                        help='Number of hard negatives per training tuple.')
    parser.add_argument('--mutually_exclusive_negs', type=bool, default=True)

    # Loss
    parser.add_argument('--loss', default='wrd',
                        help='Options: triplet, pairwise_huber_distance_neg_eigenvalue')
    parser.add_argument('--margin_1', type=float, default=0.1, help='Margin for hinge loss.')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Margin for second hinge loss in quadruplet loss.')
    parser.add_argument('--lam', type=float, default=0.5, help='Scaling factor between loss components.')

    parser.add_argument('--alpha', type=float, default=0.8, help='Scaling factor between loss components.')
    parser.add_argument('--beta', type=float, default=15, help='Scaling factor between loss components.')
    parser.add_argument('--wfunction', default='exp', help='exp, lin, tanh')
    parser.add_argument('--sumfunction', default='ms', help='ms, plain')
    parser.add_argument('--msmining', type=bool, default=False)

    parser.add_argument('--max_pos_radius', type=float, default=15) # 10
    parser.add_argument('--min_neg_radius', type=float, default=15) # 25

    # Training
    parser.add_argument('--tuples_per_batch', type=int, default=2,
                        help='Tuples per training batch.')
    parser.add_argument('--max_epoch', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('--base_lr', type=float, default=float(5e-6), help='Initial learning rate.')
    parser.add_argument('--minimal_lr', type=float, default=float(5e-12))
    parser.add_argument('--lr_down_factor', type=float, default=0.5,
                        help='Reduce learning rate by lr_down_factor every lr_down_frequency epochs.')
    parser.add_argument('--lr_down_frequency', type=float, default=1,
                        help='Reduce learning rate by lr_down_factor every lr_down_frequency epochs.')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--optimizer', default='adam', help='Options: adam, momentum')

    # Output dimensionality reduction
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--loss_dim', default=512, type=int)
    parser.add_argument('--reduction', default='spp', help='none,[1:3]fc,pca')
    parser.add_argument('--vlad_cores', default=0, type=int)
    parser.add_argument('--L', default=3, type=int)
    parser.add_argument('--f', default=0.4, type=float, help='Forgetting factor for incremental PCA')

    # Hard negative mining
    parser.add_argument('--mining_step', type=int, default=250,
                        help='Number of samples for hard negative mining.')
    parser.add_argument('--mining_cache_size', type=int, default=1000,
                        help='Number of cached images for hard negative mining.\
                         Must be larger than tuples_per_batch*mining_step')

    # Validation loss and localization testing
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=500)
    parser.add_argument('--num_eval_queries', type=int, default=50)
    parser.add_argument('--eval_ref_r', default=5, type=int)

    # Data set names
    parser.add_argument('--local_ref_set', default='train_ref', help='Name of set to train with.')
    parser.add_argument('--local_query_set', default='train_query', help='Name of set to test local localization.')
    parser.add_argument('--other_ref_set', default='test_ref', help='Name of set with other region reference.')
    parser.add_argument('--other_query_set', default='test_query', help='Name of set with other region queries.')
    parser.add_argument('--train_ref_r', default=1, type=int)

    # Run on GPU
    parser.add_argument('--task_id', default=0)
    parser.add_argument('--gpu_id', default=0)

    FLAGS = parser.parse_args()
    flags_to_globals(FLAGS)

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    SUMFUNCTION = FLAGS.sumfunction
    WFUNCTION = FLAGS.wfunction
    ALPHA = FLAGS.alpha
    BETA = FLAGS.beta
    ANCHOR_ROOT = FLAGS.anchor_root
    BASE_LR = FLAGS.base_lr
    CHECKPOINT = FLAGS.checkpoint
    EVAL_REF_R = FLAGS.eval_ref_r
    EVAL_STEP = FLAGS.eval_step
    F = FLAGS.f
    GPU_ID = FLAGS.gpu_id
    HARD_NEGATIVES_PER_TUPLE = FLAGS.hard_negatives_per_tuple
    HARD_POSITIVES_PER_TUPLE = FLAGS.hard_positives_per_tuple
    IMG_ROOT = FLAGS.img_root
    LAM = FLAGS.lam
    LOC_REF_ROOT = FLAGS.loc_ref_root
    LOCAL_QUERY_SET = FLAGS.local_query_set
    LOCAL_REF_SET = FLAGS.local_ref_set
    LOSS = FLAGS.loss
    LR_DOWN_FACTOR = FLAGS.lr_down_factor
    LR_DOWN_FREQUENCY = FLAGS.lr_down_frequency
    MARGIN_1 = FLAGS.margin_1
    MARGIN_2 = FLAGS.margin_2
    MAX_EPOCH = FLAGS.max_epoch
    MAX_POS_RADIUS = FLAGS.max_pos_radius
    MAX_TO_KEEP = FLAGS.max_to_keep
    MIN_NEG_RADIUS = FLAGS.min_neg_radius
    MINIMAL_LR = FLAGS.minimal_lr
    MINING_CACHE_SIZE = FLAGS.mining_cache_size
    MINING_STEP = FLAGS.mining_step
    MOMENTUM = FLAGS.momentum
    MSMINING = FLAGS.msmining
    MUTUALLY_EXCLUSIVE_NEGS = FLAGS.mutually_exclusive_negs
    NEGATIVES_PER_TUPLE = FLAGS.negatives_per_tuple
    NUM_EVAL_QUERIES = FLAGS.num_eval_queries
    OPTIMIZER = FLAGS.optimizer
    OTHER_QUERY_SET = FLAGS.other_query_set
    OTHER_REF_SET = FLAGS.other_ref_set
    OUT_DIM = FLAGS.out_dim
    OUT_FOLDER = FLAGS.out_folder
    OUT_ROOT = FLAGS.out_root
    POSITIVES_PER_TUPLE = FLAGS.positives_per_tuple
    REDUCTION = FLAGS.reduction
    SAVE_STEP = FLAGS.save_step
    SHUFFLED_ROOT = FLAGS.shuffled_root
    TASK_ID = FLAGS.task_id
    TRAIN_REF_R = FLAGS.train_ref_r
    TUPLES_PER_BATCH = FLAGS.tuples_per_batch
    VLAD_CORES = FLAGS.vlad_cores
    LOSS_DIM = FLAGS.loss_dim
    L = FLAGS.L

    if location() == 'aws':
        os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(GPU_ID)

    if 'eigenvalue' in LOSS:
        PN_LOSS = True
    else:
        PN_LOSS = False

    if 'pairwise' in LOSS:
        DISTANCE_TYPE = 'pairwise'
    elif 'distance' in LOSS:
        DISTANCE_TYPE = 'anchor'
    elif 'swrd' in LOSS:
        DISTANCE_TYPE = 'swrd'
    elif 'wrd' in LOSS:
        DISTANCE_TYPE = 'wrd' # also includes prodwrd and sumwrd
    elif 'wms' in LOSS:
        DISTANCE_TYPE = 'wms'
    elif 'logratio' in LOSS:
        DISTANCE_TYPE = 'logratio'
    else:
        DISTANCE_TYPE = 'none'

    OUT_DIR = os.path.join(OUT_ROOT, OUT_FOLDER)
    if TASK_ID == -1 or debugging():
        k = 0
        while os.path.exists(OUT_DIR):
            OUT_FOLDER = '{}_{:03}'.format(LOSS, k)
            OUT_DIR = os.path.join(OUT_ROOT, OUT_FOLDER)
            k = k + 1
        os.makedirs(OUT_DIR)

    GLOBAL_STEP = 0
    GLOBAL_STEP_LOCK = Lock()

    FULL_FEATS = False

    CACHED_FEATURE_TREE = []
    CACHED_FEATURES = []
    CACHED_FEATURE_INDICES = []
    CACHED_FEATURE_LOCK = Lock()

    CACHED_FEATURE_TREE = []
    CACHED_FEATURES = []
    CACHED_FEATURE_INDICES = []
    CACHED_FEATURE_LOCK = Lock()

    REF_FEATURE_LOCK = Lock()
    REF_FEATURES = []

    EPOCH = 0

    PCA_LOCK = Lock()
    S = np.array([])
    V = np.array([])
    M = np.array([])
    SEEN = 0
    TRUE_SEEN = 0
    VAR = np.array([])

    LOSS_PCA_LOCK = Lock()
    LOSS_S = np.array([])
    LOSS_V = np.array([])
    LOSS_M = np.array([])
    LOSS_SEEN = 0
    LOSS_TRUE_SEEN = 0
    LOSS_VAR = np.array([])

    TRAIN_XY = []
    TRAIN_XY_LOCK = Lock()

    LOCALIZATION_CPU_IN_QUEUE = Queue(maxsize=0)
    LOCALIZATION_GPU_IN_QUEUE = Queue(maxsize=10)
    LOCALIZATION_GPU_OUT_QUEUE = Queue(maxsize=0)

    EVAL_LOSS_CPU_IN_QUEUE = Queue(maxsize=0)
    EVAL_LOSS_GPU_IN_QUEUE = Queue(maxsize=10)
    EVAL_LOSS_GPU_OUT_QUEUE = Queue(maxsize=0)

    TRAIN_CPU_IN_QUEUE = Queue(maxsize=0)
    TRAIN_GPU_IN_QUEUE = Queue(maxsize=10)
    TRAIN_GPU_OUT_QUEUE = Queue(maxsize=10)

    LOSS_PCA_IN_QUEUE = Queue(maxsize=10)

    USED_IMAGE_LOCK = Lock()
    USED_IMAGES = set([])

    LOG = open(os.path.join(OUT_DIR, 'train_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    # Make reproducible (and same condition for all loss functions)
    random.seed(42)
    np.random.seed(42)
    if TASK_ID == -1:
        create_array_job(LOSS, OUT_DIR)
    else:
        train()
    LOG.close()


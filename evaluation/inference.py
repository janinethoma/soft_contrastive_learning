import argparse
import os
import random
from datetime import datetime
from queue import Queue
from threading import Thread

import matplotlib
import numpy as np
import tensorflow as tf
from learnlarge.model.nets import vgg16Netvlad, vgg16
from learnlarge.util.cv import resize_img, standard_size
from learnlarge.util.helper import flags_to_globals
from learnlarge.util.helper import fs_root, location, srv_root
from learnlarge.util.io import load_img, load_csv, save_pickle
from learnlarge.util.sge import run_one_job
from time import time

matplotlib.use('Agg')


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def cpu_thread():
    global CPU_IN_QUEUE
    global GPU_IN_QUEUE
    while True:
        t = time()
        index, image_info = CPU_IN_QUEUE.get()
        images = load_images(image_info)
        GPU_IN_QUEUE.put((index, images), block=True)
        CPU_IN_QUEUE.task_done()
        print('Loaded images in {}s.'.format(time() - t))


def gpu_thread(sess, ops):
    global GPU_IN_QUEUE
    global GPU_OUT_QUEUE
    while True:
        t = time()
        index, img = GPU_IN_QUEUE.get()
        feat = sess.run(ops['output'], feed_dict={ops['input']: img})
        GPU_OUT_QUEUE.put((index, feat))
        GPU_IN_QUEUE.task_done()
        print('Inferred {} images in {}s.'.format(index, time() - t))


def load_images(img_path):
    images = [[]] * len(img_path)
    for i in range(len(img_path)):
 
        print(img_path[i])
        if 'oxs' in SET:
            img_path[i] = img_path[i].replace('.png', '.jpg')

        print(os.path.join(IMG_ROOT, img_path[i]))

        if 'achen' in SET:
            images[i] = standard_size(load_img(os.path.join(IMG_ROOT, img_path[i])), h=LARGE_SIDE,
                                      w=SMALL_SIDE)  # Portrait images
        elif VLAD_CORES > 0:
            if RESCALE:
                images[i] = resize_img(load_img(os.path.join(IMG_ROOT, img_path[i])), LARGE_SIDE)
            else:
                images[i] = load_img(os.path.join(IMG_ROOT, img_path[i]))
        else:
            images[i] = standard_size(load_img(os.path.join(IMG_ROOT, img_path[i])), h=SMALL_SIDE, w=LARGE_SIDE)
    return images


def build_inference_model():
    tuple_shape = [IMAGES_PER_PASS]
    ops = dict()

    if 'achen' in SET:
        ops['input'] = tf.placeholder(dtype=tf.float32,
                                      shape=[TUPLES_PER_BATCH * sum(tuple_shape), LARGE_SIDE, SMALL_SIDE, 3])
    elif VLAD_CORES > 0:
        ops['input'] = tf.placeholder(dtype=tf.float32, shape=[TUPLES_PER_BATCH * sum(tuple_shape), None, None, 3])
    else:
        ops['input'] = tf.placeholder(dtype=tf.float32,
                                      shape=[TUPLES_PER_BATCH * sum(tuple_shape), SMALL_SIDE, LARGE_SIDE, 3])
    ops['keep_prob'] = tf.placeholder_with_default(1.0, shape=())

    if VLAD_CORES == 64:
        ops['full_out'] = vgg16Netvlad(ops['input'])
    else:
        ops['full_out'] = tf.layers.flatten(vgg16(ops['input']))

    if REDUCTION == 'none' or REDUCTION == 'pca':  # Don't actually do PCA here - doing it after
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

    return ops, tuple_shape


def restore_weights():
    to_restore = {}
    for var in tf.trainable_variables():
        print(var.name)
        if var.name != 'Variable:0':
            # name_here = var.name
            saved_name = var._shared_name
            to_restore[saved_name] = var
            print(var)
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


def infer():
    with tf.Graph().as_default() as graph:
        print("In Graph")

        ops, tuple_shape = build_inference_model()
        sess = restore_weights()

        # For better gpu utilization, loading processes and gpu inference are done in separate threads.
        # Start CPU threads
        num_loader_threads = 6
        for i in range(num_loader_threads):
            worker = Thread(target=cpu_thread)
            worker.setDaemon(True)
            worker.start()

        # Start GPU threads
        worker = Thread(target=gpu_thread, args=(sess, ops))
        worker.setDaemon(True)
        worker.start()

        csv_file = os.path.join(CSV_ROOT, '{}.csv'.format(SET))
        meta = load_csv(csv_file)
        num = len(meta['path'])

        # Clean list
        padding = [0 for i in range(IMAGES_PER_PASS - (num % IMAGES_PER_PASS))]
        image_info = [(meta['path'][i]) for i in np.concatenate((np.arange(num), np.array(padding)))]
        padded_num = len(image_info)

        batched_indices = np.reshape(np.arange(padded_num), (-1, TUPLES_PER_BATCH * sum(tuple_shape)))
        batched_image_info = np.reshape(image_info, (-1, TUPLES_PER_BATCH * sum(tuple_shape)))

        for batch_indices, batch_image_info in zip(batched_indices, batched_image_info):
            CPU_IN_QUEUE.put((batch_indices, batch_image_info))

        # Wait for completion & order output
        CPU_IN_QUEUE.join()
        GPU_IN_QUEUE.join()
        feature_pairs = list(GPU_OUT_QUEUE.queue)
        GPU_OUT_QUEUE.queue.clear()
        features = [[]] * padded_num
        for pair in feature_pairs:
            for i, f in zip(pair[0], pair[1]):
                features[i] = f
        features = features[:num]
        save_pickle(features, os.path.join(OUT_ROOT, '{}_{}.pickle'.format(SET, OUT_NAME)))


def create_array_job(loss, out_dir):
    run_one_job(script=__file__, queue='48h', cpu_only=False, memory=25, out_dir=out_dir,
                name='infer_{}'.format(loss), overwrite=True, hold_off=False, array=True, num_jobs=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--rescale', default=True)
    parser.add_argument('--small_side', default=180, type=int)
    parser.add_argument('--large_side', default=240, type=int)
    parser.add_argument('--img_root', default=os.path.join(fs_root()))
    parser.add_argument('--csv_root', default=os.path.join(srv_root(), 'neurips/lists'))
    parser.add_argument('--set', default='cmu_ref')
    parser.add_argument('--checkpoint', default=os.path.join(fs_root(),
                                                             'aws_logs/neurips/loevil_quadruplet_ou512_repca_vl0_pca_neurips_002/epoch-checkpoint-2'))
    parser.add_argument('--out_name', default='loevil_quadruplet_ou512_repca_vl0_pca_neurips_002')

    # Dim reduction
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--reduction', default='none', help='none,[1:3]fc,pca')
    parser.add_argument('--vlad_cores', default=64, type=int)

    # Output
    parser.add_argument('--out_root',
                        default=os.path.join(fs_root(), 'data/meta_eval/lv'))
    parser.add_argument('--log_dir',
                        default=os.path.join(fs_root(), 'cpu_logs/learnlarge/meta_lv'.format(location())))

    # Tuple size
    parser.add_argument('--images_per_pass', type=int, default=4,
                        help='Number of images per forward pass.')

    # Run on GPU
    parser.add_argument('--task_id', default=0, type=int)

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    RESCALE = FLAGS.rescale
    SMALL_SIDE = FLAGS.small_side
    LARGE_SIDE = FLAGS.large_side
    CHECKPOINT = FLAGS.checkpoint
    CSV_ROOT = FLAGS.csv_root
    IMAGES_PER_PASS = FLAGS.images_per_pass
    IMG_ROOT = FLAGS.img_root
    LOG_DIR = FLAGS.log_dir
    OUT_DIM = FLAGS.out_dim
    OUT_NAME = FLAGS.out_name
    OUT_ROOT = FLAGS.out_root
    REDUCTION = FLAGS.reduction
    SET = FLAGS.set
    TASK_ID = FLAGS.task_id
    VLAD_CORES = FLAGS.vlad_cores

    TUPLES_PER_BATCH = 1

    CPU_IN_QUEUE = Queue(maxsize=0)
    GPU_IN_QUEUE = Queue(maxsize=10)
    GPU_OUT_QUEUE = Queue(maxsize=0)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'train_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    # Make reproducible (and same condition for all loss functions)
    random.seed(42)
    np.random.seed(42)
    if TASK_ID == -1:
        create_array_job('p-eval', LOG_DIR)
    else:
        infer()
    LOG.close()

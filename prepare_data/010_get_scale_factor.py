import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from learnlarge.util.helper import fs_root
from learnlarge.util.io import load_pickle, save_pickle, save_csv

matplotlib.use('Agg')

part_idx = 0

lv_file = os.path.join(fs_root(), 'data/learnlarge/lv/offtheshelf/train_ref/{}.pickle'.format(part_idx))
tuple_file = os.path.join(fs_root(), 'data/learnlarge/tuples/10000/train_ref_{}_10_25.pickle'.format(part_idx))
out_file = os.path.join(fs_root(), 'data/learnlarge/scale_factor/offtheshelf_train_ref_10000_{}_10_25.pickle'.format(
    part_idx))
out_file_meta = os.path.join(fs_root(), 'data/learnlarge/scale_factor/offtheshelf_train_ref_10000_{}_10_25.csv'.format(
    part_idx))
out_file_hist = os.path.join(fs_root(), 'data/learnlarge/scale_factor/offtheshelf_train_ref_10000_{}_10_25.png'.format(
    part_idx))

if not os.path.exists(out_file) or True:

    image_info, features, xy = load_pickle(lv_file)
    tuple_info = load_pickle(tuple_file)
    xy = np.array(xy)

    f_dists = []
    e_dists = []
    for i in tqdm(range(len(xy))):
        for j in tuple_info['positives'][i]:
            if j < i:
                f_dist = np.sum((features[i] - features[j]) ** 2)
                f_dists.append(f_dist)
                e_dist = np.sum((xy[i, :] - xy[j, :]) ** 2)
                e_dists.append(e_dist)

    save_pickle([e_dists, f_dists], out_file)

else:
    e_dists, f_dists = load_pickle(out_file)

full_info = dict()
full_info['f_mean'] = np.mean(f_dists)
full_info['e_mean'] = np.mean(e_dists)
full_info['f_med'] = np.median(f_dists)
full_info['e_med'] = np.median(e_dists)
full_info['f_max'] = np.max(f_dists)
full_info['e_max'] = np.max(e_dists)
save_csv(full_info, out_file_meta)

plt.clf()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(10)
f.set_figwidth(20)
ax1.hist(f_dists, bins=10000, histtype='step')
ax1.title.set_text('F dists')
ax2.hist(e_dists, bins=10000, histtype='step')
ax2.title.set_text('E dists')
plt.savefig(out_file_hist)

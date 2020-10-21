import bisect
from learnlarge.util.helper import srv_root, mkdir, flags_to_globals
from learnlarge.util.io import load_pickle, save_csv
from learnlarge.util.experiments import get_checkpoints
import os
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import matplotlib


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def log(output):
    print(output)
    LOG.write('{}\n'.format(output))
    LOG.flush()


def compile(l, d, code):
    mkdir(OUT_ROOT)
    top_n_root = os.path.join(srv_root(), 'neurips/top_n')

    queries = [
        'oxford_night',
        'oxford_overcast',
        'oxford_snow',

        'oxford_sunny',
        'pittsburgh_query',
    ]

    min_ys = [0, 40, 50, 50, 10]
    major_s = [10, 10, 10, 10, 10]
    minor_s = [2.5 if m == 10 else 1.0 for m in major_s]

    titles = [
        'Oxford RobotCar, night',
        'Oxford RobotCar, overcast',
        'Oxford RobotCar, snow',

        'Oxford RobotCar, sunny',
        'Pittsburgh',
    ]

    checkpoints = [

        '/scratch_net/tellur_third/user/efs/data/checkpoints/offtheshelf/offtheshelf/offtheshelf',
        '/scratch_net/tellur_third/user/efs/data/checkpoints/pittsburgh30/pittsnetvlad/vd16_pitts30k_conv5_3_vlad_preL2_intra_white',

        '/scratch_net/tellur_third/user/efs/cvpr_aws_logs/learnlarge/triplet_xy_000/epoch-checkpoint-2',
        '/scratch_net/tellur_third/user/efs/cvpr_aws_logs/learnlarge/quadruplet_xy_000/epoch-checkpoint-2',

        '/srv/beegfs02/scratch/toploc/data/mai_2020_logs/ha0_lolazy_triplet_muTrue_renone_vl64_pca_neurips_002/epoch-checkpoint-2',
        '/srv/beegfs02/scratch/toploc/data/mai_2020_logs/ha0_lolazy_quadruplet_muTrue_renone_vl64_pca_neurips_002/epoch-checkpoint-2',

        '/scratch_net/tellur_third/user/efs/home_logs/learnlarge_ral/huber_distance_triplet_xy_000/epoch-checkpoint-2',
        '/srv/beegfs02/scratch/toploc/data/mai_2020_logs/ha0_lologratio_ma15_mi15_muTrue_renone_tu1_vl64_pca_neurips_002/epoch-checkpoint-1',

        '/srv/beegfs02/scratch/toploc/data/mai_2020_logs/ha0_loms_loss_msTrue_muTrue_renone_tu1_vl64_pca_neurips_001/epoch-checkpoint-0',
        '/srv/beegfs02/scratch/toploc/data/mai_2020_logs/al0.8_be15_ha0_lowms_ma15_mi15_msTrue_muTrue_renone_tu1_vl64_pca_neurips_000/epoch-checkpoint-0',
    ]

    fill_styles = [
        'none',
        'none',
        'none',
        'none',
        'none',
        'none',
        'none',
        'none',
        'none',
        'full',
    ]

    markers = [
        '',
        "^",
        "^",
        "s",
        "^",
        "s",
        "^",
        'v',
        "o",
        "d",
    ]

    losses = [

        'Off-the-shelf \\cite{arandjelovic2016netvlad}',
        'Triplet trained on Pittsburgh \\cite{arandjelovic2016netvlad}',

        'Triplet \\cite{arandjelovic2016netvlad}',
        'Quadruplet \\cite{chen2017beyond}',

        'Lazy triplet \\cite{angelina2018pointnetvlad}',
        'Lazy quadruplet \\cite{angelina2018pointnetvlad}',

        'Trip.~+ Huber dist. \\cite{thoma2020geometrically}',
        'Log-ratio \\cite{kim2019deep}',

        'Multi-similarity \\cite{wang2019multi}',
        'Ours',
    ]

    lines = [
        ':',
        ':',
        '--',
        '--',
        '-.',
        '-.',
        '--',
        '-.',
        '--',
        '-',
    ]



    colors = [
        '#000000',

        '#ff6b1c', 

        '#f03577',

        '#5f396b',

        '#1934e6', 

        '#0e6606',


        '#B0C4DE',
        '#990000',
        '#663300',
        '#11d194',

    ]

    setting = 'l{}_dim{}'.format(l, d)
    print(setting)

    rows = 2
    cols = 3

    f, axs = plt.subplots(rows, cols, constrained_layout=False)
    if rows == 1:
        axs = np.expand_dims(axs, 0)
    if cols == 1:
        axs = np.expand_dims(axs, 1)
    f.tight_layout()
    f.set_figheight(8)  # 8.875in textheight
    f.set_figwidth(10)  # 6.875in textwidth

    for i, query in enumerate(queries):
        print(query)

        print_gt = True

        t = 25.0
        l = 0.0
        out_setting = 'l{}_dim{}'.format(l, d)

        setting = 'l{}_dim{}'.format(l, d)

        min_y = 1000
        max_y = 0

        for j, (checkpoint,loss,  m, line, color) in enumerate(
                zip(checkpoints, losses, cycle(markers), cycle(lines), cycle(colors))):

            cp_name = checkpoint.split('/')[-2]
            cp_name = ''.join(os.path.basename(cp_name).split('.'))  # Removing '.'
            cp_name += '_e{}'.format(checkpoint[-1])

            t_n_file = os.path.join(top_n_root, setting, '{}_{}.pickle'.format(query, cp_name))
            if not os.path.exists(t_n_file):
                print('Missing: {}'.format(t_n_file))
                continue
            print(t_n_file)

            [top_i, top_g_dists, top_f_dists, gt_i, gt_g_dist, ref_idx] = load_pickle(t_n_file)
            top_g_dists = np.array(top_g_dists)

            if print_gt:
                print_gt = False
                X = np.linspace(0, t, num=50)
                Y = [float(sum(gt_g_dist < x)) / float(len(gt_g_dist)) * 100 for x in X]
                ax = axs[i % rows, i // rows]
                # ax = axs
                width = 0.75

                ax.plot(X, Y, label='Upper bound', linewidth=width, c='#000000')
                ax.title.set_text(titles[i])
                ax.set_xlim([0, t])
                ax.grid(True)

                x_min = X[bisect.bisect(Y, min_ys[i])]

            t_1_d = np.array([td[0] for td in top_g_dists])
            X = np.linspace(0, t, num=50)

            Y = [float(sum(t_1_d < x)) / float(len(t_1_d)) * 100 for x in X]

            min_y = min(np.min(np.array(Y)), min_y)
            max_y = max(np.max(np.array(Y)), max_y)

            ax = axs[i % rows, i // rows]
            width = 0.75
            ax.plot(X, Y, label=loss, linestyle=line, marker=m, linewidth=width, markevery=j % rows + cols,
                   c=color, markersize=3, fillstyle=fill_styles[j % len(fill_styles)])

            #ax.plot(X, Y, label=cp_name)

        ax = axs[i % rows, i // rows]
        ax.set_xlim([x_min, t])
        ax.set_ylim([min_ys[i], min(max_y + 5, 100)])

        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(x_min // (t / 5) * (t / 5), t, t / 5)[1:]
        minor_ticks_x = np.arange(x_min // (t / 5 / 4) * (t / 5 / 4), t, t / 5 / 4)[1:]

        y_step = 10

        major_ticks_y = np.arange(min_ys[i], min(max_y + 5, 100), major_s[i])
        minor_ticks_y = np.arange(min_ys[i], min(max_y + 5, 100), minor_s[i])

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    out_setting = out_setting.replace('.', '')
    out_name = os.path.join(OUT_ROOT, '{}_neurips_roc.pdf'.format(out_setting))
   
    axs[-1, -1].axis('off')

    for i in range(cols):
        axs[-1, i].set_xlabel('Distance threshold $d$ [m]')

    for i in range(rows):
        axs[i, 0].set_ylabel('Correctly localized [\%]')

    handles, labels = axs[0, 0].get_legend_handles_labels()


    left = 0.0  # the left side of the subplots of the figure
    right = 1.0  # the right side of the subplots of the figure
    bottom = 0.23  # the bottom of the subplots of the figure
    top = 1.0  # the top of the subplots of the figure
    wspace = 0.2  # the amount of width reserved for space between subplots,
    # expressed as a fraction of the average axis width
    hspace = 0.2  # the amount of height reserved for space between subplots,
    # expressed as a fraction of the average axis height

    # space = 0.2
    plt.subplots_adjust(wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)

    axs[-1, -1].legend(handles, labels, bbox_to_anchor=(0.0, 0.5), loc='center left',
                     ncol=1, borderaxespad=0., frameon=True, fontsize='medium')  # mode="expand",

    plt.savefig(out_name, bbox_inches='tight',
                pad_inches=0)

    plt.savefig(out_name.replace('.pdf', '.pgf'), bbox_inches='tight',
                pad_inches=0)

    # Test
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image folder
    parser.add_argument('--l',
                        default='0.0', type=float)
    parser.add_argument('--d',
                        default='256', type=int)
    parser.add_argument('--checkpoints',
                        default='residual')
    parser.add_argument('--log_dir', default=os.path.join(srv_root(), 'neurips', 'logs', 'roc'))
    parser.add_argument('--out_root', default='/home/user/scp_out')

    FLAGS = parser.parse_args()

    # Define each FLAG as a variable (generated automatically with util.flags_to_globals(FLAGS))
    flags_to_globals(FLAGS)

    LOG_DIR = FLAGS.log_dir
    OUT_ROOT = FLAGS.out_root

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(OUT_ROOT):
        os.makedirs(OUT_ROOT)

    LOG = open(os.path.join(LOG_DIR, 'top_n_log.txt'), 'a')
    log('Running {} at {}.'.format(__file__, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    log(FLAGS)

    compile(FLAGS.l, FLAGS.d, FLAGS.checkpoints)

    LOG.close()
